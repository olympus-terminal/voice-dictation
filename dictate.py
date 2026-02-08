#!/usr/bin/env python3
"""
Voxtral Dictation - Real-time voice-to-text.

Supports both local (faster-whisper) and API (Voxtral) transcription.

Usage:
    python dictate.py              # Start with push-to-talk (Ctrl+Shift+Space)
    python dictate.py --local      # Use local faster-whisper (no API needed)
    python dictate.py --toggle     # Use toggle mode (Ctrl+Shift+R to start/stop)
    python dictate.py --clipboard  # Output to clipboard instead of typing
    python dictate.py --test-mic   # Test microphone capture
    python dictate.py --test-api   # Test API connection
"""
import asyncio
import argparse
import signal
import sys
import threading
from typing import Optional

from config import get_config, OutputConfig
from audio_capture import AudioCapture
from transcriber import create_transcriber, VoxtralTranscriber
from transcriber_local import create_local_transcriber, WHISPER_AVAILABLE
from text_output import TextOutput, TextProcessor, OutputMode
from tray_icon import create_indicator, TrayStatus

try:
    from pynput import keyboard
    PYNPUT_AVAILABLE = True
except ImportError:
    PYNPUT_AVAILABLE = False
    print("pynput not installed. Run: pip install pynput")


class DictationApp:
    """
    Main dictation application.
    Coordinates audio capture, transcription, and text output.
    """

    def __init__(
        self,
        output_mode: OutputMode = OutputMode.TYPE,
        use_toggle: bool = False,
        use_mock: bool = False,
        use_local: bool = False,
        local_model: str = "base",
        use_tray: bool = True,
    ):
        self.config = get_config()
        self.use_toggle = use_toggle
        self.use_mock = use_mock
        self.use_local = use_local

        # Components
        self.audio = AudioCapture(
            sample_rate=self.config.audio.sample_rate,
            channels=self.config.audio.channels,
            chunk_duration_ms=self.config.audio.chunk_duration_ms,
            device_name=self.config.audio.device_name,
        )

        # Choose transcriber
        if use_local:
            print(f"Using local transcriber (faster-whisper {local_model})")
            self.transcriber = create_local_transcriber(
                model_size=local_model,
                language=self.config.voxtral.language,
                sample_rate=self.config.audio.sample_rate,
            )
        else:
            self.transcriber = create_transcriber(
                api_key=self.config.voxtral.api_key,
                model=self.config.voxtral.model,
                sample_rate=self.config.audio.sample_rate,
                use_mock=use_mock,
            )

        self.output = TextOutput(
            mode=output_mode,
            typing_delay_ms=self.config.output.typing_delay_ms,
        )

        self.processor = TextProcessor(
            auto_capitalize=self.config.output.auto_capitalize,
            auto_punctuation=self.config.output.auto_punctuation,
        )

        # Status indicator (tray icon or console)
        self.indicator = create_indicator(use_tray=use_tray)

        # Set up audio level callback
        self.audio.set_level_callback(self._on_audio_level)

        # State
        self._is_recording = False
        self._is_running = False
        self._recording_task: Optional[asyncio.Task] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._keyboard_listener: Optional[keyboard.Listener] = None

        # Hotkey parsing
        self._ptt_keys = self._parse_hotkey(self.config.hotkeys.push_to_talk)
        self._toggle_keys = self._parse_hotkey(self.config.hotkeys.toggle_record)
        self._pressed_keys: set = set()

    def _on_audio_level(self, level: float):
        """Handle audio level updates."""
        self.indicator.set_audio_level(level)

    def _parse_hotkey(self, hotkey_str: str) -> set:
        """Parse hotkey string like 'ctrl+shift+space' into key set."""
        keys = set()
        for part in hotkey_str.lower().split("+"):
            part = part.strip()
            if part == "ctrl":
                keys.add(keyboard.Key.ctrl_l)
                keys.add(keyboard.Key.ctrl_r)
            elif part == "shift":
                keys.add(keyboard.Key.shift_l)
                keys.add(keyboard.Key.shift_r)
            elif part == "alt":
                keys.add(keyboard.Key.alt_l)
                keys.add(keyboard.Key.alt_r)
            elif part == "space":
                keys.add(keyboard.Key.space)
            elif part == "escape":
                keys.add(keyboard.Key.esc)
            elif len(part) == 1:
                keys.add(keyboard.KeyCode.from_char(part))
            else:
                # Try as named key
                try:
                    keys.add(getattr(keyboard.Key, part))
                except AttributeError:
                    print(f"Unknown key: {part}")
        return keys

    def _check_hotkey(self, target_keys: set) -> bool:
        """Check if target hotkey combination is pressed."""
        # Need at least one modifier and one regular key
        modifiers_pressed = any(
            k in self._pressed_keys
            for k in [
                keyboard.Key.ctrl_l, keyboard.Key.ctrl_r,
                keyboard.Key.shift_l, keyboard.Key.shift_r,
                keyboard.Key.alt_l, keyboard.Key.alt_r,
            ]
        )

        # Check if all required keys are pressed (accounting for left/right variants)
        ctrl_needed = keyboard.Key.ctrl_l in target_keys or keyboard.Key.ctrl_r in target_keys
        shift_needed = keyboard.Key.shift_l in target_keys or keyboard.Key.shift_r in target_keys

        ctrl_ok = not ctrl_needed or (
            keyboard.Key.ctrl_l in self._pressed_keys or
            keyboard.Key.ctrl_r in self._pressed_keys
        )
        shift_ok = not shift_needed or (
            keyboard.Key.shift_l in self._pressed_keys or
            keyboard.Key.shift_r in self._pressed_keys
        )

        # Check for the main key (space, etc.)
        main_keys = target_keys - {
            keyboard.Key.ctrl_l, keyboard.Key.ctrl_r,
            keyboard.Key.shift_l, keyboard.Key.shift_r,
            keyboard.Key.alt_l, keyboard.Key.alt_r,
        }
        main_ok = any(k in self._pressed_keys for k in main_keys)

        return ctrl_ok and shift_ok and main_ok

    def _on_key_press(self, key):
        """Handle key press events."""
        self._pressed_keys.add(key)

        if self.use_toggle:
            # Toggle mode: press hotkey to start/stop
            if self._check_hotkey(self._toggle_keys):
                if self._is_recording:
                    self._stop_recording()
                else:
                    self._start_recording()
        else:
            # Push-to-talk mode: hold to record
            if not self._is_recording and self._check_hotkey(self._ptt_keys):
                self._start_recording()

    def _on_key_release(self, key):
        """Handle key release events."""
        self._pressed_keys.discard(key)

        if not self.use_toggle:
            # Push-to-talk: stop when key released
            if self._is_recording and not self._check_hotkey(self._ptt_keys):
                self._stop_recording()

        # Escape always cancels
        if key == keyboard.Key.esc and self._is_recording:
            self._stop_recording()

    def _start_recording(self):
        """Start recording and transcription."""
        if self._is_recording:
            return

        self._is_recording = True
        self.processor.reset()
        self.indicator.set_status(TrayStatus.RECORDING)
        print("\n🎤 Recording... ", end="", flush=True)

        # Start audio capture
        self.audio.start()

        # Start transcription in async context
        if self._loop:
            self._recording_task = asyncio.run_coroutine_threadsafe(
                self._transcribe_loop(),
                self._loop
            )

    def _stop_recording(self):
        """Stop recording."""
        if not self._is_recording:
            return

        self._is_recording = False
        self.indicator.set_status(TrayStatus.PROCESSING)
        self.audio.stop()
        print(" Done\n")
        self.indicator.set_status(TrayStatus.IDLE)

    async def _transcribe_loop(self):
        """Main transcription loop."""
        try:
            # Set up text callback
            def on_text(text: str, is_final: bool):
                if text:
                    processed = self.processor.process(text)
                    self.output.output(processed)
                    if not is_final:
                        print(f"{processed}", end="", flush=True)
                    else:
                        print()

            self.transcriber.set_text_callback(on_text)

            # Stream audio to transcriber
            async for result in self.transcriber.transcribe_stream(
                self.audio.iter_chunks()
            ):
                if not self._is_recording:
                    break

        except Exception as e:
            print(f"\nTranscription error: {e}")

    def _setup_keyboard(self):
        """Set up global keyboard listener."""
        if not PYNPUT_AVAILABLE:
            print("Keyboard hooks not available without pynput")
            return

        self._keyboard_listener = keyboard.Listener(
            on_press=self._on_key_press,
            on_release=self._on_key_release,
        )
        self._keyboard_listener.start()

    async def run(self):
        """Main application loop."""
        self._is_running = True
        self._loop = asyncio.get_event_loop()

        # Start status indicator
        self.indicator.start()
        self.indicator.set_status(TrayStatus.IDLE)

        # Set up keyboard listener
        self._setup_keyboard()

        # Display instructions
        if self.use_toggle:
            print(f"Press {self.config.hotkeys.toggle_record} to start/stop recording")
        else:
            print(f"Hold {self.config.hotkeys.push_to_talk} to record (push-to-talk)")
        print("Press Escape to cancel, Ctrl+C to quit\n")

        # Set up signal handlers
        def signal_handler(sig, frame):
            print("\nShutting down...")
            self._is_running = False

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        # Main loop
        try:
            while self._is_running:
                await asyncio.sleep(0.1)
        finally:
            self.cleanup()

    def cleanup(self):
        """Clean up resources."""
        if self._is_recording:
            self._stop_recording()

        if self._keyboard_listener:
            self._keyboard_listener.stop()

        self.indicator.stop()
        self.audio.close()


def test_microphone():
    """Test microphone capture."""
    from audio_capture import test_microphone as run_mic_test
    config = get_config()
    run_mic_test(duration_seconds=3.0, device_name=config.audio.device_name)


async def test_api():
    """Test API connection with a short recording."""
    config = get_config()

    if not config.voxtral.api_key:
        print("Error: MISTRAL_API_KEY not set")
        print("Set it with: export MISTRAL_API_KEY='your-key-here'")
        return

    print("Testing API connection...")
    print("Recording 3 seconds of audio...")

    audio = AudioCapture(
        sample_rate=config.audio.sample_rate,
        device_name=config.audio.device_name,
    )

    # Record audio
    audio.start()
    frames = []
    for _ in range(30):  # 3 seconds at 100ms chunks
        chunk = audio.read_chunk()
        if chunk:
            frames.append(chunk)
    audio.stop()
    audio.close()

    audio_data = b"".join(frames)
    print(f"Recorded {len(audio_data)} bytes")

    # Transcribe
    print("Sending to Voxtral API...")
    transcriber = create_transcriber(
        api_key=config.voxtral.api_key,
        sample_rate=config.audio.sample_rate,
    )

    text = await transcriber.transcribe_audio(audio_data)
    print(f"\nTranscription: {text}")


def main():
    parser = argparse.ArgumentParser(
        description="Voxtral Dictation - Real-time voice-to-text"
    )
    parser.add_argument(
        "--toggle", "-t",
        action="store_true",
        help="Use toggle mode instead of push-to-talk"
    )
    parser.add_argument(
        "--clipboard", "-c",
        action="store_true",
        help="Output to clipboard instead of typing"
    )
    parser.add_argument(
        "--both", "-b",
        action="store_true",
        help="Output to both typing and clipboard"
    )
    parser.add_argument(
        "--test-mic",
        action="store_true",
        help="Test microphone capture"
    )
    parser.add_argument(
        "--test-api",
        action="store_true",
        help="Test API connection"
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock transcriber (for testing without API)"
    )
    parser.add_argument(
        "--local", "-l",
        action="store_true",
        help="Use local faster-whisper instead of Voxtral API"
    )
    parser.add_argument(
        "--model", "-m",
        default="base",
        choices=["tiny", "base", "small", "medium", "large-v3"],
        help="Whisper model size for local transcription (default: base)"
    )
    parser.add_argument(
        "--no-tray",
        action="store_true",
        help="Disable system tray icon (use console indicator)"
    )

    args = parser.parse_args()

    # Handle test modes
    if args.test_mic:
        test_microphone()
        return

    if args.test_api:
        asyncio.run(test_api())
        return

    # Determine output mode
    if args.both:
        output_mode = OutputMode.BOTH
    elif args.clipboard:
        output_mode = OutputMode.CLIPBOARD
    else:
        output_mode = OutputMode.TYPE

    # Create and run app
    app = DictationApp(
        output_mode=output_mode,
        use_toggle=args.toggle,
        use_mock=args.mock,
        use_local=args.local,
        local_model=args.model,
        use_tray=not args.no_tray,
    )

    print("=" * 50)
    if args.local:
        print("  Voice Dictation (Local)")
        print(f"  Using faster-whisper {args.model}")
    else:
        print("  Voxtral Dictation")
        print("  Real-time voice-to-text with Mistral AI")
    print("=" * 50)
    print()

    asyncio.run(app.run())


if __name__ == "__main__":
    main()
