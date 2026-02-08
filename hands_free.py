#!/usr/bin/env python3
"""
Hands-free continuous dictation mode.
Automatically detects speech and transcribes without any keyboard input.

Usage:
    ./run.sh --hands-free           # Continuous dictation with local whisper
    ./run.sh --hands-free -m small  # Use small model for better accuracy
    ./run.sh --hands-free -c        # Output to clipboard instead of typing
"""
import asyncio
import signal
import sys
import threading
from typing import Optional
from collections import deque

from config import get_config
from audio_capture import AudioCapture
from transcriber_local import create_local_transcriber, LocalTranscriber
from text_output import TextOutput, TextProcessor, OutputMode
from tray_icon import create_indicator, TrayStatus
from vad import VoiceActivityDetector, VADState
from voice_commands import VoiceCommandProcessor, CommandExecutor, CommandAction
from homonym_fixer import HomonymFixer, RuleBasedFixer, create_fixer


class HandsFreeDictation:
    """
    Hands-free dictation that automatically detects speech.

    Flow:
    1. Continuously monitor audio with VAD
    2. When speech detected, start buffering audio
    3. When speech ends, transcribe the buffered audio
    4. Type the result into the focused window
    """

    def __init__(
        self,
        output_mode: OutputMode = OutputMode.TYPE,
        model_size: str = "base",
        language: str = "en",
        use_tray: bool = True,
        enable_commands: bool = True,
        fix_homonyms: bool = False,
        homonym_llm: bool = True,
        device_name: Optional[str] = None,
    ):
        self.config = get_config()

        # Audio capture (CLI --device overrides config)
        mic_device = device_name or self.config.audio.device_name
        self.audio = AudioCapture(
            sample_rate=self.config.audio.sample_rate,
            channels=self.config.audio.channels,
            chunk_duration_ms=self.config.audio.chunk_duration_ms,
            device_name=mic_device,
        )

        # Voice activity detection
        self.vad = VoiceActivityDetector(
            sample_rate=self.config.audio.sample_rate,
            min_speech_ms=200,
            min_silence_ms=600,
            trailing_silence_ms=1200,
        )

        # Local transcriber
        print(f"Loading faster-whisper {model_size}...")
        self.transcriber = create_local_transcriber(
            model_size=model_size,
            language=language,
            sample_rate=self.config.audio.sample_rate,
        )

        # Text output
        self.output = TextOutput(
            mode=output_mode,
            typing_delay_ms=0,
        )

        self.processor = TextProcessor(
            auto_capitalize=True,
            auto_punctuation=True,
        )

        # Status indicator
        self.indicator = create_indicator(use_tray=use_tray)

        # Voice commands (optional)
        self.enable_commands = enable_commands
        if enable_commands:
            self.command_processor = VoiceCommandProcessor()
            self.command_executor = CommandExecutor(self.output)
        else:
            self.command_processor = None
            self.command_executor = None

        # Homonym fixer (optional)
        self.fix_homonyms = fix_homonyms
        if fix_homonyms:
            self.homonym_fixer = create_fixer(use_llm=homonym_llm)
            fixer_type = "LLM" if homonym_llm else "rule-based"
            print(f"Homonym correction: {fixer_type}")
        else:
            self.homonym_fixer = None

        # State
        self._is_running = False
        self._audio_buffer = bytearray()
        self._is_transcribing = False
        self._utterance_count = 0
        self._paused = False

        # Set up VAD callbacks
        self.vad.set_callbacks(
            on_speech_start=self._on_speech_start,
            on_speech_end=self._on_speech_end,
            on_level=self._on_level,
        )

    def _on_speech_start(self):
        """Called when VAD detects speech start."""
        self._audio_buffer.clear()
        self.indicator.set_status(TrayStatus.RECORDING)
        print("\n🎤 Listening...", end="", flush=True)

    def _on_speech_end(self):
        """Called when VAD detects speech end."""
        audio_bytes = len(self._audio_buffer)
        audio_seconds = audio_bytes / 2 / 16000  # 16-bit mono at 16kHz

        if audio_bytes < 8000:  # Less than 250ms at 16kHz
            print(f" (too short: {audio_seconds:.1f}s)", flush=True)
            self.indicator.set_status(TrayStatus.IDLE)
            return

        self._utterance_count += 1
        audio_data = bytes(self._audio_buffer)
        self._audio_buffer.clear()

        # Transcribe
        self.indicator.set_status(TrayStatus.PROCESSING)
        print(f" ({audio_seconds:.1f}s) transcribing...", end="", flush=True)

        # Run transcription
        self._transcribe_and_output(audio_data)

    def _on_level(self, level: float, is_speech: bool):
        """Called with audio level updates."""
        self.indicator.set_audio_level(level)

    def _transcribe_and_output(self, audio_data: bytes):
        """Transcribe audio and output the text."""
        try:
            text = self.transcriber.transcribe_audio(audio_data)

            if text and text.strip():
                text = text.strip()

                # Apply homonym correction (if enabled)
                if self.fix_homonyms and self.homonym_fixer:
                    result = self.homonym_fixer.fix(text)
                    if result.changed:
                        print(f" [homonyms: {result.corrections}]", end="")
                        text = result.corrected

                # Check for voice commands (if enabled)
                if self.enable_commands and self.command_processor:
                    command, remaining_text = self.command_processor.process(text)

                    if command:
                        # Handle special mode commands
                        if command.action == CommandAction.STOP_LISTENING:
                            print(f" \"{text}\" -> [STOP LISTENING]")
                            self._is_running = False
                            return

                        if command.action == CommandAction.PAUSE:
                            print(f" \"{text}\" -> [PAUSED]")
                            self._paused = True
                            return

                        if command.action == CommandAction.CAPS_ON:
                            self.command_processor.set_caps_mode(True)
                            print(f" \"{text}\" -> [CAPS ON]")
                        elif command.action == CommandAction.CAPS_OFF:
                            self.command_processor.set_caps_mode(False)
                            print(f" \"{text}\" -> [CAPS OFF]")
                        else:
                            # Execute the command
                            print(f" \"{text}\"", end="")
                            self.command_executor.execute(command)

                        # Type any remaining text after the command
                        if remaining_text:
                            processed = self.processor.process(remaining_text)
                            processed = self.command_processor.apply_formatting(processed)
                            if processed and not processed[0].isupper():
                                processed = " " + processed
                            self.output.output(processed)
                            print(f" + \"{processed}\"")

                        self.indicator.set_status(TrayStatus.IDLE)
                        return

                # No command (or commands disabled) - just type the text
                processed = self.processor.process(text)

                # Add space before if not starting sentence
                if processed and not processed[0].isupper():
                    processed = " " + processed

                print(f" \"{processed}\"")
                self.output.output(processed)
            else:
                print(" (no speech detected)")

        except Exception as e:
            print(f" error: {e}")

        self.indicator.set_status(TrayStatus.IDLE)

    async def run(self):
        """Main loop - continuously listen and transcribe."""
        self._is_running = True

        # Start components
        self.indicator.start()
        self.indicator.set_status(TrayStatus.IDLE)
        self.audio.start()

        print("=" * 50)
        print("  Hands-Free Dictation")
        print("  Speak naturally - text appears where you type")
        if self.enable_commands:
            print("  Voice commands: ON (say 'computer <command>')")
        else:
            print("  Voice commands: OFF (pure dictation)")
        if self.fix_homonyms:
            fixer_type = "LLM" if isinstance(self.homonym_fixer, HomonymFixer) else "rules"
            print(f"  Homonym correction: ON ({fixer_type})")
        print("=" * 50)
        print()
        print("Calibrating ambient noise (1 second, stay quiet)...")
        print()

        # Set up signal handler
        def signal_handler(sig, frame):
            print("\n\nShutting down...")
            self._is_running = False

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        try:
            while self._is_running:
                chunk = self.audio.read_chunk()
                if chunk:
                    # Process through VAD
                    is_speaking = self.vad.process_chunk(chunk)

                    # Buffer audio while speaking
                    if is_speaking:
                        self._audio_buffer.extend(chunk)

                await asyncio.sleep(0.001)

        finally:
            self.cleanup()

    def cleanup(self):
        """Clean up resources."""
        self.audio.stop()
        self.audio.close()
        self.indicator.stop()
        print(f"\nTotal utterances: {self._utterance_count}")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Hands-free voice dictation"
    )
    parser.add_argument(
        "--model", "-m",
        default="base",
        choices=["tiny", "base", "small", "medium", "large-v3"],
        help="Whisper model size (default: base)"
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
        "--no-tray",
        action="store_true",
        help="Disable system tray icon"
    )
    parser.add_argument(
        "--no-commands",
        action="store_true",
        help="Disable voice commands (pure dictation only)"
    )
    parser.add_argument(
        "--test-vad",
        action="store_true",
        help="Test VAD without transcription"
    )
    parser.add_argument(
        "--fix-homonyms",
        action="store_true",
        help="Enable context-aware homonym correction (their/there/they're, etc.)"
    )
    parser.add_argument(
        "--homonym-rules",
        action="store_true",
        help="Use rule-based homonym fixer instead of LLM (faster but less accurate)"
    )
    parser.add_argument(
        "--device", "-d",
        default=None,
        help="Microphone device name pattern (e.g. 'RØDE NT-USB+', 'Blue Yeti')"
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List available audio input devices and exit"
    )

    args = parser.parse_args()

    if args.list_devices:
        from audio_capture import AudioCapture
        ac = AudioCapture()
        print("Available input devices:")
        for dev in ac.list_devices():
            print(f"  [{dev['index']}] {dev['name']} ({dev['sample_rate']}Hz, {dev['channels']}ch)")
        return

    if args.test_vad:
        from vad import test_vad
        test_vad()
        return

    # Determine output mode
    if args.both:
        output_mode = OutputMode.BOTH
    elif args.clipboard:
        output_mode = OutputMode.CLIPBOARD
    else:
        output_mode = OutputMode.TYPE

    app = HandsFreeDictation(
        output_mode=output_mode,
        model_size=args.model,
        use_tray=not args.no_tray,
        enable_commands=not args.no_commands,
        fix_homonyms=args.fix_homonyms,
        homonym_llm=not args.homonym_rules,
        device_name=args.device,
    )

    asyncio.run(app.run())


if __name__ == "__main__":
    main()
