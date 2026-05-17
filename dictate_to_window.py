#!/usr/bin/env python3
"""
GPU-accelerated voice dictation using faster-whisper + Silero VAD.
Types transcribed speech directly into the focused window.

Features:
  - faster-whisper (CTranslate2) for ~4x speedup over openai-whisper
  - Silero VAD for accurate speech detection (eliminates hallucinations)
  - Threaded pipeline: records while transcribing
  - Push-to-talk mode (hold Right Ctrl) or always-on mode
  - Microphone selection

Usage:
  python dictate_to_window.py                    # always-on, default mic
  python dictate_to_window.py --push-to-talk     # hold Right Ctrl to record
  python dictate_to_window.py --model large-v3   # use specific model
  python dictate_to_window.py --list-devices      # show available mics
  python dictate_to_window.py --device 1          # use specific mic

Ctrl+C to stop.
"""

import argparse
import sys
import os
import re
import subprocess
import threading
import queue
import time
import numpy as np
import pyaudio

# ---------------------------------------------------------------------------
# Configuration defaults
# ---------------------------------------------------------------------------
SAMPLE_RATE = 16000
FRAME_SIZE = 512           # samples per VAD frame (32ms at 16kHz)
SPEECH_PAD_MS = 300        # padding around detected speech (ms)
MIN_SPEECH_MS = 250        # minimum speech duration to transcribe
MAX_SPEECH_S = 15          # maximum single utterance length (seconds)
SILENCE_AFTER_SPEECH_MS = 600  # silence duration to end an utterance

# ---------------------------------------------------------------------------
# Voice commands — punctuation requires "insert" prefix
# ---------------------------------------------------------------------------
COMMANDS = {
    'send message': 'ENTER',
    'send it': 'ENTER',
    'press enter': 'ENTER',
    'new line': '\n',
    'newline': '\n',
    'insert period': '.',
    'insert comma': ',',
    'insert question mark': '?',
    'insert exclamation': '!',
    'insert colon': ':',
    'insert semicolon': ';',
    'delete that': 'DELETE',
    'delete word': 'DELETE',
}

# Whisper hallucination phrases (produced on silence / noise)
HALLUCINATIONS = {
    '', 'you', 'the', 'a', 'i', 'it', 'bye', 'okay', 'ok', 'yeah', 'yes',
    'no', 'thanks', 'thank you', 'thanks for watching', 'thank you for watching',
    'like and subscribe', 'subscribe', 'see you next time', 'bye bye',
    'see you in the next video', 'yes sir', 'so',
}

# ---------------------------------------------------------------------------
# Input helpers (xdotool)
# ---------------------------------------------------------------------------
def type_text(text):
    """Type text into the focused window."""
    if text:
        subprocess.run(['xdotool', 'type', '--clearmodifiers', '--delay', '12', text],
                       check=False)

def press_key(key):
    """Press a key in the focused window."""
    subprocess.run(['xdotool', 'key', '--clearmodifiers', key], check=False)

def process_commands(text):
    """Check for voice commands and execute them. Returns text to type or None."""
    text_lower = text.lower().strip()

    # Exact match
    if text_lower in COMMANDS:
        cmd = COMMANDS[text_lower]
        if cmd == 'ENTER':
            press_key('Return')
            return None
        elif cmd == 'DELETE':
            press_key('ctrl+BackSpace')
            return None
        else:
            return cmd

    # Command at end of utterance
    for phrase, replacement in COMMANDS.items():
        if phrase in text_lower:
            if replacement in ('ENTER', 'DELETE'):
                if text_lower.endswith(phrase):
                    remaining = text_lower.replace(phrase, '').strip()
                    if remaining:
                        type_text(remaining + ' ')
                    if replacement == 'ENTER':
                        press_key('Return')
                    elif replacement == 'DELETE':
                        press_key('ctrl+BackSpace')
                    return None
            else:
                text = re.sub(re.escape(phrase), replacement, text, flags=re.IGNORECASE)

    return text

# ---------------------------------------------------------------------------
# Hallucination / repetition filters
# ---------------------------------------------------------------------------
def is_hallucination(text):
    """Return True if text looks like a Whisper hallucination."""
    clean = text.lower().strip().strip('.!,?')
    if clean in HALLUCINATIONS:
        return True
    # "for watching" anywhere
    if 'for watching' in clean or 'subscribe' in clean:
        return True
    # All-bye variants
    if clean.replace('bye', '').replace(' ', '').replace('for', '').replace('now', '') == '':
        return True
    return False

def has_repetition(text):
    """Return True if text contains a looping phrase."""
    words = text.lower().split()
    if len(words) < 6:
        return False
    for plen in range(2, 5):
        if len(words) >= plen * 3:
            pattern = ' '.join(words[:plen])
            if text.lower().count(pattern) >= 3:
                return True
    return False

# ---------------------------------------------------------------------------
# Audio device listing
# ---------------------------------------------------------------------------
def list_devices():
    """Print available input devices."""
    p = pyaudio.PyAudio()
    print("\nAvailable input devices:")
    print("-" * 50)
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if info['maxInputChannels'] > 0:
            default = " (DEFAULT)" if i == p.get_default_input_device_info()['index'] else ""
            print(f"  {i}: {info['name']}{default}")
    print()
    p.terminate()

# ---------------------------------------------------------------------------
# VAD-based audio recorder
# ---------------------------------------------------------------------------
class VoiceRecorder:
    """Records audio and yields speech segments using Silero VAD."""

    def __init__(self, device_index=None, push_to_talk=False):
        from silero_vad import load_silero_vad
        import torch

        self.vad_model = load_silero_vad()
        self.torch = torch
        self.device_index = device_index
        self.push_to_talk = push_to_talk
        self._ptt_active = not push_to_talk  # if no PTT, always active
        self._running = True

        # Audio setup
        self.pa = pyaudio.PyAudio()
        kwargs = dict(
            format=pyaudio.paInt16,
            channels=1,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=FRAME_SIZE,
        )
        if device_index is not None:
            kwargs['input_device_index'] = device_index
        self.stream = self.pa.open(**kwargs)

        # If push-to-talk, start keyboard listener
        if push_to_talk:
            self._start_ptt_listener()

    def _start_ptt_listener(self):
        """Listen for Right Ctrl key for push-to-talk."""
        from pynput import keyboard

        def on_press(key):
            if key == keyboard.Key.ctrl_r:
                self._ptt_active = True

        def on_release(key):
            if key == keyboard.Key.ctrl_r:
                self._ptt_active = False

        self._kb_listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        self._kb_listener.daemon = True
        self._kb_listener.start()

    def _read_frame(self):
        """Read one VAD frame from the mic. Returns float32 numpy array."""
        data = self.stream.read(FRAME_SIZE, exception_on_overflow=False)
        return np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0

    def _vad_is_speech(self, frame):
        """Run Silero VAD on a single frame. Returns True if speech detected."""
        tensor = self.torch.from_numpy(frame)
        confidence = self.vad_model(tensor, SAMPLE_RATE).item()
        return confidence > 0.5

    def iter_utterances(self):
        """Yield numpy arrays of complete speech utterances."""
        frames_per_ms = SAMPLE_RATE / 1000
        silence_frames_needed = int(SILENCE_AFTER_SPEECH_MS / (FRAME_SIZE / SAMPLE_RATE * 1000))
        min_speech_frames = int(MIN_SPEECH_MS / (FRAME_SIZE / SAMPLE_RATE * 1000))
        max_speech_frames = int(MAX_SPEECH_S * SAMPLE_RATE / FRAME_SIZE)

        while self._running:
            # Wait for speech to start
            frame = self._read_frame()

            if not self._ptt_active:
                continue

            if not self._vad_is_speech(frame):
                continue

            # Speech detected — accumulate frames
            speech_frames = [frame]
            silence_count = 0

            while self._running and silence_count < silence_frames_needed:
                frame = self._read_frame()
                speech_frames.append(frame)

                if self._vad_is_speech(frame):
                    silence_count = 0
                else:
                    silence_count += 1

                # Safety: cap utterance length
                if len(speech_frames) >= max_speech_frames:
                    break

            # Check minimum length
            if len(speech_frames) < min_speech_frames:
                continue

            # Concatenate and yield
            audio = np.concatenate(speech_frames)

            # Reset VAD state between utterances
            self.vad_model.reset_states()

            yield audio

    def stop(self):
        self._running = False
        self.stream.stop_stream()
        self.stream.close()
        self.pa.terminate()

# ---------------------------------------------------------------------------
# Transcription worker
# ---------------------------------------------------------------------------
def transcription_worker(model, audio_queue, text_queue):
    """Thread that transcribes audio from the queue."""
    while True:
        audio = audio_queue.get()
        if audio is None:
            break

        segments, info = model.transcribe(
            audio,
            language='en',
            beam_size=5,
            vad_filter=False,  # we already run VAD ourselves
            condition_on_previous_text=False,  # prevents repetition loops
        )

        text = ' '.join(seg.text.strip() for seg in segments).strip()
        if text:
            text_queue.put(text)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="GPU-accelerated voice dictation with faster-whisper + Silero VAD")
    parser.add_argument('--model', default='large-v3',
                        help='Whisper model (tiny, base, small, medium, '
                             'large-v2, large-v3) (default: large-v3)')
    parser.add_argument('--device', type=int, default=None,
                        help='Audio input device index (see --list-devices)')
    parser.add_argument('--list-devices', action='store_true',
                        help='List available input devices and exit')
    parser.add_argument('--push-to-talk', action='store_true',
                        help='Hold Right Ctrl to record (default: always listening)')
    parser.add_argument('--no-type', action='store_true',
                        help='Print to terminal only, do not type into windows')
    args = parser.parse_args()

    if args.list_devices:
        list_devices()
        return

    # Load faster-whisper model on GPU
    from faster_whisper import WhisperModel

    print(f"Loading faster-whisper model '{args.model}' on GPU...")
    model = WhisperModel(args.model, device="cuda", compute_type="float16")
    print("Model loaded!")

    # Queues for threaded pipeline
    audio_queue = queue.Queue(maxsize=2)
    text_queue = queue.Queue()

    # Start transcription thread
    transcriber = threading.Thread(
        target=transcription_worker, args=(model, audio_queue, text_queue), daemon=True)
    transcriber.start()

    # Start recorder
    recorder = VoiceRecorder(device_index=args.device, push_to_talk=args.push_to_talk)

    mode = "PUSH-TO-TALK (Right Ctrl)" if args.push_to_talk else "ALWAYS ON"
    output = "TERMINAL ONLY" if args.no_type else "TYPING TO WINDOW"

    print("\n" + "=" * 60)
    print(f"  WHISPER GPU DICTATION")
    print(f"  Model:  {args.model} | Mode: {mode}")
    print(f"  Output: {output}")
    print(f"  Engine: faster-whisper + Silero VAD")
    print("=" * 60)
    if not args.push_to_talk:
        print("Speak naturally. Pauses end an utterance.")
    else:
        print("Hold Right Ctrl and speak. Release to end.")
    print("Ctrl+C to stop.")
    print("=" * 60 + "\n")

    try:
        for audio in recorder.iter_utterances():
            duration = len(audio) / SAMPLE_RATE
            print(f"  [{duration:.1f}s] Transcribing...", end='\r')

            # Send to transcription thread
            audio_queue.put(audio)

            # Check for results (non-blocking for previous, then block for this one)
            try:
                text = text_queue.get(timeout=10)
            except queue.Empty:
                print("  [timeout]                ")
                continue

            # Filter
            if is_hallucination(text):
                print(f"  [filtered: {text[:40]}]")
                continue
            if has_repetition(text):
                print(f"  [repetition skipped]")
                continue

            print(f"  >>> {text}")

            # Type or just print
            if not args.no_type:
                processed = process_commands(text)
                if processed:
                    type_text(processed + ' ')

    except KeyboardInterrupt:
        print("\n\nStopping...")
    finally:
        recorder.stop()
        audio_queue.put(None)  # signal transcription thread to exit
        transcriber.join(timeout=5)
        print("Done.")

if __name__ == "__main__":
    main()
