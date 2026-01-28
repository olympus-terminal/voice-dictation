#!/usr/bin/env python3
"""
GPU-accelerated voice dictation using faster-whisper + Silero VAD.
Prints transcribed speech to the terminal (no xdotool needed).

Usage:
  python dictate_terminal.py                    # always-on, default mic
  python dictate_terminal.py --model large-v3   # use specific model
  python dictate_terminal.py --list-devices     # show available mics
  python dictate_terminal.py --device 1         # use specific mic

Ctrl+C to stop.
"""

import argparse
import sys
import threading
import queue
import numpy as np
import pyaudio

# ---------------------------------------------------------------------------
# Configuration defaults
# ---------------------------------------------------------------------------
SAMPLE_RATE = 16000
FRAME_SIZE = 512           # samples per VAD frame (32ms at 16kHz)
MIN_SPEECH_MS = 250        # minimum speech duration to transcribe
MAX_SPEECH_S = 15          # maximum single utterance length (seconds)
SILENCE_AFTER_SPEECH_MS = 600  # silence duration to end an utterance

# Whisper hallucination phrases (produced on silence / noise)
HALLUCINATIONS = {
    '', 'you', 'the', 'a', 'i', 'it', 'bye', 'okay', 'ok', 'yeah', 'yes',
    'no', 'thanks', 'thank you', 'thanks for watching', 'thank you for watching',
    'like and subscribe', 'subscribe', 'see you next time', 'bye bye',
    'see you in the next video', 'yes sir', 'so',
}

# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------
def is_hallucination(text):
    """Return True if text looks like a Whisper hallucination."""
    clean = text.lower().strip().strip('.!,?')
    if clean in HALLUCINATIONS:
        return True
    if 'for watching' in clean or 'subscribe' in clean:
        return True
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

    def __init__(self, device_index=None):
        from silero_vad import load_silero_vad
        import torch

        self.vad_model = load_silero_vad()
        self.torch = torch
        self._running = True

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

    def _read_frame(self):
        data = self.stream.read(FRAME_SIZE, exception_on_overflow=False)
        return np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0

    def _vad_is_speech(self, frame):
        tensor = self.torch.from_numpy(frame)
        confidence = self.vad_model(tensor, SAMPLE_RATE).item()
        return confidence > 0.5

    def iter_utterances(self):
        """Yield numpy arrays of complete speech utterances."""
        silence_frames_needed = int(SILENCE_AFTER_SPEECH_MS / (FRAME_SIZE / SAMPLE_RATE * 1000))
        min_speech_frames = int(MIN_SPEECH_MS / (FRAME_SIZE / SAMPLE_RATE * 1000))
        max_speech_frames = int(MAX_SPEECH_S * SAMPLE_RATE / FRAME_SIZE)

        while self._running:
            frame = self._read_frame()
            if not self._vad_is_speech(frame):
                continue

            speech_frames = [frame]
            silence_count = 0

            while self._running and silence_count < silence_frames_needed:
                frame = self._read_frame()
                speech_frames.append(frame)
                if self._vad_is_speech(frame):
                    silence_count = 0
                else:
                    silence_count += 1
                if len(speech_frames) >= max_speech_frames:
                    break

            if len(speech_frames) < min_speech_frames:
                continue

            audio = np.concatenate(speech_frames)
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
            vad_filter=False,
            condition_on_previous_text=False,
        )

        text = ' '.join(seg.text.strip() for seg in segments).strip()
        if text:
            text_queue.put(text)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="GPU-accelerated voice dictation (terminal output)")
    parser.add_argument('--model', default='large-v3',
                        help='Whisper model (default: large-v3)')
    parser.add_argument('--device', type=int, default=None,
                        help='Audio input device index (see --list-devices)')
    parser.add_argument('--list-devices', action='store_true',
                        help='List available input devices and exit')
    args = parser.parse_args()

    if args.list_devices:
        list_devices()
        return

    from faster_whisper import WhisperModel

    print(f"Loading faster-whisper model '{args.model}' on GPU...")
    model = WhisperModel(args.model, device="cuda", compute_type="float16")
    print("Model loaded!")

    audio_queue = queue.Queue(maxsize=2)
    text_queue = queue.Queue()

    transcriber = threading.Thread(
        target=transcription_worker, args=(model, audio_queue, text_queue), daemon=True)
    transcriber.start()

    recorder = VoiceRecorder(device_index=args.device)

    print("\n" + "=" * 60)
    print(f"  WHISPER GPU DICTATION (terminal)")
    print(f"  Model:  {args.model}")
    print(f"  Engine: faster-whisper + Silero VAD")
    print("=" * 60)
    print("Speak naturally. Pauses end an utterance.")
    print("Ctrl+C to stop.")
    print("=" * 60 + "\n")

    try:
        for audio in recorder.iter_utterances():
            duration = len(audio) / SAMPLE_RATE
            print(f"  [{duration:.1f}s] Transcribing...", end='\r')

            audio_queue.put(audio)

            try:
                text = text_queue.get(timeout=10)
            except queue.Empty:
                print("  [timeout]                ")
                continue

            if is_hallucination(text):
                print(f"  [filtered: {text[:40]}]")
                continue
            if has_repetition(text):
                print(f"  [repetition skipped]")
                continue

            print(f"  >>> {text}")

    except KeyboardInterrupt:
        print("\n\nStopping...")
    finally:
        recorder.stop()
        audio_queue.put(None)
        transcriber.join(timeout=5)
        print("Done.")

if __name__ == "__main__":
    main()
