#!/usr/bin/env python3
"""
GPU-accelerated voice dictation using OpenAI Whisper.
Records audio in chunks and transcribes using GPU.

Usage: python ~/dictate_whisper.py [--model small]
Models: tiny, base, small, medium, large (default: small)
"""

import whisper
import pyaudio
import numpy as np
import tempfile
import wave
import argparse
import sys

# Audio settings
SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_DURATION = 5  # seconds per chunk
SILENCE_THRESHOLD = 500  # amplitude threshold for silence detection

def record_chunk(stream, duration):
    """Record audio chunk and return numpy array."""
    frames = []
    num_frames = int(SAMPLE_RATE * duration / 1024)

    for _ in range(num_frames):
        data = stream.read(1024, exception_on_overflow=False)
        frames.append(data)

    audio_data = b''.join(frames)
    return np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

def is_silent(audio_chunk, threshold=0.01):
    """Check if audio chunk is mostly silence."""
    return np.max(np.abs(audio_chunk)) < threshold

def save_temp_wav(audio_data, filename):
    """Save audio data to temporary WAV file."""
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes((audio_data * 32768).astype(np.int16).tobytes())

def main():
    parser = argparse.ArgumentParser(description="Whisper GPU dictation")
    parser.add_argument('--model', default='small',
                       choices=['tiny', 'base', 'small', 'medium', 'large'],
                       help='Whisper model size (default: small)')
    parser.add_argument('--chunk', type=int, default=5,
                       help='Recording chunk duration in seconds (default: 5)')
    args = parser.parse_args()

    print(f"Loading Whisper model '{args.model}' on GPU...")
    model = whisper.load_model(args.model, device="cuda")
    print(f"Model loaded successfully!")

    # Initialize audio
    p = pyaudio.PyAudio()
    stream = p.open(
        format=pyaudio.paInt16,
        channels=CHANNELS,
        rate=SAMPLE_RATE,
        input=True,
        frames_per_buffer=1024
    )

    print("\n" + "=" * 50)
    print(f"WHISPER DICTATION (GPU) - {args.chunk}s chunks")
    print("Speak now (Ctrl+C to stop)")
    print("=" * 50 + "\n")

    try:
        while True:
            # Record chunk
            audio = record_chunk(stream, args.chunk)

            # Skip if silent
            if is_silent(audio):
                print("    [silence]", end='\r')
                continue

            # Save to temp file for Whisper
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                temp_path = f.name
            save_temp_wav(audio, temp_path)

            # Transcribe with GPU
            result = model.transcribe(
                temp_path,
                language='en',
                fp16=True,  # Use FP16 for faster GPU inference
                task='transcribe'
            )

            text = result['text'].strip()
            if text and text.lower() not in ['', 'you', 'thanks for watching!', 'thank you.']:
                print(f">>> {text}")

            # Cleanup
            import os
            os.unlink(temp_path)

    except KeyboardInterrupt:
        print("\n\nStopped.")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

if __name__ == "__main__":
    main()
