#!/usr/bin/env python3
"""
GPU-accelerated voice dictation using Whisper that types into focused window.

Usage:
  1. Run this in a separate terminal
  2. Click on your Claude Code window
  3. Speak - text appears in Claude Code after each chunk
  4. Say "send message" to press Enter

Ctrl+C to stop.
"""

import whisper
import pyaudio
import numpy as np
import tempfile
import wave
import subprocess
import os
import re

# Config
SAMPLE_RATE = 16000
CHUNK_DURATION = 4  # seconds - shorter for faster response
MODEL_SIZE = "large"  # most accurate, uses ~3GB VRAM

# Voice commands - punctuation requires "insert" prefix to avoid mangling normal speech
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

def type_text(text):
    """Type text into focused window."""
    if text:
        subprocess.run(['xdotool', 'type', '--delay', '5', text], check=False)

def press_key(key):
    """Press a key."""
    subprocess.run(['xdotool', 'key', key], check=False)

def process_commands(text):
    """Process voice commands in text."""
    text_lower = text.lower().strip()

    # Check for exact command
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

    # Replace command phrases within text
    for phrase, replacement in COMMANDS.items():
        if phrase in text_lower:
            if replacement in ('ENTER', 'DELETE'):
                # Handle these specially at end
                if text_lower.endswith(phrase):
                    text = text_lower.replace(phrase, '').strip()
                    if text:
                        type_text(text + ' ')
                    if replacement == 'ENTER':
                        press_key('Return')
                    return None
            else:
                text = re.sub(re.escape(phrase), replacement, text, flags=re.IGNORECASE)

    return text

def record_chunk(stream, duration):
    """Record audio chunk."""
    frames = []
    for _ in range(int(SAMPLE_RATE * duration / 1024)):
        data = stream.read(1024, exception_on_overflow=False)
        frames.append(data)
    audio = np.frombuffer(b''.join(frames), dtype=np.int16).astype(np.float32) / 32768.0
    return audio

def is_silent(audio, threshold=0.08):
    """Check if audio is mostly silence. Higher threshold = less false triggers."""
    return np.max(np.abs(audio)) < threshold

def save_wav(audio, path):
    """Save audio to WAV."""
    with wave.open(path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes((audio * 32768).astype(np.int16).tobytes())

def main():
    print(f"Loading Whisper '{MODEL_SIZE}' model on GPU...")
    model = whisper.load_model(MODEL_SIZE, device="cuda")
    print("Model loaded!")

    p = pyaudio.PyAudio()
    stream = p.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=SAMPLE_RATE,
        input=True,
        frames_per_buffer=1024
    )

    print("\n" + "=" * 60)
    print("WHISPER GPU DICTATION - Click Claude Code, then speak")
    print("=" * 60)
    print(f"Recording {CHUNK_DURATION}s chunks, transcribing on GPU")
    print("Say 'send message' or 'send' to submit")
    print("Ctrl+C to stop")
    print("=" * 60 + "\n")

    try:
        while True:
            print("Listening...", end='\r')
            audio = record_chunk(stream, CHUNK_DURATION)

            if is_silent(audio):
                continue

            print("Transcribing...    ", end='\r')

            # Transcribe directly from numpy array (no ffmpeg needed)
            # Pad or trim to 30 seconds as Whisper expects
            audio_padded = whisper.pad_or_trim(audio)

            # Make log-Mel spectrogram
            mel = whisper.log_mel_spectrogram(audio_padded, n_mels=model.dims.n_mels).to(model.device)

            # Decode
            options = whisper.DecodingOptions(language='en', fp16=True)
            result = whisper.decode(model, mel, options)

            text = result.text.strip()

            # Filter out hallucinations
            text_clean = text.lower().strip().strip('.!,')
            hallucination_phrases = ['thank', 'watching', 'subscribe', 'bye', 'goodbye']
            # Skip if it's mostly a hallucination phrase
            if any(phrase in text_clean for phrase in ['for watching', 'subscribe',
                   'see you next', 'see you in the next', 'yes sir', 'yes, sir']):
                print(f"[skipped hallucination]")
                continue
            # Short "thanks" alone is usually hallucination
            if text_clean in ['thanks', 'thank you', 'thanks.', 'thank you.']:
                print(f"[skipped short thanks]")
                continue
            # Skip very short unclear outputs and repeated bye/thanks
            if text_clean in ['', 'you', 'the', 'a', 'i', 'it', 'bye', 'okay', 'ok', 'yeah', 'yes', 'no']:
                continue
            if text_clean.replace('bye', '').replace(' ', '').replace('for', '').replace('now', '') == '':  # "bye bye bye" or "bye for now"
                print(f"[skipped bye spam]")
                continue

            # Detect repetition loops (same phrase repeated 3+ times)
            def has_repetition(txt):
                words = txt.lower().split()
                if len(words) < 6:
                    return False
                for pattern_len in range(2, 5):
                    if len(words) >= pattern_len * 3:
                        pattern = ' '.join(words[:pattern_len])
                        if txt.lower().count(pattern) >= 3:
                            return True
                return False

            if has_repetition(text):
                print(f"[skipped repetition loop]")
                continue

            print(f"[heard] {text}")

            # Process and type
            processed = process_commands(text)
            if processed:
                type_text(processed + ' ')

    except KeyboardInterrupt:
        print("\n\nStopped.")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

if __name__ == "__main__":
    main()
