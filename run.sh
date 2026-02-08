#!/bin/bash
# Launcher script for Voice Dictation
# Supports both local (faster-whisper) and API (Voxtral) transcription
#
# Usage:
#   ./run.sh --local           # Use local faster-whisper (no API needed)
#   ./run.sh --local -m small  # Use small model for better accuracy
#   ./run.sh                   # Use Voxtral API (requires MISTRAL_API_KEY)
#   ./run.sh --test-mic        # Test microphone
#   ./run.sh --help            # Show all options

cd "$(dirname "$0")"

# Use system libstdc++ to avoid GLIBCXX version conflicts with conda
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6

exec python dictate.py "$@"
