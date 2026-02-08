#!/bin/bash
# Hands-free voice dictation
# Automatically detects speech - no keyboard needed
#
# Usage:
#   ./dictate-handsfree.sh                    # Use base model with default mic
#   ./dictate-handsfree.sh -m small           # Use small model (more accurate)
#   ./dictate-handsfree.sh -d "Blue Yeti"     # Specify microphone by name
#   ./dictate-handsfree.sh -c                 # Output to clipboard only
#   ./dictate-handsfree.sh --list-devices     # Show available microphones
#   ./dictate-handsfree.sh --test-vad         # Test voice detection only

cd "$(dirname "$0")"

# Use system libstdc++ to avoid GLIBCXX version conflicts
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6

# Default microphone - change this to match your setup, or use -d flag
# Run ./dictate-handsfree.sh --list-devices to see available mics
DEFAULT_DEVICE="RØDE NT-USB+"

exec python hands_free.py --device "$DEFAULT_DEVICE" "$@"
