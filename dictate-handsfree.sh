#!/bin/bash
# Hands-free voice dictation
# Automatically detects speech - no keyboard needed
#
# Usage:
#   ./dictate-handsfree.sh           # Use base model
#   ./dictate-handsfree.sh -m small  # Use small model (more accurate)
#   ./dictate-handsfree.sh -c        # Output to clipboard only
#   ./dictate-handsfree.sh --test-vad  # Test voice detection only

cd "$(dirname "$0")"

# Use system libstdc++ to avoid GLIBCXX version conflicts
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6

exec python hands_free.py "$@"
