# Hands-Free Voice Dictation

Local, GPU-accelerated voice dictation for Linux. Speak naturally and text appears wherever your cursor is. No cloud APIs required.

Uses [faster-whisper](https://github.com/SYSTRAN/faster-whisper) for on-device transcription with automatic voice activity detection (VAD) -- just talk and it types.

## Features

- **Hands-free operation** -- no hotkeys needed, VAD detects speech automatically
- **Local transcription** -- runs entirely on your machine via faster-whisper (CUDA GPU or CPU)
- **Auto-calibrating VAD** -- calibrates to your ambient noise level at startup
- **Voice commands** -- say "computer new line", "computer delete that", etc.
- **System tray indicator** -- shows recording/processing status
- **Multiple output modes** -- type into focused window (xdotool), clipboard, or both
- **Homonym correction** -- optional context-aware fixing of their/there/they're, etc.

## Requirements

- Linux (X11 -- uses xdotool for typing)
- Python 3.10+
- NVIDIA GPU recommended (works on CPU but slower)
- `xdotool` and `xclip` system packages

## Quick Start

```bash
# System dependencies
sudo apt install xdotool xclip portaudio19-dev

# Python dependencies
pip install -r requirements.txt

# Run hands-free dictation
./dictate-handsfree.sh
```

On first run, the whisper model will be downloaded (~150 MB for base). Stay quiet for 1 second while VAD calibrates to your room noise, then start talking.

## Usage

```bash
# Default (base model, type into focused window)
./dictate-handsfree.sh

# Better accuracy with small model
./dictate-handsfree.sh -m small

# Output to clipboard instead of typing
./dictate-handsfree.sh -c

# Both typing and clipboard
./dictate-handsfree.sh -b

# Test voice activity detection only (no transcription)
./dictate-handsfree.sh --test-vad

# Push-to-talk / toggle mode (alternative to hands-free)
./run.sh --local
```

## Voice Commands

Say "computer" followed by a command:

| Command | Action |
|---|---|
| computer new line | Press Enter |
| computer new paragraph | Press Enter twice |
| computer tab | Press Tab |
| computer delete that | Delete last utterance |
| computer stop listening | Stop dictation |
| computer pause | Pause dictation |
| computer caps on/off | Toggle caps lock mode |

Disable voice commands with `--no-commands` for pure dictation.

## Architecture

| Module | Purpose |
|---|---|
| `hands_free.py` | Main entry point for hands-free mode |
| `dictate.py` | Push-to-talk / toggle mode entry point |
| `vad.py` | Voice activity detection with auto-calibration |
| `audio_capture.py` | Microphone input via sounddevice with auto-resampling |
| `transcriber_local.py` | Local transcription via faster-whisper |
| `transcriber.py` | Voxtral API transcription (optional cloud mode) |
| `text_output.py` | Text output via xdotool / xclip |
| `voice_commands.py` | Voice command parsing and execution |
| `homonym_fixer.py` | Context-aware homonym correction |
| `tray_icon.py` | System tray status indicator |
| `config.py` | Configuration defaults |

## Configuration

Edit `config.py` to set your microphone device name, sample rate, hotkeys, etc. The defaults work for most USB microphones.

To use a specific mic, set `device_name` in `AudioConfig`:

```python
device_name: Optional[str] = "Your Mic Name"
```

Run with `--test-vad` to verify your mic is detected and VAD is working before doing full dictation.

## License

MIT
