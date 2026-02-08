# Hands-Free Voice Dictation

Local, GPU-accelerated voice dictation for Linux. Speak naturally and text appears wherever your cursor is. No cloud APIs required.

Uses [faster-whisper](https://github.com/SYSTRAN/faster-whisper) for on-device transcription with automatic voice activity detection (VAD) -- just talk and it types.

## Features

- **Hands-free operation** -- no hotkeys needed, VAD detects speech automatically
- **Local transcription** -- runs entirely on your machine via faster-whisper (CUDA GPU or CPU)
- **Auto-calibrating VAD** -- calibrates to your ambient noise level at startup
- **Voice commands** -- say "kimmy new line", "kimmy delete that", etc.
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

# Specify microphone by name
./dictate-handsfree.sh -d "Blue Yeti"

# List available microphones
./dictate-handsfree.sh --list-devices

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

Say **"kimmy"** followed by a command. The wake word prevents accidental triggers during normal dictation. You can change it by editing `WAKE_WORD` in `voice_commands.py`.

### Deletion

| Command | Action |
|---|---|
| kimmy delete all | Select all + delete |
| kimmy delete everything | Select all + delete |
| kimmy clear all | Select all + delete |
| kimmy erase all | Select all + delete |
| kimmy delete that | Delete current line |
| kimmy delete line | Delete current line |
| kimmy scratch that | Delete current line |
| kimmy delete word | Delete last word (ctrl+w) |
| kimmy undo | Undo (ctrl+shift+z) |
| kimmy undo that | Undo (ctrl+shift+z) |
| kimmy redo | Redo (ctrl+shift+y) |

### Navigation

| Command | Action |
|---|---|
| kimmy new line | Newline |
| kimmy next line | Newline |
| kimmy new paragraph | Double newline |
| kimmy press enter | Enter key |
| kimmy press tab | Tab key |

### Editing

| Command | Action |
|---|---|
| kimmy select all | Select all (ctrl+shift+a) |
| kimmy copy that | Copy (ctrl+shift+c) |
| kimmy paste | Paste (ctrl+shift+v) |
| kimmy cut that | Cut (ctrl+shift+x) |
| kimmy save file | Save (ctrl+s) |
| kimmy escape | Escape key |
| kimmy send message | Press Enter |
| kimmy send it | Press Enter |

### Mode Control

| Command | Action |
|---|---|
| kimmy stop listening | Stop dictation |
| kimmy pause listening | Pause dictation |
| kimmy caps on | ALL CAPS mode on |
| kimmy caps off | ALL CAPS mode off |

Commands can be mixed with text in the same utterance -- e.g. "kimmy scratch that I meant something else" will delete the line then type "I meant something else."

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

## Troubleshooting

**VAD stuck in "speaking" / never transcribes:**
The VAD calibrates to ambient noise for 1 second at startup. If you speak or make noise during calibration, the threshold will be set too high and real speech won't trigger transcription. Restart the program and stay quiet during the "Calibrating..." phase. Environments with variable background noise (fans cycling, treadmills) are handled well by the calibration, but the noise must be present during the calibration window.

**"No speech detected" on every utterance:**
Usually means the wrong microphone is selected. Run `./dictate-handsfree.sh --list-devices` to see available mics, then either pass `--device "Your Mic Name"` or edit the `DEFAULT_DEVICE` line in `dictate-handsfree.sh`. The system default mic (often a laptop mic) has a much higher noise floor than a USB condenser mic, which can make the VAD unusable.

**Long pause before first transcription result:**
The whisper model is loaded eagerly at startup. If you still see a delay, check that you have GPU acceleration working -- CPU inference is significantly slower. Run `python -c "import torch; print(torch.cuda.is_available())"` to verify CUDA is available.

**"Audio status: input overflow" messages:**
This means audio chunks are arriving faster than they're being processed, usually during transcription. Occasional overflow messages are harmless -- the VAD will recover. Frequent overflow suggests the system is under heavy load. Using a smaller model (`-m tiny`) or closing GPU-intensive applications can help.

**Terminal hotkeys not working (copy, paste, undo):**
The voice commands use `ctrl+shift+c/v/x/z` for terminal compatibility. These work in most Linux terminal emulators (gnome-terminal, kitty, alacritty, etc.) but not in GUI applications where `ctrl+c/v` is standard. If you primarily dictate into GUI apps, edit the keybindings in `voice_commands.py`.

## License

MIT
