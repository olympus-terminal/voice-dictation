# Voice-to-Text GPU

GPU-accelerated voice dictation using OpenAI Whisper. Real-time speech-to-text that can either print to terminal or type directly into any focused window.

## Features

- GPU-accelerated transcription using CUDA
- Two modes: terminal output or direct window typing (via xdotool)
- Voice commands for punctuation and actions
- Hallucination filtering (blocks common Whisper artifacts like "Thanks for watching")
- Repetition loop detection
- Configurable silence threshold and model size

## Requirements

- NVIDIA GPU with CUDA support
- Python 3.8+
- ffmpeg
- xdotool (for window typing mode)
- PulseAudio/PipeWire

## Installation

```bash
# Create conda environment
conda create -n dictation python=3.11
conda activate dictation

# Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install dependencies
pip install openai-whisper pyaudio numpy

# Install system dependencies (Ubuntu/Debian)
sudo apt install ffmpeg xdotool portaudio19-dev
```

## Usage

### Terminal Mode
Prints transcribed text to the terminal:

```bash
python dictate_terminal.py [--model small] [--chunk 5]
```

### Window Typing Mode
Types directly into the focused window:

```bash
python dictate_to_window.py
```

1. Run the script in a terminal
2. Click on the window you want to type into (e.g., your editor, chat app)
3. Speak - text appears in the focused window
4. Use voice commands for punctuation and actions

## Voice Commands

| Command | Action |
|---------|--------|
| `send message` / `send it` / `press enter` | Press Enter |
| `new line` | Insert newline |
| `insert period` | Insert `.` |
| `insert comma` | Insert `,` |
| `insert question mark` | Insert `?` |
| `insert exclamation` | Insert `!` |
| `insert colon` | Insert `:` |
| `insert semicolon` | Insert `;` |
| `delete that` / `delete word` | Delete previous word |

## Configuration

Edit the config section at the top of the scripts:

```python
SAMPLE_RATE = 16000
CHUNK_DURATION = 4  # seconds per recording chunk
MODEL_SIZE = "large"  # tiny, base, small, medium, large
```

### Model Sizes

| Model | VRAM | Speed | Accuracy |
|-------|------|-------|----------|
| tiny | ~1GB | Fastest | Basic |
| base | ~1GB | Fast | Fair |
| small | ~2GB | Fast | Good |
| medium | ~5GB | Medium | Better |
| large | ~10GB | Slower | Best |

### Silence Threshold

If you're getting too many hallucinations from background noise, increase the threshold:

```python
def is_silent(audio, threshold=0.08):  # increase this value
```

## Troubleshooting

### "Thanks for watching" / hallucinations
- Increase the silence threshold
- Use a better microphone (avoid Bluetooth - compression causes issues)
- Switch to a larger model

### Repetition loops ("a little bit of a little bit of...")
- Built-in detection should skip these
- Usually caused by unclear audio or background noise

### X server crash / display manager restart
- If using `dictate_to_window.py`, xdotool can sometimes cause issues
- Increase xdotool delay in the script if needed
- Consider using on Wayland with ydotool instead

### JACK errors
```
Cannot connect to server socket err = No such file or directory
jack server is not running or cannot be started
```
These are harmless warnings and can be ignored.

## License

MIT
