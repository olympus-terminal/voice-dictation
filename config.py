"""
Configuration for Voxtral dictation system.
"""
import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AudioConfig:
    """Audio capture settings."""
    sample_rate: int = 16000
    channels: int = 1
    chunk_duration_ms: int = 100  # 100ms chunks for streaming
    format_bits: int = 16
    # Set device_name to match your mic (e.g. "RØDE NT-USB+")
    # or leave as None to use system default
    device_index: Optional[int] = None
    device_name: Optional[str] = None


@dataclass
class VoxtralConfig:
    """Voxtral API settings."""
    api_key: str = field(default_factory=lambda: os.environ.get("MISTRAL_API_KEY", ""))
    model: str = "voxtral-mini-transcribe-realtime-2602"
    # Transcription delay in ms (480ms is recommended sweet spot)
    delay_ms: int = 480
    # Context biasing words for domain-specific vocabulary
    context_bias: list[str] = field(default_factory=list)
    # Languages supported: en, zh, hi, es, ar, fr, pt, ru, de, ja, ko, it, nl
    language: str = "en"


@dataclass
class HotkeyConfig:
    """Hotkey settings."""
    # Push-to-talk key (hold to record)
    push_to_talk: str = "ctrl+shift+space"
    # Toggle recording on/off
    toggle_record: str = "ctrl+shift+r"
    # Stop and cancel current recording
    cancel: str = "escape"


@dataclass
class OutputConfig:
    """Text output settings."""
    # Output mode: "type" (xdotool), "clipboard", or "both"
    mode: str = "type"
    # Delay between characters when typing (ms)
    typing_delay_ms: int = 0
    # Add punctuation automatically
    auto_punctuation: bool = True
    # Capitalize sentences
    auto_capitalize: bool = True


@dataclass
class Config:
    """Main configuration."""
    audio: AudioConfig = field(default_factory=AudioConfig)
    voxtral: VoxtralConfig = field(default_factory=VoxtralConfig)
    hotkeys: HotkeyConfig = field(default_factory=HotkeyConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    # UI settings
    show_tray_icon: bool = True
    show_audio_level: bool = True

    # Debug
    debug: bool = False
    save_recordings: bool = False
    recordings_dir: str = "/tmp/dictation_recordings"


def load_config() -> Config:
    """Load configuration, checking for API key."""
    config = Config()

    if not config.voxtral.api_key:
        print("Warning: MISTRAL_API_KEY not set in environment")
        print("Set it with: export MISTRAL_API_KEY='your-key-here'")

    return config


# Global config instance
_config: Optional[Config] = None


def get_config() -> Config:
    """Get or create global config instance."""
    global _config
    if _config is None:
        _config = load_config()
    return _config
