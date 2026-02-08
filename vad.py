"""
Voice Activity Detection (VAD) for hands-free dictation.
Automatically detects speech start/stop without requiring hotkeys.
"""
import numpy as np
from typing import Optional, Callable
from collections import deque
from enum import Enum


class VADState(Enum):
    SILENCE = "silence"
    SPEAKING = "speaking"
    TRAILING = "trailing"  # Brief pause, might continue speaking


class VoiceActivityDetector:
    """
    Detects voice activity using energy-based VAD with hysteresis.

    Uses a simple but effective approach:
    - Track RMS energy over short windows
    - Adaptive threshold based on ambient noise
    - Hysteresis to avoid choppy detection
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        # Energy thresholds (relative to ambient noise)
        # Rode NT-USB+ at low gain: ambient ~0.002, speech peaks ~0.03
        # That's about 23dB difference, so 10dB threshold is conservative
        speech_threshold_db: float = 8.0,   # dB above ambient to trigger speech
        silence_threshold_db: float = 4.0,  # dB above ambient to maintain speech
        # Timing parameters
        min_speech_ms: int = 150,      # Minimum speech duration to trigger
        min_silence_ms: int = 500,     # Silence duration to end utterance
        trailing_silence_ms: int = 1200,  # Max trailing silence before commit
        # Ambient noise adaptation
        ambient_update_rate: float = 0.05,  # How fast to adapt to ambient noise
    ):
        self.sample_rate = sample_rate
        self.speech_threshold_db = speech_threshold_db
        self.silence_threshold_db = silence_threshold_db
        self.min_speech_ms = min_speech_ms
        self.min_silence_ms = min_silence_ms
        self.trailing_silence_ms = trailing_silence_ms
        self.ambient_update_rate = ambient_update_rate

        # State
        self._state = VADState.SILENCE
        self._ambient_rms = 0.002  # Initial estimate (normalized float)
        self._speech_start_time = 0
        self._silence_start_time = 0
        self._total_samples = 0

        # Calibration: measure actual ambient noise before detecting speech
        self._calibrating = True
        self._calibration_samples = 0
        self._calibration_target = int(sample_rate * 1.0)  # 1 second of audio
        self._calibration_rms_values: list[float] = []

        # Callbacks
        self._on_speech_start: Optional[Callable[[], None]] = None
        self._on_speech_end: Optional[Callable[[], None]] = None
        self._on_level: Optional[Callable[[float, bool], None]] = None

        # Energy history for smoothing
        self._energy_history = deque(maxlen=5)

    def set_callbacks(
        self,
        on_speech_start: Optional[Callable[[], None]] = None,
        on_speech_end: Optional[Callable[[], None]] = None,
        on_level: Optional[Callable[[float, bool], None]] = None,
    ):
        """
        Set callbacks for VAD events.

        on_speech_start: Called when speech begins
        on_speech_end: Called when speech ends (utterance complete)
        on_level: Called with (level, is_speech) for visualization
        """
        self._on_speech_start = on_speech_start
        self._on_speech_end = on_speech_end
        self._on_level = on_level

    def _samples_to_ms(self, samples: int) -> float:
        return samples * 1000 / self.sample_rate

    def _rms_to_db(self, rms: float, ref: float) -> float:
        """Convert RMS to dB relative to reference."""
        if rms <= 0 or ref <= 0:
            return -60.0
        return 20 * np.log10(rms / ref)

    def _calculate_rms(self, samples: np.ndarray) -> float:
        """Calculate RMS energy of audio samples (int16 input)."""
        # Convert int16 to float for calculation
        float_samples = samples.astype(np.float32) / 32767.0
        return np.sqrt(np.mean(float_samples ** 2))

    def process_chunk(self, audio_data: bytes) -> bool:
        """
        Process an audio chunk and update VAD state.

        Args:
            audio_data: Raw PCM audio (16-bit mono)

        Returns:
            True if currently in speech, False otherwise
        """
        # Convert bytes to numpy array
        samples = np.frombuffer(audio_data, dtype=np.int16)
        chunk_samples = len(samples)
        self._total_samples += chunk_samples

        # Calculate energy
        rms = self._calculate_rms(samples)

        # Calibration phase: measure ambient noise before doing any detection
        if self._calibrating:
            self._calibration_samples += chunk_samples
            self._calibration_rms_values.append(rms)
            if self._calibration_samples >= self._calibration_target:
                measured = float(np.mean(self._calibration_rms_values))
                # Use the higher of mean and peak to be conservative,
                # then add 20% margin so normal noise variance doesn't trigger
                peak = float(np.max(self._calibration_rms_values))
                self._ambient_rms = max(measured * 1.2, peak, 0.0005)
                self._calibrating = False
                print(f"VAD calibrated: ambient RMS = {self._ambient_rms:.5f} (measured mean={measured:.5f}, peak={peak:.5f})")
            return False  # Not speaking during calibration

        self._energy_history.append(rms)
        smoothed_rms = np.mean(self._energy_history)

        # Calculate dB above ambient
        db_above_ambient = self._rms_to_db(smoothed_rms, self._ambient_rms)

        # Determine if this chunk is speech
        is_speech_energy = db_above_ambient > self.speech_threshold_db
        is_above_silence = db_above_ambient > self.silence_threshold_db

        # Normalize level for display (0-1)
        display_level = min(1.0, max(0.0, (db_above_ambient + 10) / 30))

        # State machine
        current_time_ms = self._samples_to_ms(self._total_samples)

        if self._state == VADState.SILENCE:
            # Update ambient noise estimate during silence
            self._ambient_rms = (
                self._ambient_rms * (1 - self.ambient_update_rate) +
                smoothed_rms * self.ambient_update_rate
            )

            if is_speech_energy:
                self._speech_start_time = current_time_ms
                self._state = VADState.SPEAKING

                # Check if speech duration meets minimum
                if self._on_speech_start:
                    self._on_speech_start()

        elif self._state == VADState.SPEAKING:
            if not is_above_silence:
                # Potential end of speech
                self._silence_start_time = current_time_ms
                self._state = VADState.TRAILING

        elif self._state == VADState.TRAILING:
            silence_duration = current_time_ms - self._silence_start_time

            if is_above_silence:
                # Speech resumed
                self._state = VADState.SPEAKING
            elif silence_duration > self.trailing_silence_ms:
                # Long silence - end utterance
                self._state = VADState.SILENCE
                if self._on_speech_end:
                    self._on_speech_end()
            elif silence_duration > self.min_silence_ms and not is_speech_energy:
                # Short pause might be end of utterance
                # Keep waiting for trailing_silence_ms
                pass

        # Report level
        is_speaking = self._state in (VADState.SPEAKING, VADState.TRAILING)
        if self._on_level:
            self._on_level(display_level, is_speaking)

        return is_speaking

    def reset(self):
        """Reset VAD state."""
        self._state = VADState.SILENCE
        self._speech_start_time = 0
        self._silence_start_time = 0
        self._total_samples = 0
        self._energy_history.clear()
        self._calibrating = True
        self._calibration_samples = 0
        self._calibration_rms_values.clear()

    @property
    def state(self) -> VADState:
        return self._state

    @property
    def is_speaking(self) -> bool:
        return self._state in (VADState.SPEAKING, VADState.TRAILING)


def test_vad():
    """Test VAD with microphone input."""
    import sys
    sys.path.insert(0, '.')
    from audio_capture import AudioCapture

    print("=== VAD Test ===")
    print("Speak to test voice activity detection.")
    print("Press Ctrl+C to stop.\n")

    vad = VoiceActivityDetector()
    capture = AudioCapture(device_name="RØDE NT-USB+")

    utterance_count = 0

    def on_speech_start():
        print("\n🎤 Speech started...", end="", flush=True)

    def on_speech_end():
        nonlocal utterance_count
        utterance_count += 1
        print(f" ended. (Utterance #{utterance_count})")

    def on_level(level: float, is_speech: bool):
        bars = int(level * 30)
        indicator = "🔴" if is_speech else "⚪"
        print(f"\r{indicator} [{'█' * bars}{' ' * (30 - bars)}] {level:.2f}", end="", flush=True)

    vad.set_callbacks(
        on_speech_start=on_speech_start,
        on_speech_end=on_speech_end,
        on_level=on_level,
    )

    capture.start()

    try:
        while True:
            chunk = capture.read_chunk()
            if chunk:
                vad.process_chunk(chunk)
    except KeyboardInterrupt:
        print("\n\nStopping...")
    finally:
        capture.stop()
        capture.close()

    print(f"Total utterances detected: {utterance_count}")


if __name__ == "__main__":
    import os
    os.environ["LD_PRELOAD"] = "/usr/lib/x86_64-linux-gnu/libstdc++.so.6"
    test_vad()
