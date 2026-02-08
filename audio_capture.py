"""
Audio capture module for Voxtral dictation.
Handles microphone input using sounddevice with support for the Rode NT-USB+.
Includes automatic resampling from device native rate to target rate.
"""
import asyncio
import queue
import threading
from typing import AsyncIterator, Optional, Callable
import numpy as np

try:
    import sounddevice as sd
except ImportError:
    print("sounddevice not installed. Run: pip install sounddevice")
    raise

try:
    import samplerate
    RESAMPLE_AVAILABLE = True
except ImportError:
    RESAMPLE_AVAILABLE = False


class AudioCapture:
    """
    Captures audio from microphone and yields PCM chunks for streaming.
    Uses sounddevice for cross-platform audio capture.
    Automatically resamples from device native rate to target rate.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        chunk_duration_ms: int = 100,
        device_index: Optional[int] = None,
        device_name: Optional[str] = None,
    ):
        self.sample_rate = sample_rate  # Target sample rate
        self.channels = channels
        self.chunk_duration_ms = chunk_duration_ms
        self.chunk_samples = int(sample_rate * chunk_duration_ms / 1000)

        self._stream: Optional[sd.InputStream] = None
        self._is_recording = False
        self._device_index = device_index
        self._device_name = device_name

        # Actual device sample rate (may differ from target)
        self._device_sample_rate: int = sample_rate
        self._needs_resample = False
        self._resampler = None

        # Audio data queue for async iteration
        self._audio_queue: queue.Queue = queue.Queue()

        # Callbacks for audio level monitoring
        self._level_callback: Optional[Callable[[float], None]] = None

    def find_device(self, name_pattern: Optional[str] = None) -> Optional[int]:
        """
        Find audio input device by name pattern.
        Returns device index or None if not found.
        """
        pattern = name_pattern or self._device_name
        if pattern is None:
            return None

        pattern_lower = pattern.lower()
        devices = sd.query_devices()

        for i, dev in enumerate(devices):
            if dev['max_input_channels'] > 0:
                device_name = dev['name'].lower()
                if pattern_lower in device_name:
                    print(f"Found device: {dev['name']} (index {i})")
                    return i

        print(f"Device matching '{pattern}' not found")
        return None

    def _get_device_sample_rate(self, device_index: Optional[int]) -> int:
        """Get the native sample rate for a device."""
        if device_index is None:
            return self.sample_rate

        try:
            info = sd.query_devices(device_index)
            return int(info['default_samplerate'])
        except Exception:
            return self.sample_rate

    def _check_sample_rate_support(self, device_index: Optional[int], sr: int) -> bool:
        """Check if device supports a specific sample rate."""
        try:
            sd.check_input_settings(device=device_index, samplerate=sr, channels=1)
            return True
        except Exception:
            return False

    def list_devices(self) -> list[dict]:
        """List all available input devices."""
        devices = []
        for i, dev in enumerate(sd.query_devices()):
            if dev['max_input_channels'] > 0:
                devices.append({
                    "index": i,
                    "name": dev['name'],
                    "channels": dev['max_input_channels'],
                    "sample_rate": int(dev['default_samplerate']),
                })
        return devices

    def set_level_callback(self, callback: Callable[[float], None]) -> None:
        """Set callback for audio level updates (0.0 to 1.0)."""
        self._level_callback = callback

    def _calculate_level(self, samples: np.ndarray) -> float:
        """Calculate RMS audio level from samples."""
        if len(samples) == 0:
            return 0.0
        # Calculate RMS
        rms = np.sqrt(np.mean(samples.astype(np.float32) ** 2))
        # Normalize to 0-1 range (16-bit max is 32767)
        level = min(1.0, rms / 32767.0 * 10)  # Scale up for visibility
        return level

    def _resample(self, audio_float: np.ndarray) -> np.ndarray:
        """Resample audio from device rate to target rate."""
        if not self._needs_resample:
            return audio_float

        ratio = self.sample_rate / self._device_sample_rate

        if RESAMPLE_AVAILABLE:
            # High quality resampling
            return samplerate.resample(audio_float, ratio, 'sinc_best')
        else:
            # Simple linear interpolation fallback
            old_len = len(audio_float)
            new_len = int(old_len * ratio)
            old_indices = np.arange(old_len)
            new_indices = np.linspace(0, old_len - 1, new_len)
            return np.interp(new_indices, old_indices, audio_float)

    def _audio_callback(self, indata, frames, time, status):
        """Callback for sounddevice stream."""
        if status:
            print(f"Audio status: {status}")

        # Get mono audio as float
        audio_float = indata[:, 0].copy()

        # Resample if needed
        if self._needs_resample:
            audio_float = self._resample(audio_float)

        # Convert to int16
        audio_int16 = (audio_float * 32767).astype(np.int16)

        # Calculate and report level
        if self._level_callback:
            level = self._calculate_level(audio_int16)
            self._level_callback(level)

        # Convert to bytes and queue
        self._audio_queue.put(audio_int16.tobytes())

    def start(self) -> None:
        """Start audio capture stream."""
        if self._is_recording:
            return

        # Clear queue
        while not self._audio_queue.empty():
            try:
                self._audio_queue.get_nowait()
            except queue.Empty:
                break

        # Determine device index
        device_index = self._device_index
        if device_index is None and self._device_name:
            device_index = self.find_device(self._device_name)

        # Determine sample rate to use
        if self._check_sample_rate_support(device_index, self.sample_rate):
            # Device supports target rate directly
            self._device_sample_rate = self.sample_rate
            self._needs_resample = False
        else:
            # Use device native rate and resample
            self._device_sample_rate = self._get_device_sample_rate(device_index)
            self._needs_resample = True
            print(f"Device uses {self._device_sample_rate}Hz, resampling to {self.sample_rate}Hz")

        # Calculate chunk size for device rate
        device_chunk_samples = int(self._device_sample_rate * self.chunk_duration_ms / 1000)

        try:
            self._stream = sd.InputStream(
                samplerate=self._device_sample_rate,
                channels=self.channels,
                dtype='float32',
                blocksize=device_chunk_samples,
                device=device_index,
                callback=self._audio_callback,
            )
            self._stream.start()
            self._is_recording = True
            device_info = "default" if device_index is None else f"device {device_index}"
            print(f"Recording started ({device_info}, {self._device_sample_rate}Hz -> {self.sample_rate}Hz)")
        except Exception as e:
            print(f"Failed to open audio stream: {e}")
            raise

    def stop(self) -> None:
        """Stop audio capture stream."""
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        self._is_recording = False
        print("Recording stopped")

    def read_chunk(self) -> Optional[bytes]:
        """Read a single chunk of audio data (blocking)."""
        if not self._is_recording:
            return None

        try:
            # Wait for audio with timeout
            data = self._audio_queue.get(timeout=0.5)
            return data
        except queue.Empty:
            return None
        except Exception as e:
            print(f"Error reading audio: {e}")
            return None

    async def iter_chunks(self) -> AsyncIterator[bytes]:
        """
        Async iterator that yields audio chunks.
        Use this for streaming to Voxtral API.
        """
        while self._is_recording:
            data = self.read_chunk()
            if data:
                yield data
            else:
                await asyncio.sleep(0.01)

    def close(self) -> None:
        """Clean up resources."""
        self.stop()

    @property
    def is_recording(self) -> bool:
        return self._is_recording

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def test_microphone(duration_seconds: float = 3.0, device_name: Optional[str] = None):
    """
    Test microphone capture and playback.
    Records for specified duration, then plays back.
    """
    print("=== Microphone Test ===")

    capture = AudioCapture(sample_rate=16000, device_name=device_name)

    # List devices
    print("\nAvailable input devices:")
    for dev in capture.list_devices():
        marker = " <-- SELECTED" if device_name and device_name.lower() in dev['name'].lower() else ""
        print(f"  [{dev['index']}] {dev['name']} ({dev['sample_rate']}Hz){marker}")

    # Set up level display
    def show_level(level: float):
        bars = int(level * 40)
        print(f"\rLevel: [{'█' * bars}{' ' * (40 - bars)}] {level:.2f}", end="", flush=True)

    capture.set_level_callback(show_level)

    # Record
    print(f"\nRecording for {duration_seconds} seconds...")
    capture.start()

    frames = []
    chunks_needed = int(duration_seconds * 1000 / capture.chunk_duration_ms)

    for _ in range(chunks_needed):
        chunk = capture.read_chunk()
        if chunk:
            frames.append(chunk)

    capture.stop()
    print("\n")

    # Playback
    audio_data = b"".join(frames)
    print(f"Recorded {len(audio_data)} bytes ({len(audio_data) / 2 / 16000:.1f} seconds at 16kHz)")

    if len(audio_data) > 0:
        print("Playing back...")
        # Convert bytes back to numpy array for playback
        audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32767
        sd.play(audio_array, samplerate=16000)
        sd.wait()

    capture.close()
    print("Test complete!")


if __name__ == "__main__":
    # Test with Rode NT-USB+
    test_microphone(duration_seconds=3.0, device_name="RØDE NT-USB+")
