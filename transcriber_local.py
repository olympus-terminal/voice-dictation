"""
Local transcription using faster-whisper.
Provides offline speech-to-text without API dependency.
"""
import asyncio
import io
import wave
import tempfile
from typing import AsyncIterator, Optional, Callable
from dataclasses import dataclass

try:
    from faster_whisper import WhisperModel
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    print("faster-whisper not installed. Run: pip install faster-whisper")


@dataclass
class TranscriptionResult:
    """Result from transcription."""
    text: str
    is_final: bool
    confidence: Optional[float] = None


class LocalTranscriber:
    """
    Local speech transcription using faster-whisper.
    Processes audio in chunks for near-real-time transcription.
    """

    def __init__(
        self,
        model_size: str = "base",
        device: str = "cuda",
        compute_type: str = "float16",
        language: str = "en",
        sample_rate: int = 16000,
    ):
        if not WHISPER_AVAILABLE:
            raise RuntimeError("faster-whisper not installed")

        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self.sample_rate = sample_rate

        self._model: Optional[WhisperModel] = None
        self._text_callback: Optional[Callable[[str, bool], None]] = None

        # Buffer for accumulating audio
        self._audio_buffer = bytearray()
        # Minimum audio length before processing (in seconds)
        self._min_chunk_seconds = 1.0
        # Maximum buffer before forcing transcription
        self._max_chunk_seconds = 5.0

    def _load_model(self) -> WhisperModel:
        """Load the Whisper model (lazy loading)."""
        if self._model is None:
            print(f"Loading faster-whisper {self.model_size} model...")
            self._model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
            )
            print("Model loaded!")
        return self._model

    def set_text_callback(self, callback: Callable[[str, bool], None]) -> None:
        """Set callback for transcription updates."""
        self._text_callback = callback

    def _bytes_to_seconds(self, num_bytes: int) -> float:
        """Convert byte count to seconds of audio."""
        # 16-bit mono = 2 bytes per sample
        return num_bytes / 2 / self.sample_rate

    def _create_wav_buffer(self, audio_data: bytes) -> io.BytesIO:
        """Create a WAV file in memory from raw PCM data."""
        buffer = io.BytesIO()
        with wave.open(buffer, 'wb') as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)  # 16-bit
            wav.setframerate(self.sample_rate)
            wav.writeframes(audio_data)
        buffer.seek(0)
        return buffer

    def transcribe_audio(self, audio_data: bytes) -> str:
        """
        Transcribe audio data.

        Args:
            audio_data: Raw PCM audio (16-bit mono)

        Returns:
            Transcribed text
        """
        model = self._load_model()

        # Create WAV in memory
        wav_buffer = self._create_wav_buffer(audio_data)

        # Save to temp file (faster-whisper needs file path)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(wav_buffer.read())
            temp_path = f.name

        try:
            segments, info = model.transcribe(
                temp_path,
                language=self.language,
                beam_size=5,
                vad_filter=True,
                vad_parameters=dict(
                    min_silence_duration_ms=500,
                    speech_pad_ms=200,
                ),
            )

            # Collect all text
            text_parts = []
            for segment in segments:
                text_parts.append(segment.text.strip())

            return " ".join(text_parts)
        finally:
            import os
            os.unlink(temp_path)

    async def transcribe_stream(
        self,
        audio_stream: AsyncIterator[bytes],
    ) -> AsyncIterator[TranscriptionResult]:
        """
        Transcribe audio stream in near-real-time.

        Buffers audio and transcribes in chunks for low latency.
        """
        model = self._load_model()
        self._audio_buffer.clear()
        accumulated_text = ""

        min_bytes = int(self._min_chunk_seconds * self.sample_rate * 2)
        max_bytes = int(self._max_chunk_seconds * self.sample_rate * 2)

        async for chunk in audio_stream:
            self._audio_buffer.extend(chunk)

            # Process when we have enough audio
            if len(self._audio_buffer) >= min_bytes:
                # Transcribe current buffer
                audio_data = bytes(self._audio_buffer)

                # Run transcription in thread pool
                loop = asyncio.get_event_loop()
                text = await loop.run_in_executor(
                    None, self.transcribe_audio, audio_data
                )

                if text and text != accumulated_text:
                    # Emit incremental result
                    new_text = text[len(accumulated_text):] if text.startswith(accumulated_text) else text
                    accumulated_text = text

                    result = TranscriptionResult(text=new_text, is_final=False)

                    if self._text_callback:
                        self._text_callback(new_text, False)

                    yield result

                # Keep buffer under max size (sliding window)
                if len(self._audio_buffer) > max_bytes:
                    # Keep last portion for context
                    keep_bytes = int(self._min_chunk_seconds * self.sample_rate * 2)
                    self._audio_buffer = self._audio_buffer[-keep_bytes:]

        # Final transcription of remaining audio
        if len(self._audio_buffer) > 0:
            audio_data = bytes(self._audio_buffer)
            loop = asyncio.get_event_loop()
            text = await loop.run_in_executor(
                None, self.transcribe_audio, audio_data
            )

            if text:
                result = TranscriptionResult(text=text, is_final=True)
                if self._text_callback:
                    self._text_callback(text, True)
                yield result

        self._audio_buffer.clear()

    def reset(self) -> None:
        """Reset transcription state."""
        self._audio_buffer.clear()


def create_local_transcriber(
    model_size: str = "base",
    device: str = "auto",
    language: str = "en",
    **kwargs,
) -> Optional[LocalTranscriber]:
    """
    Factory function to create local transcriber.

    Args:
        model_size: Whisper model size (tiny, base, small, medium, large)
        device: Device to use (auto, cuda, cpu)
        language: Language code

    Returns:
        LocalTranscriber or None if not available
    """
    if not WHISPER_AVAILABLE:
        print("faster-whisper not available")
        return None

    # Auto-detect device
    if device == "auto":
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"

    compute_type = "float16" if device == "cuda" else "int8"

    return LocalTranscriber(
        model_size=model_size,
        device=device,
        compute_type=compute_type,
        language=language,
        **kwargs,
    )


async def test_local_transcriber():
    """Test local transcriber."""
    print("=== Local Transcriber Test ===\n")

    transcriber = create_local_transcriber(model_size="base")
    if not transcriber:
        print("Could not create transcriber")
        return

    # Test with silence (simulated)
    print("Testing with silence...")

    async def fake_audio():
        # Generate 2 seconds of near-silence
        import numpy as np
        for _ in range(20):  # 20 x 100ms = 2 seconds
            noise = (np.random.randn(1600) * 100).astype("<i2").tobytes()
            yield noise
            await asyncio.sleep(0.05)

    transcriber.set_text_callback(
        lambda t, f: print(f"{'[FINAL]' if f else '[...]'} {t}")
    )

    async for result in transcriber.transcribe_stream(fake_audio()):
        pass

    print("\nTest complete!")


if __name__ == "__main__":
    asyncio.run(test_local_transcriber())
