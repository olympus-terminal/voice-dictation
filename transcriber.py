"""
Voxtral transcription module.
Handles real-time speech-to-text via Mistral API.
"""
import asyncio
from typing import AsyncIterator, Optional, Callable
from dataclasses import dataclass

try:
    from mistralai import Mistral
    from mistralai.models import AudioFormat
    MISTRAL_AVAILABLE = True
except ImportError:
    MISTRAL_AVAILABLE = False
    print("Mistral AI client not installed. Run: pip install 'mistralai[realtime]'")


@dataclass
class TranscriptionResult:
    """Result from transcription."""
    text: str
    is_final: bool
    confidence: Optional[float] = None


class VoxtralTranscriber:
    """
    Real-time speech transcription using Voxtral via Mistral API.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "voxtral-mini-transcribe-realtime-2602",
        sample_rate: int = 16000,
        language: str = "en",
        context_bias: Optional[list[str]] = None,
    ):
        if not MISTRAL_AVAILABLE:
            raise RuntimeError("mistralai package not installed")

        if not api_key:
            raise ValueError("MISTRAL_API_KEY is required")

        self.api_key = api_key
        self.model = model
        self.sample_rate = sample_rate
        self.language = language
        self.context_bias = context_bias or []

        self._client: Optional[Mistral] = None
        self._current_text = ""
        self._text_callback: Optional[Callable[[str, bool], None]] = None

    def _get_client(self) -> Mistral:
        """Get or create Mistral client."""
        if self._client is None:
            self._client = Mistral(api_key=self.api_key)
        return self._client

    def set_text_callback(self, callback: Callable[[str, bool], None]) -> None:
        """
        Set callback for transcription updates.
        Callback receives (text, is_final).
        """
        self._text_callback = callback

    async def transcribe_stream(
        self,
        audio_stream: AsyncIterator[bytes],
    ) -> AsyncIterator[TranscriptionResult]:
        """
        Transcribe audio stream in real-time.

        Args:
            audio_stream: Async iterator yielding PCM audio chunks (16-bit, mono)

        Yields:
            TranscriptionResult objects as text becomes available
        """
        client = self._get_client()

        audio_format = AudioFormat(
            encoding="pcm_s16le",
            sample_rate=self.sample_rate,
        )

        try:
            # Import the event types
            from mistralai.models import (
                TranscriptionStreamTextDelta,
                TranscriptionStreamEnd,
            )

            async for event in client.audio.realtime.transcribe_stream(
                audio_stream=audio_stream,
                model=self.model,
                audio_format=audio_format,
            ):
                if isinstance(event, TranscriptionStreamTextDelta):
                    # Incremental text update
                    self._current_text += event.text
                    result = TranscriptionResult(
                        text=event.text,
                        is_final=False,
                    )

                    if self._text_callback:
                        self._text_callback(event.text, False)

                    yield result

                elif isinstance(event, TranscriptionStreamEnd):
                    # Final result
                    result = TranscriptionResult(
                        text=self._current_text,
                        is_final=True,
                    )

                    if self._text_callback:
                        self._text_callback(self._current_text, True)

                    yield result
                    self._current_text = ""

        except Exception as e:
            print(f"Transcription error: {e}")
            raise

    async def transcribe_audio(self, audio_data: bytes) -> str:
        """
        Transcribe a complete audio buffer (non-streaming).

        Args:
            audio_data: Complete PCM audio data (16-bit, mono)

        Returns:
            Transcribed text
        """

        async def audio_iter():
            yield audio_data

        full_text = ""
        async for result in self.transcribe_stream(audio_iter()):
            if result.is_final:
                full_text = result.text
                break

        return full_text

    def reset(self) -> None:
        """Reset transcription state."""
        self._current_text = ""


class MockTranscriber:
    """
    Mock transcriber for testing without API.
    Simulates transcription with placeholder text.
    """

    def __init__(self):
        self._text_callback: Optional[Callable[[str, bool], None]] = None

    def set_text_callback(self, callback: Callable[[str, bool], None]) -> None:
        self._text_callback = callback

    async def transcribe_stream(
        self,
        audio_stream: AsyncIterator[bytes],
    ) -> AsyncIterator[TranscriptionResult]:
        """Simulate transcription."""
        words = ["This", " is", " a", " test", " of", " voice", " dictation", "."]

        # Consume some audio to simulate processing
        chunk_count = 0
        async for _ in audio_stream:
            chunk_count += 1
            if chunk_count % 5 == 0 and chunk_count // 5 <= len(words):
                word = words[chunk_count // 5 - 1]
                result = TranscriptionResult(text=word, is_final=False)
                if self._text_callback:
                    self._text_callback(word, False)
                yield result

            if chunk_count > 50:
                break

        # Final result
        full_text = "".join(words)
        yield TranscriptionResult(text=full_text, is_final=True)
        if self._text_callback:
            self._text_callback(full_text, True)

    def reset(self) -> None:
        pass


def create_transcriber(
    api_key: Optional[str] = None,
    use_mock: bool = False,
    **kwargs,
):
    """
    Factory function to create appropriate transcriber.

    Args:
        api_key: Mistral API key
        use_mock: If True, use mock transcriber for testing
        **kwargs: Additional arguments for VoxtralTranscriber

    Returns:
        Transcriber instance
    """
    if use_mock or not MISTRAL_AVAILABLE:
        print("Using mock transcriber")
        return MockTranscriber()

    if not api_key:
        import os
        api_key = os.environ.get("MISTRAL_API_KEY", "")

    if not api_key:
        print("No API key provided, using mock transcriber")
        return MockTranscriber()

    return VoxtralTranscriber(api_key=api_key, **kwargs)


async def test_transcriber():
    """Test transcriber with mock or real API."""
    import os

    print("=== Transcriber Test ===\n")

    api_key = os.environ.get("MISTRAL_API_KEY", "")
    transcriber = create_transcriber(api_key=api_key, use_mock=not api_key)

    # Simulate audio stream
    async def fake_audio():
        for _ in range(60):
            yield b"\x00" * 3200  # 100ms of silence at 16kHz
            await asyncio.sleep(0.05)

    print("Simulating transcription...")
    transcriber.set_text_callback(lambda t, f: print(f"{'[FINAL]' if f else '[...]'} {t}"))

    async for result in transcriber.transcribe_stream(fake_audio()):
        pass

    print("\nTest complete!")


if __name__ == "__main__":
    asyncio.run(test_transcriber())
