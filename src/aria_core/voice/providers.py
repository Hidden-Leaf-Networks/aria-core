"""Voice provider protocols — pluggable STT and TTS backends.

Supported backends (via optional deps):
- STT: Whisper (local), Deepgram, OpenAI Whisper API
- TTS: ElevenLabs, OpenAI TTS, local (pyttsx3)
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from pydantic import Field
from aria_core.runtime.models import BaseModel


class STTResult(BaseModel):
    """Speech-to-text result."""

    text: str
    language: str = "en"
    confidence: float = 1.0
    duration_ms: int = 0
    segments: list[dict[str, Any]] = Field(default_factory=list)


class TTSResult(BaseModel):
    """Text-to-speech result."""

    audio_data: bytes = b""
    format: str = "mp3"
    duration_ms: int = 0
    sample_rate: int = 24000


@runtime_checkable
class STTProvider(Protocol):
    """Speech-to-text provider protocol."""

    async def transcribe(self, audio_data: bytes, language: str = "en") -> STTResult: ...

    @property
    def name(self) -> str: ...


@runtime_checkable
class TTSProvider(Protocol):
    """Text-to-speech provider protocol."""

    async def synthesize(self, text: str, voice: str = "default") -> TTSResult: ...

    @property
    def name(self) -> str: ...


class EchoSTT:
    """Stub STT provider for testing — returns a canned response."""

    @property
    def name(self) -> str:
        return "echo-stt"

    async def transcribe(self, audio_data: bytes, language: str = "en") -> STTResult:
        return STTResult(
            text=f"[echo: {len(audio_data)} bytes received]",
            language=language,
            duration_ms=len(audio_data) // 32,
        )


class EchoTTS:
    """Stub TTS provider for testing — returns empty audio."""

    @property
    def name(self) -> str:
        return "echo-tts"

    async def synthesize(self, text: str, voice: str = "default") -> TTSResult:
        return TTSResult(
            audio_data=text.encode("utf-8"),
            format="raw",
            duration_ms=len(text) * 50,
        )
