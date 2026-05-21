"""Deepgram STT provider — real speech-to-text via Deepgram Nova API.

Uses httpx for async HTTP calls to Deepgram's /v1/listen endpoint.
Implements the STTProvider protocol from aria_core.voice.providers.
"""

from __future__ import annotations

import logging
import os

import httpx

from aria_core.voice.providers import STTProvider, STTResult

logger = logging.getLogger(__name__)

DEEPGRAM_API_URL = "https://api.deepgram.com/v1/listen"


class DeepgramSTT:
    """Speech-to-text via Deepgram Nova API.

    Implements STTProvider protocol.

    Usage:
        stt = DeepgramSTT(api_key="dg-...")
        result = await stt.transcribe(audio_bytes)
    """

    def __init__(
        self,
        api_key: str,
        model: str = "nova-2",
        language: str = "en",
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._language = language

    @classmethod
    def from_env(cls, **kwargs) -> DeepgramSTT:
        """Create from DEEPGRAM_API_KEY environment variable."""
        api_key = os.environ.get("DEEPGRAM_API_KEY", "")
        if not api_key:
            raise ValueError("DEEPGRAM_API_KEY environment variable not set")
        return cls(api_key=api_key, **kwargs)

    @property
    def name(self) -> str:
        return "deepgram"

    async def transcribe(
        self, audio_data: bytes, language: str = "en"
    ) -> STTResult:
        """Transcribe audio bytes via Deepgram API.

        Args:
            audio_data: Raw audio bytes (WAV, MP3, etc.)
            language: BCP-47 language code

        Returns:
            STTResult with transcribed text, or empty text on failure.
        """
        lang = language or self._language

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    DEEPGRAM_API_URL,
                    headers={
                        "Authorization": f"Token {self._api_key}",
                        "Content-Type": "audio/wav",
                    },
                    params={
                        "model": self._model,
                        "language": lang,
                        "punctuate": "true",
                        "utterances": "false",
                    },
                    content=audio_data,
                )
                response.raise_for_status()
                data = response.json()

            # Parse Deepgram response structure
            results = data.get("results", {})
            channels = results.get("channels", [])

            if not channels:
                return STTResult(text="", language=lang)

            alternatives = channels[0].get("alternatives", [])
            if not alternatives:
                return STTResult(text="", language=lang)

            best = alternatives[0]
            text = best.get("transcript", "")
            confidence = best.get("confidence", 0.0)

            # Duration from metadata
            metadata = data.get("metadata", {})
            duration_seconds = metadata.get("duration", 0.0)
            duration_ms = int(duration_seconds * 1000)

            return STTResult(
                text=text,
                language=lang,
                confidence=confidence,
                duration_ms=duration_ms,
            )

        except httpx.HTTPStatusError as e:
            logger.error("Deepgram API error: %s %s", e.response.status_code, e.response.text)
            return STTResult(text="", language=lang)
        except Exception as e:
            logger.error("Deepgram transcription failed: %s", e)
            return STTResult(text="", language=lang)
