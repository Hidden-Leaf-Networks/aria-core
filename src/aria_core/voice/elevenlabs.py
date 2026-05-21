"""ElevenLabs TTS + OpenAI Whisper STT providers.

ElevenLabsTTS: real text-to-speech via ElevenLabs API.
WhisperSTT: speech-to-text via OpenAI Whisper API (fallback/alternative).

Both use httpx for async HTTP and implement the voice provider protocols.
"""

from __future__ import annotations

import logging
import os

import httpx

from aria_core.voice.providers import STTProvider, STTResult, TTSProvider, TTSResult

logger = logging.getLogger(__name__)

ELEVENLABS_API_URL = "https://api.elevenlabs.io/v1/text-to-speech"
OPENAI_API_URL = "https://api.openai.com/v1/audio/transcriptions"

# Rachel — ElevenLabs default demo voice
DEFAULT_VOICE_ID = "21m00Tcm4TlvDq8ikWAM"


class ElevenLabsTTS:
    """Text-to-speech via ElevenLabs API.

    Implements TTSProvider protocol.

    Usage:
        tts = ElevenLabsTTS(api_key="el-...")
        result = await tts.synthesize("Hello world")
    """

    def __init__(
        self,
        api_key: str,
        voice_id: str = DEFAULT_VOICE_ID,
        model: str = "eleven_flash_v2_5",
    ) -> None:
        self._api_key = api_key
        self._voice_id = voice_id
        self._model = model

    @classmethod
    def from_env(cls, **kwargs) -> ElevenLabsTTS:
        """Create from ELEVENLABS_API_KEY environment variable."""
        api_key = os.environ.get("ELEVENLABS_API_KEY", "")
        if not api_key:
            raise ValueError("ELEVENLABS_API_KEY environment variable not set")
        return cls(api_key=api_key, **kwargs)

    @property
    def name(self) -> str:
        return "elevenlabs"

    def _resolve_voice_id(self, voice: str) -> str:
        """Map voice parameter to an ElevenLabs voice ID.

        'default' maps to the configured voice_id.
        Any other value is treated as a direct voice ID.
        """
        if voice == "default":
            return self._voice_id
        return voice

    async def synthesize(
        self, text: str, voice: str = "default"
    ) -> TTSResult:
        """Synthesize text to speech via ElevenLabs API.

        Args:
            text: Text to synthesize.
            voice: Voice ID or 'default' for configured voice.

        Returns:
            TTSResult with MP3 audio bytes, or empty on failure.
        """
        voice_id = self._resolve_voice_id(voice)
        url = f"{ELEVENLABS_API_URL}/{voice_id}"

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    url,
                    headers={
                        "xi-api-key": self._api_key,
                        "Content-Type": "application/json",
                        "Accept": "audio/mpeg",
                    },
                    json={
                        "text": text,
                        "model_id": self._model,
                        "voice_settings": {
                            "stability": 0.5,
                            "similarity_boost": 0.75,
                        },
                    },
                )
                response.raise_for_status()
                audio_data = response.content

            # Estimate duration: ~150 words/min, avg word ~5 chars
            word_count = max(len(text.split()), 1)
            estimated_duration_ms = int((word_count / 150) * 60 * 1000)

            return TTSResult(
                audio_data=audio_data,
                format="mp3",
                duration_ms=estimated_duration_ms,
                sample_rate=24000,
            )

        except httpx.HTTPStatusError as e:
            logger.error(
                "ElevenLabs API error: %s %s",
                e.response.status_code,
                e.response.text,
            )
            return TTSResult(audio_data=b"", format="mp3", duration_ms=0)
        except Exception as e:
            logger.error("ElevenLabs synthesis failed: %s", e)
            return TTSResult(audio_data=b"", format="mp3", duration_ms=0)


class WhisperSTT:
    """Speech-to-text via OpenAI Whisper API.

    Implements STTProvider protocol. Serves as a fallback/alternative
    to Deepgram for speech recognition.

    Usage:
        stt = WhisperSTT(api_key="sk-...")
        result = await stt.transcribe(audio_bytes)
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "whisper-1",
    ) -> None:
        self._api_key = api_key or ""
        self._model = model

    @classmethod
    def from_env(cls, **kwargs) -> WhisperSTT:
        """Create from OPENAI_API_KEY environment variable."""
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        return cls(api_key=api_key, **kwargs)

    @property
    def name(self) -> str:
        return "whisper"

    async def transcribe(
        self, audio_data: bytes, language: str = "en"
    ) -> STTResult:
        """Transcribe audio via OpenAI Whisper API.

        Args:
            audio_data: Audio bytes (WAV, MP3, etc.)
            language: ISO-639-1 language code

        Returns:
            STTResult with transcribed text, or empty on failure.
        """
        if not self._api_key:
            logger.error("Whisper STT: no API key configured")
            return STTResult(text="", language=language)

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    OPENAI_API_URL,
                    headers={
                        "Authorization": f"Bearer {self._api_key}",
                    },
                    data={
                        "model": self._model,
                        "language": language,
                        "response_format": "verbose_json",
                    },
                    files={
                        "file": ("audio.wav", audio_data, "audio/wav"),
                    },
                )
                response.raise_for_status()
                data = response.json()

            text = data.get("text", "")
            duration = data.get("duration", 0.0)
            detected_language = data.get("language", language)

            return STTResult(
                text=text,
                language=detected_language,
                confidence=1.0,  # Whisper API doesn't return confidence
                duration_ms=int(duration * 1000),
            )

        except httpx.HTTPStatusError as e:
            logger.error(
                "Whisper API error: %s %s",
                e.response.status_code,
                e.response.text,
            )
            return STTResult(text="", language=language)
        except Exception as e:
            logger.error("Whisper transcription failed: %s", e)
            return STTResult(text="", language=language)
