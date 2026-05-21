"""Tests for real voice providers — Deepgram STT, ElevenLabs TTS, Whisper STT.

All API calls are mocked; no real API keys required.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from aria_core.voice.deepgram import DeepgramSTT, DEEPGRAM_API_URL
from aria_core.voice.elevenlabs import (
    DEFAULT_VOICE_ID,
    ELEVENLABS_API_URL,
    ElevenLabsTTS,
    WhisperSTT,
)
from aria_core.voice.providers import STTProvider, STTResult, TTSProvider, TTSResult


# ---------------------------------------------------------------------------
# DeepgramSTT tests
# ---------------------------------------------------------------------------


class TestDeepgramSTTInit:
    """DeepgramSTT initialization and properties."""

    def test_init_defaults(self):
        stt = DeepgramSTT(api_key="dg-test-key")
        assert stt._api_key == "dg-test-key"
        assert stt._model == "nova-2"
        assert stt._language == "en"

    def test_init_custom(self):
        stt = DeepgramSTT(api_key="dg-key", model="nova-3", language="fr")
        assert stt._model == "nova-3"
        assert stt._language == "fr"

    def test_name_property(self):
        stt = DeepgramSTT(api_key="dg-key")
        assert stt.name == "deepgram"

    def test_implements_stt_protocol(self):
        stt = DeepgramSTT(api_key="dg-key")
        assert isinstance(stt, STTProvider)

    def test_from_env_success(self):
        with patch.dict("os.environ", {"DEEPGRAM_API_KEY": "dg-env-key"}):
            stt = DeepgramSTT.from_env()
            assert stt._api_key == "dg-env-key"

    def test_from_env_missing_key(self):
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="DEEPGRAM_API_KEY"):
                DeepgramSTT.from_env()

    def test_from_env_with_overrides(self):
        with patch.dict("os.environ", {"DEEPGRAM_API_KEY": "dg-key"}):
            stt = DeepgramSTT.from_env(model="nova-3", language="ja")
            assert stt._model == "nova-3"
            assert stt._language == "ja"


class TestDeepgramSTTTranscribe:
    """DeepgramSTT.transcribe() with mocked HTTP."""

    @pytest.mark.asyncio
    async def test_successful_transcription(self):
        deepgram_response = {
            "results": {
                "channels": [
                    {
                        "alternatives": [
                            {
                                "transcript": "hello world",
                                "confidence": 0.98,
                            }
                        ]
                    }
                ]
            },
            "metadata": {"duration": 2.5},
        }

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = deepgram_response
        mock_response.raise_for_status = MagicMock()

        stt = DeepgramSTT(api_key="dg-test")

        with patch("aria_core.voice.deepgram.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await stt.transcribe(b"fake-audio-data")

        assert result.text == "hello world"
        assert result.confidence == 0.98
        assert result.duration_ms == 2500
        assert result.language == "en"

    @pytest.mark.asyncio
    async def test_empty_channels(self):
        """Empty channels list returns empty STTResult."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"results": {"channels": []}, "metadata": {}}
        mock_response.raise_for_status = MagicMock()

        stt = DeepgramSTT(api_key="dg-test")

        with patch("aria_core.voice.deepgram.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await stt.transcribe(b"silence")

        assert result.text == ""

    @pytest.mark.asyncio
    async def test_http_error_graceful(self):
        """HTTP errors return empty STTResult instead of raising."""
        stt = DeepgramSTT(api_key="bad-key")

        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.text = "Unauthorized"
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "401", request=MagicMock(), response=mock_response
        )

        with patch("aria_core.voice.deepgram.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await stt.transcribe(b"audio")

        assert result.text == ""
        assert isinstance(result, STTResult)

    @pytest.mark.asyncio
    async def test_network_error_graceful(self):
        """Network failures return empty STTResult."""
        stt = DeepgramSTT(api_key="dg-test")

        with patch("aria_core.voice.deepgram.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(side_effect=httpx.ConnectError("connection refused"))
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await stt.transcribe(b"audio")

        assert result.text == ""


# ---------------------------------------------------------------------------
# ElevenLabsTTS tests
# ---------------------------------------------------------------------------


class TestElevenLabsTTSInit:
    """ElevenLabsTTS initialization and properties."""

    def test_init_defaults(self):
        tts = ElevenLabsTTS(api_key="el-test-key")
        assert tts._api_key == "el-test-key"
        assert tts._voice_id == DEFAULT_VOICE_ID
        assert tts._model == "eleven_flash_v2_5"

    def test_init_custom(self):
        tts = ElevenLabsTTS(api_key="el-key", voice_id="custom-id", model="eleven_turbo_v2")
        assert tts._voice_id == "custom-id"
        assert tts._model == "eleven_turbo_v2"

    def test_name_property(self):
        tts = ElevenLabsTTS(api_key="el-key")
        assert tts.name == "elevenlabs"

    def test_implements_tts_protocol(self):
        tts = ElevenLabsTTS(api_key="el-key")
        assert isinstance(tts, TTSProvider)

    def test_from_env_success(self):
        with patch.dict("os.environ", {"ELEVENLABS_API_KEY": "el-env-key"}):
            tts = ElevenLabsTTS.from_env()
            assert tts._api_key == "el-env-key"

    def test_from_env_missing_key(self):
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="ELEVENLABS_API_KEY"):
                ElevenLabsTTS.from_env()

    def test_voice_id_mapping_default(self):
        tts = ElevenLabsTTS(api_key="el-key", voice_id="my-voice")
        assert tts._resolve_voice_id("default") == "my-voice"

    def test_voice_id_mapping_custom(self):
        tts = ElevenLabsTTS(api_key="el-key")
        assert tts._resolve_voice_id("some-other-voice") == "some-other-voice"


class TestElevenLabsTTSSynthesize:
    """ElevenLabsTTS.synthesize() with mocked HTTP."""

    @pytest.mark.asyncio
    async def test_successful_synthesis(self):
        fake_audio = b"\xff\xfb\x90\x00" * 100  # fake MP3 bytes

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = fake_audio
        mock_response.raise_for_status = MagicMock()

        tts = ElevenLabsTTS(api_key="el-test")

        with patch("aria_core.voice.elevenlabs.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await tts.synthesize("Hello world")

        assert result.audio_data == fake_audio
        assert result.format == "mp3"
        assert result.duration_ms > 0

    @pytest.mark.asyncio
    async def test_http_error_graceful(self):
        """HTTP errors return empty TTSResult."""
        tts = ElevenLabsTTS(api_key="bad-key")

        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.text = "Unauthorized"
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "401", request=MagicMock(), response=mock_response
        )

        with patch("aria_core.voice.elevenlabs.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await tts.synthesize("test")

        assert result.audio_data == b""
        assert isinstance(result, TTSResult)

    @pytest.mark.asyncio
    async def test_custom_voice_in_url(self):
        """Custom voice ID is used in the API URL."""
        mock_response = MagicMock()
        mock_response.content = b"audio"
        mock_response.raise_for_status = MagicMock()

        tts = ElevenLabsTTS(api_key="el-test", voice_id="default-voice")

        with patch("aria_core.voice.elevenlabs.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            await tts.synthesize("test", voice="custom-voice-123")

            # Verify the URL contains the custom voice ID
            call_args = mock_client.post.call_args
            url = call_args[0][0] if call_args[0] else call_args[1].get("url", "")
            assert "custom-voice-123" in url


# ---------------------------------------------------------------------------
# WhisperSTT tests
# ---------------------------------------------------------------------------


class TestWhisperSTTInit:
    """WhisperSTT initialization and properties."""

    def test_init_defaults(self):
        stt = WhisperSTT()
        assert stt._api_key == ""
        assert stt._model == "whisper-1"

    def test_init_with_key(self):
        stt = WhisperSTT(api_key="sk-test", model="whisper-2")
        assert stt._api_key == "sk-test"
        assert stt._model == "whisper-2"

    def test_name_property(self):
        stt = WhisperSTT()
        assert stt.name == "whisper"

    def test_implements_stt_protocol(self):
        stt = WhisperSTT(api_key="sk-test")
        assert isinstance(stt, STTProvider)

    def test_from_env_success(self):
        with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-env-key"}):
            stt = WhisperSTT.from_env()
            assert stt._api_key == "sk-env-key"

    def test_from_env_missing_key(self):
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="OPENAI_API_KEY"):
                WhisperSTT.from_env()

    @pytest.mark.asyncio
    async def test_no_api_key_returns_empty(self):
        """Without API key, transcribe returns empty result."""
        stt = WhisperSTT()
        result = await stt.transcribe(b"audio-data")
        assert result.text == ""

    @pytest.mark.asyncio
    async def test_successful_transcription(self):
        whisper_response = {
            "text": "hello from whisper",
            "language": "en",
            "duration": 3.2,
        }

        mock_response = MagicMock()
        mock_response.json.return_value = whisper_response
        mock_response.raise_for_status = MagicMock()

        stt = WhisperSTT(api_key="sk-test")

        with patch("aria_core.voice.elevenlabs.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await stt.transcribe(b"audio-data")

        assert result.text == "hello from whisper"
        assert result.duration_ms == 3200
        assert result.language == "en"


# ---------------------------------------------------------------------------
# STTResult / TTSResult construction tests
# ---------------------------------------------------------------------------


class TestResultModels:
    """STTResult and TTSResult construction edge cases."""

    def test_stt_result_defaults(self):
        r = STTResult(text="hello")
        assert r.language == "en"
        assert r.confidence == 1.0
        assert r.duration_ms == 0
        assert r.segments == []

    def test_tts_result_defaults(self):
        r = TTSResult()
        assert r.audio_data == b""
        assert r.format == "mp3"
        assert r.sample_rate == 24000

    def test_stt_result_full(self):
        r = STTResult(
            text="test",
            language="ja",
            confidence=0.95,
            duration_ms=1500,
            segments=[{"start": 0, "end": 1.5, "text": "test"}],
        )
        assert r.language == "ja"
        assert r.confidence == 0.95
        assert len(r.segments) == 1

    def test_tts_result_with_audio(self):
        r = TTSResult(audio_data=b"mp3data", format="wav", duration_ms=2000, sample_rate=44100)
        assert r.audio_data == b"mp3data"
        assert r.format == "wav"
        assert r.sample_rate == 44100


# ---------------------------------------------------------------------------
# Module-level exports
# ---------------------------------------------------------------------------


class TestModuleExports:
    """Verify __init__.py exports are accessible."""

    def test_deepgram_export(self):
        from aria_core.voice import DeepgramSTT
        assert DeepgramSTT is not None

    def test_elevenlabs_export(self):
        from aria_core.voice import ElevenLabsTTS
        assert ElevenLabsTTS is not None

    def test_whisper_export(self):
        from aria_core.voice import WhisperSTT
        assert WhisperSTT is not None
