"""Tests for voice pipeline."""

from __future__ import annotations

from uuid import uuid4

import pytest

from aria_core.voice.providers import EchoSTT, EchoTTS, STTResult, TTSResult
from aria_core.voice.pipeline import VoicePipeline, VoiceConfig


class TestProviders:
    async def test_echo_stt(self) -> None:
        stt = EchoSTT()
        assert stt.name == "echo-stt"
        result = await stt.transcribe(b"fake audio data")
        assert "15 bytes" in result.text
        assert result.language == "en"

    async def test_echo_tts(self) -> None:
        tts = EchoTTS()
        assert tts.name == "echo-tts"
        result = await tts.synthesize("Hello world")
        assert result.audio_data == b"Hello world"
        assert result.duration_ms > 0


class TestVoiceConfig:
    def test_default_config(self) -> None:
        config = VoiceConfig()
        assert config.enabled is False
        assert config.stt_provider == "whisper"
        assert config.tts_provider == "elevenlabs"
        assert config.language == "en"

    def test_custom_config(self) -> None:
        config = VoiceConfig(
            enabled=True,
            language="ja",
            voice_id="aria-v4",
            wake_word="hey aria",
        )
        assert config.enabled is True
        assert config.language == "ja"
        assert config.wake_word == "hey aria"


class TestVoicePipeline:
    async def test_process_audio_without_agent(self) -> None:
        """Pipeline runs STT without agent processing."""
        pipeline = VoicePipeline(
            stt=EchoSTT(),
            tts=EchoTTS(),
        )
        interaction = await pipeline.process_audio(b"test audio")
        assert interaction.input_text != ""
        assert interaction.agent_response == ""
        assert len(pipeline.history) == 1

    async def test_direct_stt(self) -> None:
        pipeline = VoicePipeline(stt=EchoSTT(), tts=EchoTTS())
        result = await pipeline.speech_to_text(b"audio bytes")
        assert isinstance(result, STTResult)
        assert result.text != ""

    async def test_direct_tts(self) -> None:
        pipeline = VoicePipeline(stt=EchoSTT(), tts=EchoTTS())
        result = await pipeline.text_to_speech("Hello")
        assert isinstance(result, TTSResult)
        assert len(result.audio_data) > 0

    async def test_clear_history(self) -> None:
        pipeline = VoicePipeline(stt=EchoSTT(), tts=EchoTTS())
        await pipeline.process_audio(b"a")
        await pipeline.process_audio(b"b")
        assert len(pipeline.history) == 2

        pipeline.clear_history()
        assert len(pipeline.history) == 0

    async def test_tenant_scoped(self) -> None:
        tid = uuid4()
        pipeline = VoicePipeline(
            stt=EchoSTT(), tts=EchoTTS(), tenant_id=tid
        )
        interaction = await pipeline.process_audio(b"test")
        assert interaction.tenant_id == tid

    async def test_custom_language(self) -> None:
        pipeline = VoicePipeline(
            stt=EchoSTT(),
            tts=EchoTTS(),
            config=VoiceConfig(language="ja"),
        )
        interaction = await pipeline.process_audio(b"test", language="ja")
        assert interaction.input_language == "ja"

    async def test_auto_respond_disabled(self) -> None:
        pipeline = VoicePipeline(
            stt=EchoSTT(),
            tts=EchoTTS(),
            config=VoiceConfig(auto_respond=False),
        )
        interaction = await pipeline.process_audio(b"test")
        # No TTS output when auto_respond is off
        assert interaction.output_duration_ms == 0
