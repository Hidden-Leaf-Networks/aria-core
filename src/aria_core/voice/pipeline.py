"""Voice pipeline — orchestrates the listen → process → speak cycle.

Integrates with AgentStateMachine: voice input becomes agent messages,
agent responses become voice output.

Usage:
    pipeline = VoicePipeline(stt=whisper, tts=elevenlabs, agent=machine)
    await pipeline.process_audio(audio_bytes)
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

from pydantic import Field

from aria_core.runtime.models import BaseModel
from aria_core.voice.providers import STTProvider, TTSProvider, STTResult, TTSResult


class VoiceConfig(BaseModel):
    """Per-tenant voice configuration."""

    enabled: bool = False
    stt_provider: str = "whisper"
    tts_provider: str = "elevenlabs"
    language: str = "en"
    voice_id: str = "default"
    wake_word: str | None = None
    silence_threshold_ms: int = 1500
    max_recording_ms: int = 30000
    auto_respond: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)


class VoiceInteraction(BaseModel):
    """Record of a voice interaction (input + output)."""

    id: str
    tenant_id: UUID | None = None
    input_text: str = ""
    input_language: str = "en"
    input_confidence: float = 0.0
    input_duration_ms: int = 0
    output_text: str = ""
    output_duration_ms: int = 0
    agent_response: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class VoicePipeline:
    """Orchestrates voice interactions with an agent.

    Flow:
    1. Audio in → STT → text
    2. Text → AgentStateMachine.process_message() → response
    3. Response → TTS → audio out
    """

    def __init__(
        self,
        stt: STTProvider,
        tts: TTSProvider,
        agent: Any | None = None,
        config: VoiceConfig | None = None,
        tenant_id: UUID | None = None,
    ) -> None:
        self.stt = stt
        self.tts = tts
        self.agent = agent
        self.config = config or VoiceConfig()
        self.tenant_id = tenant_id
        self._history: list[VoiceInteraction] = []

    async def process_audio(
        self, audio_data: bytes, language: str | None = None
    ) -> VoiceInteraction:
        """Full voice pipeline: audio → text → agent → speech.

        Args:
            audio_data: Raw audio bytes
            language: Override language (default from config)

        Returns:
            VoiceInteraction with full I/O record
        """
        from uuid import uuid4

        lang = language or self.config.language

        # Step 1: STT
        stt_result = await self.stt.transcribe(audio_data, language=lang)

        # Step 2: Agent processing
        agent_response = ""
        if self.agent and stt_result.text:
            try:
                result = await self.agent.process_message(
                    stt_result.text,
                    tenant_id=self.tenant_id,
                )
                agent_response = result.context.metadata.get("response", "") or ""
            except Exception as e:
                agent_response = f"Error: {e}"

        # Step 3: TTS
        tts_result = TTSResult()
        if agent_response and self.config.auto_respond:
            tts_result = await self.tts.synthesize(
                agent_response, voice=self.config.voice_id
            )

        # Record interaction
        interaction = VoiceInteraction(
            id=str(uuid4()),
            tenant_id=self.tenant_id,
            input_text=stt_result.text,
            input_language=stt_result.language,
            input_confidence=stt_result.confidence,
            input_duration_ms=stt_result.duration_ms,
            output_text=agent_response,
            output_duration_ms=tts_result.duration_ms,
            agent_response=agent_response,
        )
        self._history.append(interaction)

        return interaction

    async def text_to_speech(self, text: str) -> TTSResult:
        """Direct TTS without agent processing."""
        return await self.tts.synthesize(text, voice=self.config.voice_id)

    async def speech_to_text(self, audio_data: bytes) -> STTResult:
        """Direct STT without agent processing."""
        return await self.stt.transcribe(audio_data, language=self.config.language)

    @property
    def history(self) -> list[VoiceInteraction]:
        return list(self._history)

    def clear_history(self) -> None:
        self._history.clear()
