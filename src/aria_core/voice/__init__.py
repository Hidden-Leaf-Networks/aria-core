"""Voice pipeline — STT/TTS integration for voice-driven agents.

Provides:
- STTProvider protocol: speech-to-text (Whisper, Deepgram, etc.)
- TTSProvider protocol: text-to-speech (ElevenLabs, OpenAI TTS, etc.)
- VoicePipeline: orchestrates listen → process → speak cycle
- VoiceConfig: per-tenant voice settings

Reintegrates ARIA v4 voice capabilities (ARIA-199 through ARIA-205)
into the aria-core framework as a pluggable module.
"""

from aria_core.voice.pipeline import VoicePipeline, VoiceConfig
from aria_core.voice.providers import STTProvider, TTSProvider, STTResult, TTSResult
from aria_core.voice.deepgram import DeepgramSTT
from aria_core.voice.elevenlabs import ElevenLabsTTS, WhisperSTT

__all__ = [
    "DeepgramSTT",
    "ElevenLabsTTS",
    "STTProvider",
    "STTResult",
    "TTSProvider",
    "TTSResult",
    "VoiceConfig",
    "VoicePipeline",
    "WhisperSTT",
]
