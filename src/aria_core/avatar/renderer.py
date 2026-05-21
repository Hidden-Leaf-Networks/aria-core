"""Avatar renderer — drives 3D avatar state, expressions, and lip sync.

Supports VRM/GLB model references, built-in expression sets, and simple
phoneme-to-viseme mapping for lip synchronisation.

ARIA-310
"""

from __future__ import annotations

import re
import sys
import time
from typing import Any

from pydantic import Field

from aria_core.runtime.models import BaseModel

if sys.version_info >= (3, 11):
    from enum import StrEnum
else:
    from enum import Enum

    class StrEnum(str, Enum):
        def __new__(cls, value: str) -> StrEnum:
            member = str.__new__(cls, value)
            member._value_ = value
            return member

        def __str__(self) -> str:
            return self.value


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class AvatarStyle(StrEnum):
    ANIME = "anime"
    REALISTIC = "realistic"
    CHIBI = "chibi"
    MINIMAL = "minimal"


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class Viseme(BaseModel):
    """Single viseme — a phoneme with weight and timestamp for lip sync."""

    phoneme: str
    weight: float = Field(ge=0.0, le=1.0)
    timestamp_ms: int = Field(ge=0)


class AvatarConfig(BaseModel):
    """Configuration for a 3D avatar."""

    style: AvatarStyle = AvatarStyle.ANIME
    model_url: str | None = None
    idle_animation: str = "idle_breathe"
    talk_animation: str = "talk_default"
    think_animation: str = "think_head_tilt"
    expression_set: dict[str, dict[str, Any]] = Field(default_factory=dict)
    voice_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class AvatarState(BaseModel):
    """Real-time avatar state snapshot."""

    current_expression: str = "neutral"
    is_talking: bool = False
    is_thinking: bool = False
    visemes: list[Viseme] = Field(default_factory=list)
    audio_url: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Built-in expressions
# ---------------------------------------------------------------------------

BUILTIN_EXPRESSIONS: dict[str, dict[str, float]] = {
    "neutral": {"browInnerUp": 0.0, "mouthSmile": 0.0, "eyeSquint": 0.0},
    "happy": {"browInnerUp": 0.1, "mouthSmile": 0.8, "eyeSquint": 0.3},
    "sad": {"browInnerUp": 0.5, "mouthFrown": 0.6, "eyeSquint": 0.0},
    "surprised": {"browInnerUp": 0.9, "mouthOpen": 0.7, "eyeWide": 0.8},
    "thinking": {"browInnerUp": 0.4, "mouthPucker": 0.2, "eyeSquint": 0.5},
    "speaking": {"browInnerUp": 0.1, "mouthOpen": 0.3, "eyeSquint": 0.1},
}


# ---------------------------------------------------------------------------
# Phoneme-to-viseme mapping (simplified)
# ---------------------------------------------------------------------------

# Maps common phoneme groups to viseme shapes + default weights.
_PHONEME_MAP: dict[str, tuple[str, float]] = {
    "a": ("aa", 0.9),
    "e": ("ee", 0.8),
    "i": ("ih", 0.7),
    "o": ("oh", 0.9),
    "u": ("ou", 0.8),
    "m": ("pp", 0.6),
    "b": ("pp", 0.7),
    "p": ("pp", 0.8),
    "f": ("ff", 0.7),
    "v": ("ff", 0.6),
    "t": ("dd", 0.5),
    "d": ("dd", 0.6),
    "s": ("ss", 0.6),
    "z": ("ss", 0.5),
    "n": ("nn", 0.5),
    "l": ("nn", 0.4),
    "r": ("rr", 0.5),
    "k": ("kk", 0.5),
    "g": ("kk", 0.6),
    "w": ("ou", 0.5),
    "y": ("ih", 0.4),
    "h": ("ss", 0.3),
    "j": ("ss", 0.4),
    "c": ("kk", 0.5),
    "x": ("kk", 0.4),
    "q": ("kk", 0.5),
    " ": ("sil", 0.0),
}


# ---------------------------------------------------------------------------
# AvatarRenderer
# ---------------------------------------------------------------------------


class AvatarRenderer:
    """Drives avatar state transitions, expressions, and viseme generation.

    Usage::

        renderer = AvatarRenderer(config)
        renderer.set_expression("happy")
        visemes = renderer.start_talking("Hello world")
        state = renderer.get_state()
    """

    def __init__(self, config: AvatarConfig) -> None:
        self._config = config
        self._expression = "neutral"
        self._is_talking = False
        self._is_thinking = False
        self._visemes: list[Viseme] = []
        self._audio_url: str | None = None

        # Merge built-in expressions with config overrides.
        self._expressions: dict[str, dict[str, Any]] = {
            **BUILTIN_EXPRESSIONS,
            **config.expression_set,
        }

    # -- Properties ---------------------------------------------------------

    @property
    def config(self) -> AvatarConfig:
        return self._config

    @property
    def expressions(self) -> dict[str, dict[str, Any]]:
        return dict(self._expressions)

    # -- Expression control -------------------------------------------------

    def set_expression(self, name: str) -> None:
        """Set the current facial expression by name.

        Raises ValueError if the expression is not registered.
        """
        if name not in self._expressions:
            raise ValueError(
                f"Unknown expression '{name}'. "
                f"Available: {sorted(self._expressions.keys())}"
            )
        self._expression = name

    # -- Talking control ----------------------------------------------------

    def start_talking(self, text: str) -> list[Viseme]:
        """Begin talking — generates visemes from *text* and sets state."""
        self._is_talking = True
        self._expression = "speaking"
        self._visemes = self.text_to_visemes(text)
        return list(self._visemes)

    def stop_talking(self) -> None:
        """Stop talking — clears visemes and resets expression to neutral."""
        self._is_talking = False
        self._visemes = []
        self._expression = "neutral"

    # -- Thinking control ---------------------------------------------------

    def start_thinking(self) -> None:
        """Enter the thinking state."""
        self._is_thinking = True
        self._expression = "thinking"

    def stop_thinking(self) -> None:
        """Exit the thinking state — reset to neutral."""
        self._is_thinking = False
        self._expression = "neutral"

    # -- State snapshot -----------------------------------------------------

    def get_state(self) -> AvatarState:
        """Return the current avatar state as an immutable snapshot."""
        return AvatarState(
            current_expression=self._expression,
            is_talking=self._is_talking,
            is_thinking=self._is_thinking,
            visemes=list(self._visemes),
            audio_url=self._audio_url,
            metadata={
                "style": str(self._config.style),
                "active_animation": self._active_animation(),
            },
        )

    # -- Viseme generation --------------------------------------------------

    def text_to_visemes(self, text: str) -> list[Viseme]:
        """Convert *text* into a list of visemes using simple phoneme mapping.

        Each character is mapped to its closest viseme shape. Non-alpha
        characters are treated as silence.
        """
        visemes: list[Viseme] = []
        ms_per_char = 80  # ~80ms per character for natural cadence
        for i, char in enumerate(text.lower()):
            phoneme, weight = _PHONEME_MAP.get(char, ("sil", 0.0))
            visemes.append(
                Viseme(
                    phoneme=phoneme,
                    weight=weight,
                    timestamp_ms=i * ms_per_char,
                )
            )
        return visemes

    # -- Internal -----------------------------------------------------------

    def _active_animation(self) -> str:
        if self._is_talking:
            return self._config.talk_animation
        if self._is_thinking:
            return self._config.think_animation
        return self._config.idle_animation


# ---------------------------------------------------------------------------
# Built-in avatar presets
# ---------------------------------------------------------------------------

AVATAR_PRESETS: dict[str, AvatarConfig] = {
    "aria": AvatarConfig(
        style=AvatarStyle.ANIME,
        idle_animation="idle_float",
        talk_animation="talk_gesture",
        think_animation="think_glow",
        expression_set={
            "determined": {"browInnerUp": 0.3, "mouthSmile": 0.2, "eyeSquint": 0.4},
        },
        metadata={
            "color_primary": "#00C9A7",
            "color_secondary": "#1A1A2E",
            "accent": "feather",
            "description": "ARIA — anime style, teal/green color scheme, feather accent",
        },
    ),
    "assistant": AvatarConfig(
        style=AvatarStyle.MINIMAL,
        idle_animation="idle_subtle",
        talk_animation="talk_minimal",
        think_animation="think_pulse",
        metadata={
            "color_primary": "#4A90D9",
            "color_secondary": "#F5F5F5",
            "description": "Professional minimal assistant",
        },
    ),
    "companion": AvatarConfig(
        style=AvatarStyle.CHIBI,
        idle_animation="idle_bounce",
        talk_animation="talk_wiggle",
        think_animation="think_sparkle",
        metadata={
            "color_primary": "#FF6B9D",
            "color_secondary": "#FFF0F5",
            "description": "Chibi style, friendly companion",
        },
    ),
}
