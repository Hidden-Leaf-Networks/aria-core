"""3D Avatar System — animated avatar rendering with expressions and lip sync.

Provides:
- AvatarConfig: style, model, animations, expressions, voice linkage
- AvatarState: real-time avatar state (expression, talking, thinking, visemes)
- Viseme: phoneme-to-weight mapping for lip sync
- AvatarRenderer: drives avatar state transitions and viseme generation
- Built-in presets: aria, assistant, companion

ARIA-310
"""

from aria_core.avatar.renderer import (
    AVATAR_PRESETS,
    AvatarConfig,
    AvatarRenderer,
    AvatarState,
    AvatarStyle,
    Viseme,
)

__all__ = [
    "AVATAR_PRESETS",
    "AvatarConfig",
    "AvatarRenderer",
    "AvatarState",
    "AvatarStyle",
    "Viseme",
]
