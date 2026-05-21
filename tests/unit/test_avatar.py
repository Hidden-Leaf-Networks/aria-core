"""Tests for 3D Avatar System.

ARIA-310
"""

from __future__ import annotations

import pytest

from aria_core.avatar import (
    AVATAR_PRESETS,
    AvatarConfig,
    AvatarRenderer,
    AvatarState,
    AvatarStyle,
    Viseme,
)


# ---------------------------------------------------------------------------
# Model tests
# ---------------------------------------------------------------------------


class TestViseme:
    def test_valid_viseme(self) -> None:
        v = Viseme(phoneme="aa", weight=0.8, timestamp_ms=100)
        assert v.phoneme == "aa"
        assert v.weight == 0.8
        assert v.timestamp_ms == 100

    def test_weight_bounds(self) -> None:
        Viseme(phoneme="aa", weight=0.0, timestamp_ms=0)
        Viseme(phoneme="aa", weight=1.0, timestamp_ms=0)
        with pytest.raises(Exception):
            Viseme(phoneme="aa", weight=1.1, timestamp_ms=0)
        with pytest.raises(Exception):
            Viseme(phoneme="aa", weight=-0.1, timestamp_ms=0)

    def test_negative_timestamp_rejected(self) -> None:
        with pytest.raises(Exception):
            Viseme(phoneme="aa", weight=0.5, timestamp_ms=-1)


class TestAvatarConfig:
    def test_defaults(self) -> None:
        cfg = AvatarConfig()
        assert cfg.style == AvatarStyle.ANIME
        assert cfg.model_url is None
        assert cfg.idle_animation == "idle_breathe"
        assert cfg.voice_id is None
        assert cfg.expression_set == {}

    def test_all_styles(self) -> None:
        for style in AvatarStyle:
            cfg = AvatarConfig(style=style)
            assert cfg.style == style

    def test_custom_expression_set(self) -> None:
        cfg = AvatarConfig(expression_set={"wink": {"eyeLeft": 0.0}})
        assert "wink" in cfg.expression_set


class TestAvatarState:
    def test_defaults(self) -> None:
        state = AvatarState()
        assert state.current_expression == "neutral"
        assert state.is_talking is False
        assert state.is_thinking is False
        assert state.visemes == []
        assert state.audio_url is None


# ---------------------------------------------------------------------------
# Renderer tests
# ---------------------------------------------------------------------------


class TestAvatarRenderer:
    def test_init_defaults(self) -> None:
        r = AvatarRenderer(AvatarConfig())
        state = r.get_state()
        assert state.current_expression == "neutral"
        assert state.is_talking is False
        assert state.is_thinking is False

    def test_set_expression_valid(self) -> None:
        r = AvatarRenderer(AvatarConfig())
        r.set_expression("happy")
        assert r.get_state().current_expression == "happy"

    def test_set_expression_invalid(self) -> None:
        r = AvatarRenderer(AvatarConfig())
        with pytest.raises(ValueError, match="Unknown expression"):
            r.set_expression("nonexistent")

    def test_start_talking_generates_visemes(self) -> None:
        r = AvatarRenderer(AvatarConfig())
        visemes = r.start_talking("hi")
        assert len(visemes) == 2
        state = r.get_state()
        assert state.is_talking is True
        assert state.current_expression == "speaking"
        assert len(state.visemes) == 2

    def test_stop_talking_clears_state(self) -> None:
        r = AvatarRenderer(AvatarConfig())
        r.start_talking("hi")
        r.stop_talking()
        state = r.get_state()
        assert state.is_talking is False
        assert state.visemes == []
        assert state.current_expression == "neutral"

    def test_start_thinking(self) -> None:
        r = AvatarRenderer(AvatarConfig())
        r.start_thinking()
        state = r.get_state()
        assert state.is_thinking is True
        assert state.current_expression == "thinking"

    def test_stop_thinking(self) -> None:
        r = AvatarRenderer(AvatarConfig())
        r.start_thinking()
        r.stop_thinking()
        state = r.get_state()
        assert state.is_thinking is False
        assert state.current_expression == "neutral"

    def test_text_to_visemes_basic(self) -> None:
        r = AvatarRenderer(AvatarConfig())
        visemes = r.text_to_visemes("ab")
        assert len(visemes) == 2
        assert visemes[0].phoneme == "aa"
        assert visemes[1].phoneme == "pp"  # 'b' maps to pp
        assert visemes[0].timestamp_ms == 0
        assert visemes[1].timestamp_ms == 80

    def test_text_to_visemes_space(self) -> None:
        r = AvatarRenderer(AvatarConfig())
        visemes = r.text_to_visemes("a b")
        assert visemes[1].phoneme == "sil"
        assert visemes[1].weight == 0.0

    def test_custom_expression_merged(self) -> None:
        cfg = AvatarConfig(expression_set={"custom1": {"x": 1.0}})
        r = AvatarRenderer(cfg)
        r.set_expression("custom1")
        assert r.get_state().current_expression == "custom1"
        # Built-ins still available
        r.set_expression("happy")
        assert r.get_state().current_expression == "happy"

    def test_active_animation_metadata(self) -> None:
        r = AvatarRenderer(AvatarConfig())
        state = r.get_state()
        assert state.metadata["active_animation"] == "idle_breathe"
        r.start_talking("x")
        assert r.get_state().metadata["active_animation"] == "talk_default"
        r.stop_talking()
        r.start_thinking()
        assert r.get_state().metadata["active_animation"] == "think_head_tilt"


# ---------------------------------------------------------------------------
# Preset tests
# ---------------------------------------------------------------------------


class TestAvatarPresets:
    def test_aria_preset(self) -> None:
        cfg = AVATAR_PRESETS["aria"]
        assert cfg.style == AvatarStyle.ANIME
        assert "feather" in cfg.metadata.get("accent", "")

    def test_assistant_preset(self) -> None:
        cfg = AVATAR_PRESETS["assistant"]
        assert cfg.style == AvatarStyle.MINIMAL

    def test_companion_preset(self) -> None:
        cfg = AVATAR_PRESETS["companion"]
        assert cfg.style == AvatarStyle.CHIBI

    def test_all_presets_renderable(self) -> None:
        for name, cfg in AVATAR_PRESETS.items():
            r = AvatarRenderer(cfg)
            state = r.get_state()
            assert state.current_expression == "neutral"
