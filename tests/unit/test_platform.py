"""Tests for Multi-Platform Agent Presence.

ARIA-311
"""

from __future__ import annotations

import pytest

from aria_core.platform import (
    DiscordAdapter,
    Platform,
    PlatformAdapter,
    PlatformConfig,
    PresenceManager,
    SlackAdapter,
    TelegramAdapter,
    WebChatAdapter,
    XAdapter,
)
from aria_core.platform.presence import adapter_from_config


# ---------------------------------------------------------------------------
# Model tests
# ---------------------------------------------------------------------------


class TestPlatformConfig:
    def test_defaults(self) -> None:
        cfg = PlatformConfig(platform=Platform.DISCORD)
        assert cfg.platform == Platform.DISCORD
        assert cfg.api_key == ""
        assert cfg.channel_ids == []
        assert cfg.webhook_url is None
        assert cfg.enabled is True

    def test_all_platforms(self) -> None:
        for p in Platform:
            cfg = PlatformConfig(platform=p, api_key="k")
            assert cfg.platform == p


# ---------------------------------------------------------------------------
# Adapter tests
# ---------------------------------------------------------------------------


class TestAdapters:
    def test_discord_adapter_name(self) -> None:
        a = DiscordAdapter(PlatformConfig(platform=Platform.DISCORD))
        assert a.name == "discord"
        assert a.connected is True

    def test_slack_adapter_name(self) -> None:
        a = SlackAdapter(PlatformConfig(platform=Platform.SLACK))
        assert a.name == "slack"

    def test_x_adapter_name(self) -> None:
        a = XAdapter(PlatformConfig(platform=Platform.X))
        assert a.name == "x"

    def test_telegram_adapter_name(self) -> None:
        a = TelegramAdapter(PlatformConfig(platform=Platform.TELEGRAM))
        assert a.name == "telegram"

    def test_webchat_adapter_name(self) -> None:
        a = WebChatAdapter(PlatformConfig(platform=Platform.WEB_CHAT))
        assert a.name == "web_chat"

    @pytest.mark.asyncio
    async def test_send_message(self) -> None:
        a = DiscordAdapter(PlatformConfig(platform=Platform.DISCORD, enabled=True))
        result = await a.send_message("ch1", "hello")
        assert result["ok"] is True
        assert result["text"] == "hello"
        assert result["channel_id"] == "ch1"
        assert result["platform"] == "discord"

    @pytest.mark.asyncio
    async def test_send_when_disconnected(self) -> None:
        a = DiscordAdapter(PlatformConfig(platform=Platform.DISCORD, enabled=False))
        result = await a.send_message("ch1", "hello")
        assert result["ok"] is False
        assert result["error"] == "not_connected"

    @pytest.mark.asyncio
    async def test_receive_messages_empty(self) -> None:
        a = SlackAdapter(PlatformConfig(platform=Platform.SLACK))
        msgs = await a.receive_messages("ch1")
        assert msgs == []

    def test_from_config_factory(self) -> None:
        cfg = PlatformConfig(platform=Platform.X, api_key="key")
        a = XAdapter.from_config(cfg)
        assert a.name == "x"
        assert a.connected is True

    def test_adapter_from_config_factory(self) -> None:
        cfg = PlatformConfig(platform=Platform.TELEGRAM)
        a = adapter_from_config(cfg)
        assert a.name == "telegram"

    def test_adapter_satisfies_protocol(self) -> None:
        a = DiscordAdapter(PlatformConfig(platform=Platform.DISCORD))
        assert isinstance(a, PlatformAdapter)


# ---------------------------------------------------------------------------
# PresenceManager tests
# ---------------------------------------------------------------------------


class TestPresenceManager:
    def _make_mgr(self) -> PresenceManager:
        mgr = PresenceManager(tenant_id="t1", agent_id="aria")
        mgr.register_platform(
            PlatformConfig(
                platform=Platform.DISCORD,
                api_key="dk",
                channel_ids=["general", "bot"],
            )
        )
        mgr.register_platform(
            PlatformConfig(
                platform=Platform.SLACK,
                api_key="sk",
                channel_ids=["random"],
            )
        )
        return mgr

    def test_register_and_list(self) -> None:
        mgr = self._make_mgr()
        platforms = mgr.list_platforms()
        assert len(platforms) == 2
        names = {str(p.platform) for p in platforms}
        assert names == {"discord", "slack"}

    def test_unregister(self) -> None:
        mgr = self._make_mgr()
        mgr.unregister_platform("discord")
        assert len(mgr.list_platforms()) == 1

    def test_unregister_unknown_raises(self) -> None:
        mgr = PresenceManager(tenant_id="t1", agent_id="aria")
        with pytest.raises(KeyError):
            mgr.unregister_platform("nonexistent")

    @pytest.mark.asyncio
    async def test_broadcast(self) -> None:
        mgr = self._make_mgr()
        results = await mgr.broadcast("hello everyone")
        assert "discord" in results
        assert "slack" in results
        assert results["discord"]["ok"] is True
        # Discord has 2 channels
        assert len(results["discord"]["channels"]) == 2

    @pytest.mark.asyncio
    async def test_send_to(self) -> None:
        mgr = self._make_mgr()
        result = await mgr.send_to("discord", "general", "targeted")
        assert result["ok"] is True
        assert result["text"] == "targeted"

    @pytest.mark.asyncio
    async def test_send_to_unknown_platform(self) -> None:
        mgr = PresenceManager(tenant_id="t1", agent_id="aria")
        with pytest.raises(KeyError):
            await mgr.send_to("nonexistent", "ch", "hi")

    @pytest.mark.asyncio
    async def test_receive_from(self) -> None:
        mgr = self._make_mgr()
        msgs = await mgr.receive_from("slack", "random")
        assert msgs == []

    @pytest.mark.asyncio
    async def test_receive_from_unknown_platform(self) -> None:
        mgr = PresenceManager(tenant_id="t1", agent_id="aria")
        with pytest.raises(KeyError):
            await mgr.receive_from("nonexistent", "ch")

    def test_get_unified_context(self) -> None:
        mgr = self._make_mgr()
        ctx = mgr.get_unified_context("user123")
        assert ctx["user_id"] == "user123"
        assert ctx["tenant_id"] == "t1"
        assert ctx["agent_id"] == "aria"
        assert ctx["platform_count"] == 2
        assert ctx["connected_count"] == 2
        assert "discord" in ctx["platforms"]
        assert "slack" in ctx["platforms"]

    def test_get_status(self) -> None:
        mgr = self._make_mgr()
        status = mgr.get_status()
        assert status["tenant_id"] == "t1"
        assert status["total_platforms"] == 2
        assert status["connected_platforms"] == 2
        assert status["platforms"]["discord"]["connected"] is True

    def test_properties(self) -> None:
        mgr = PresenceManager(tenant_id="t1", agent_id="aria")
        assert mgr.tenant_id == "t1"
        assert mgr.agent_id == "aria"
