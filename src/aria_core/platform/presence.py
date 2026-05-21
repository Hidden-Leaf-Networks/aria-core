"""Multi-platform agent presence — unified messaging across platforms.

Provides stub adapters for Discord, Slack, X, Telegram, and WebChat,
plus a PresenceManager that orchestrates cross-platform broadcasting
and context unification.

ARIA-311
"""

from __future__ import annotations

import sys
import time
from typing import Any, Protocol, runtime_checkable

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
# Enums & Models
# ---------------------------------------------------------------------------


class Platform(StrEnum):
    DISCORD = "discord"
    SLACK = "slack"
    X = "x"
    TELEGRAM = "telegram"
    WEB_CHAT = "web_chat"


class PlatformConfig(BaseModel):
    """Connection configuration for a single platform."""

    platform: Platform
    api_key: str = ""
    channel_ids: list[str] = Field(default_factory=list)
    webhook_url: str | None = None
    enabled: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# PlatformAdapter protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class PlatformAdapter(Protocol):
    """Protocol that all platform adapters must satisfy."""

    @property
    def name(self) -> str: ...

    @property
    def connected(self) -> bool: ...

    async def send_message(
        self, channel_id: str, text: str, metadata: dict[str, Any] | None = None
    ) -> dict[str, Any]: ...

    async def receive_messages(
        self, channel_id: str, limit: int = 10
    ) -> list[dict[str, Any]]: ...


# ---------------------------------------------------------------------------
# Base adapter with shared logic
# ---------------------------------------------------------------------------


class _BaseAdapter:
    """Shared stub implementation for platform adapters."""

    _platform_name: str = "base"

    def __init__(self, config: PlatformConfig) -> None:
        self._config = config
        self._connected = config.enabled
        self._sent: list[dict[str, Any]] = []
        self._inbox: list[dict[str, Any]] = []

    @classmethod
    def from_config(cls, config: PlatformConfig) -> _BaseAdapter:
        """Factory — create an adapter from a PlatformConfig."""
        return cls(config)

    @property
    def name(self) -> str:
        return self._platform_name

    @property
    def connected(self) -> bool:
        return self._connected

    async def send_message(
        self, channel_id: str, text: str, metadata: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Simulate sending a message. Returns a delivery receipt."""
        if not self._connected:
            return {"ok": False, "error": "not_connected", "platform": self.name}
        record = {
            "ok": True,
            "platform": self.name,
            "channel_id": channel_id,
            "text": text,
            "metadata": metadata or {},
            "timestamp_ms": int(time.time() * 1000),
        }
        self._sent.append(record)
        return record

    async def receive_messages(
        self, channel_id: str, limit: int = 10
    ) -> list[dict[str, Any]]:
        """Return simulated inbox messages (up to *limit*)."""
        matching = [m for m in self._inbox if m.get("channel_id") == channel_id]
        return matching[:limit]


# ---------------------------------------------------------------------------
# Concrete stub adapters
# ---------------------------------------------------------------------------


class DiscordAdapter(_BaseAdapter):
    _platform_name = "discord"


class SlackAdapter(_BaseAdapter):
    _platform_name = "slack"


class XAdapter(_BaseAdapter):
    _platform_name = "x"


class TelegramAdapter(_BaseAdapter):
    _platform_name = "telegram"


class WebChatAdapter(_BaseAdapter):
    _platform_name = "web_chat"


# ---------------------------------------------------------------------------
# Adapter factory
# ---------------------------------------------------------------------------

_ADAPTER_MAP: dict[str, type[_BaseAdapter]] = {
    Platform.DISCORD: DiscordAdapter,
    Platform.SLACK: SlackAdapter,
    Platform.X: XAdapter,
    Platform.TELEGRAM: TelegramAdapter,
    Platform.WEB_CHAT: WebChatAdapter,
}


def adapter_from_config(config: PlatformConfig) -> _BaseAdapter:
    """Create the correct adapter for a given PlatformConfig."""
    cls = _ADAPTER_MAP.get(str(config.platform))
    if cls is None:
        raise ValueError(f"Unsupported platform: {config.platform}")
    return cls.from_config(config)


# ---------------------------------------------------------------------------
# PresenceManager
# ---------------------------------------------------------------------------


class PresenceManager:
    """Orchestrates agent presence across multiple platforms.

    Usage::

        mgr = PresenceManager(tenant_id="t1", agent_id="aria")
        mgr.register_platform(PlatformConfig(platform="discord", api_key="..."))
        results = await mgr.broadcast("Hello from all platforms!")
    """

    def __init__(self, tenant_id: str, agent_id: str) -> None:
        self._tenant_id = tenant_id
        self._agent_id = agent_id
        self._platforms: dict[str, PlatformConfig] = {}
        self._adapters: dict[str, _BaseAdapter] = {}
        self._user_contexts: dict[str, dict[str, Any]] = {}

    # -- Properties ---------------------------------------------------------

    @property
    def tenant_id(self) -> str:
        return self._tenant_id

    @property
    def agent_id(self) -> str:
        return self._agent_id

    # -- Platform lifecycle -------------------------------------------------

    def register_platform(self, config: PlatformConfig) -> None:
        """Register a platform and instantiate its adapter."""
        name = str(config.platform)
        self._platforms[name] = config
        self._adapters[name] = adapter_from_config(config)

    def unregister_platform(self, platform_name: str) -> None:
        """Remove a platform by name. Raises KeyError if not found."""
        if platform_name not in self._platforms:
            raise KeyError(f"Platform '{platform_name}' is not registered")
        del self._platforms[platform_name]
        del self._adapters[platform_name]

    def list_platforms(self) -> list[PlatformConfig]:
        """Return all registered platform configs."""
        return list(self._platforms.values())

    # -- Messaging ----------------------------------------------------------

    async def broadcast(
        self, text: str, metadata: dict[str, Any] | None = None
    ) -> dict[str, dict[str, Any]]:
        """Send *text* to ALL registered platforms (all channels).

        Returns a dict mapping ``platform_name -> result``.
        """
        results: dict[str, dict[str, Any]] = {}
        meta = metadata or {}
        for name, adapter in self._adapters.items():
            config = self._platforms[name]
            platform_results: list[dict[str, Any]] = []
            for ch in config.channel_ids:
                res = await adapter.send_message(ch, text, meta)
                platform_results.append(res)
            results[name] = {
                "ok": all(r.get("ok") for r in platform_results),
                "channels": platform_results,
            }
        return results

    async def send_to(
        self, platform: str, channel_id: str, text: str
    ) -> dict[str, Any]:
        """Send *text* to a specific platform + channel."""
        adapter = self._adapters.get(platform)
        if adapter is None:
            raise KeyError(f"Platform '{platform}' is not registered")
        return await adapter.send_message(channel_id, text)

    async def receive_from(
        self, platform: str, channel_id: str
    ) -> list[dict[str, Any]]:
        """Receive messages from a specific platform + channel."""
        adapter = self._adapters.get(platform)
        if adapter is None:
            raise KeyError(f"Platform '{platform}' is not registered")
        return await adapter.receive_messages(channel_id)

    # -- Context unification ------------------------------------------------

    def get_unified_context(self, user_id: str) -> dict[str, Any]:
        """Merge context for a user across all registered platforms.

        Returns a dict with platform-keyed sub-contexts plus summary fields.
        """
        ctx: dict[str, Any] = {
            "user_id": user_id,
            "tenant_id": self._tenant_id,
            "agent_id": self._agent_id,
            "platforms": {},
        }
        for name, config in self._platforms.items():
            adapter = self._adapters[name]
            ctx["platforms"][name] = {
                "connected": adapter.connected,
                "channels": config.channel_ids,
                "metadata": config.metadata,
            }
        ctx["platform_count"] = len(self._platforms)
        ctx["connected_count"] = sum(
            1 for a in self._adapters.values() if a.connected
        )
        return ctx

    # -- Status -------------------------------------------------------------

    def get_status(self) -> dict[str, Any]:
        """Return a status dict showing which platforms are active."""
        platform_statuses: dict[str, dict[str, Any]] = {}
        for name, adapter in self._adapters.items():
            config = self._platforms[name]
            platform_statuses[name] = {
                "connected": adapter.connected,
                "enabled": config.enabled,
                "channels": config.channel_ids,
            }
        return {
            "tenant_id": self._tenant_id,
            "agent_id": self._agent_id,
            "total_platforms": len(self._platforms),
            "connected_platforms": sum(
                1 for a in self._adapters.values() if a.connected
            ),
            "platforms": platform_statuses,
        }
