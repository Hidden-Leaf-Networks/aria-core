"""Multi-Platform Agent Presence — unified messaging across platforms.

Provides:
- PlatformAdapter protocol: send/receive on any platform
- PlatformConfig: per-platform connection settings
- PresenceManager: orchestrates multi-platform presence
- Built-in adapters: Discord, Slack, X, Telegram, WebChat

ARIA-311
"""

from aria_core.platform.presence import (
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

__all__ = [
    "DiscordAdapter",
    "Platform",
    "PlatformAdapter",
    "PlatformConfig",
    "PresenceManager",
    "SlackAdapter",
    "TelegramAdapter",
    "WebChatAdapter",
    "XAdapter",
]
