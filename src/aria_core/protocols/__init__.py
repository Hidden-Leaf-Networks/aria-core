"""W3C Agent Protocol compatibility layer.

Bridges aria-core's A2A and MCP protocols to the emerging W3C AI Agent
Protocol patterns (Community Group, 2026-2027 timeline).

Usage:
    from aria_core.protocols.w3c import W3CProtocolAdapter, ProtocolBridge

    adapter = W3CProtocolAdapter("aria-agent", "https://agent.example.com")
    descriptor = adapter.get_descriptor()

    bridge = ProtocolBridge()
    bridge.register_adapter("w3c", adapter)
"""

from aria_core.protocols.w3c import (
    ProtocolBridge,
    W3CAgentDescriptor,
    W3CMessage,
    W3CProtocolAdapter,
)

__all__ = [
    "ProtocolBridge",
    "W3CAgentDescriptor",
    "W3CMessage",
    "W3CProtocolAdapter",
]
