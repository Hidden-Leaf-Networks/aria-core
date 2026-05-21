"""W3C AI Agent Protocol compatibility adapter.

Implements data models and translation logic based on the emerging W3C AI
Agent Protocol Community Group specification patterns.  The adapter bridges
aria-core's native A2A and MCP representations to the W3C wire format so
that aria-core agents can participate in W3C-compliant discovery, messaging,
and capability negotiation without changes to the core runtime.

Spec tracking:
    https://www.w3.org/community/ai-agent-protocol/
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

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


# ---------------------------------------------------------------------------
# W3C-aligned data models
# ---------------------------------------------------------------------------


class W3CAgentDescriptor(BaseModel):
    """W3C-compatible agent descriptor for discovery and capability exchange.

    Based on the emerging W3C AI Agent Protocol patterns for agent identity
    and capability advertisement.  The ``id`` field uses URI format per the
    W3C convention (e.g. ``urn:agent:aria-core:abc123``).
    """

    id: str = Field(default_factory=lambda: f"urn:agent:aria-core:{uuid4().hex[:12]}")
    name: str
    description: str = ""
    version: str = "1.0.0"
    provider: dict[str, Any] = Field(default_factory=lambda: {"name": "Unknown", "url": ""})
    capabilities: list[str] = Field(default_factory=list)
    interfaces: list[dict[str, Any]] = Field(default_factory=list)
    authentication: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class W3CMessageType(StrEnum):
    """Message types in the W3C agent messaging pattern."""

    REQUEST = "request"
    RESPONSE = "response"
    NOTIFICATION = "notification"
    ERROR = "error"


class W3CContentType(StrEnum):
    """Supported content types for W3C messages."""

    TEXT = "text"
    JSON = "json"
    BINARY = "binary"
    MULTIPART = "multipart"


class W3CMessage(BaseModel):
    """W3C-compatible agent message envelope.

    Provides a uniform message format that can carry payloads between agents
    regardless of the underlying transport (REST, WebSocket, A2A, etc.).
    """

    id: str = Field(default_factory=lambda: str(uuid4()))
    type: W3CMessageType = W3CMessageType.REQUEST
    sender: str = ""
    receiver: str = ""
    content_type: W3CContentType = W3CContentType.TEXT
    body: Any = None
    headers: dict[str, Any] = Field(default_factory=dict)
    correlation_id: str | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ---------------------------------------------------------------------------
# Protocol priority used during negotiation
# ---------------------------------------------------------------------------

PROTOCOL_PRIORITY: list[str] = ["w3c", "a2a", "mcp", "rest", "websocket"]

# Mapping from A2A skill tags to W3C capability identifiers
_SKILL_TAG_TO_CAPABILITY: dict[str, str] = {
    "planning": "planning",
    "orchestration": "planning",
    "execution": "tool-use",
    "code": "code-generation",
    "engineering": "code-generation",
    "development": "code-generation",
    "research": "text-generation",
    "analysis": "text-generation",
    "reports": "text-generation",
    "security": "tool-use",
    "risk": "tool-use",
    "compliance": "tool-use",
}


# ---------------------------------------------------------------------------
# W3CProtocolAdapter
# ---------------------------------------------------------------------------


class W3CProtocolAdapter:
    """Translates between aria-core native formats and W3C agent protocol.

    Parameters
    ----------
    agent_name:
        Human-readable name for the agent in generated descriptors.
    base_url:
        Root URL where the agent's endpoints are reachable.
    """

    def __init__(self, agent_name: str, base_url: str) -> None:
        self.agent_name = agent_name
        self.base_url = base_url.rstrip("/")
        self._descriptor_id = f"urn:agent:aria-core:{uuid4().hex[:12]}"

    # -- descriptor generation ---------------------------------------------

    def get_descriptor(self) -> W3CAgentDescriptor:
        """Generate a W3C-compatible descriptor from aria-core defaults."""
        import aria_core

        return W3CAgentDescriptor(
            id=self._descriptor_id,
            name=self.agent_name,
            description=(
                f"Aria Core v{aria_core.__version__} agent with deterministic "
                "execution, multi-model consensus, and risk scoring."
            ),
            version=aria_core.__version__,
            provider={"name": "Hidden Leaf Networks", "url": "https://hiddenleafnetworks.com"},
            capabilities=[
                "text-generation",
                "tool-use",
                "planning",
                "code-generation",
            ],
            interfaces=[
                {"protocol": "a2a", "endpoint": f"{self.base_url}/.well-known/a2a/agent-card"},
                {"protocol": "mcp", "endpoint": f"{self.base_url}/mcp"},
                {"protocol": "rest", "endpoint": f"{self.base_url}/api/v1"},
                {"protocol": "websocket", "endpoint": f"{self.base_url}/ws"},
            ],
            authentication=[
                {"type": "bearer", "scheme": "JWT"},
                {"type": "api_key", "in": "header", "name": "X-API-Key"},
            ],
            metadata={"framework": "aria-core", "spec_version": "0.1.0-draft"},
        )

    # -- AgentCard <-> W3CAgentDescriptor ----------------------------------

    def from_aria_agent_card(self, card: Any) -> W3CAgentDescriptor:
        """Convert an A2A ``AgentCard`` to a ``W3CAgentDescriptor``."""
        capabilities: list[str] = []
        seen: set[str] = set()

        for skill in getattr(card, "skills", []):
            for tag in getattr(skill, "tags", []):
                cap = _SKILL_TAG_TO_CAPABILITY.get(tag, tag)
                if cap not in seen:
                    seen.add(cap)
                    capabilities.append(cap)

        if getattr(card, "capabilities", None) and getattr(card.capabilities, "streaming", False):
            if "streaming" not in seen:
                capabilities.append("streaming")

        interfaces: list[dict[str, Any]] = []
        card_url = getattr(card, "url", None) or self.base_url
        interfaces.append({"protocol": "a2a", "endpoint": f"{card_url}/.well-known/a2a/agent-card"})

        provider_data: dict[str, Any] = {"name": "Unknown", "url": ""}
        if hasattr(card, "provider") and card.provider:
            provider_data = {
                "name": getattr(card.provider, "organization", "Unknown"),
                "url": getattr(card.provider, "url", "") or "",
            }

        auth_schemes: list[dict[str, Any]] = []
        for scheme in getattr(card, "security_schemes", []):
            auth_schemes.append(scheme if isinstance(scheme, dict) else {"type": str(scheme)})

        return W3CAgentDescriptor(
            id=f"urn:agent:aria-core:{getattr(card, 'id', uuid4().hex[:12])}",
            name=getattr(card, "name", "Unknown"),
            description=getattr(card, "description", ""),
            version=getattr(card, "version", "1.0.0"),
            provider=provider_data,
            capabilities=capabilities,
            interfaces=interfaces,
            authentication=auth_schemes,
            metadata=getattr(card, "metadata", {}),
        )

    def to_aria_agent_card(self, descriptor: W3CAgentDescriptor) -> dict[str, Any]:
        """Convert a ``W3CAgentDescriptor`` back to an A2A AgentCard dict."""
        skills: list[dict[str, Any]] = []
        for cap in descriptor.capabilities:
            skills.append({
                "id": cap,
                "name": cap.replace("-", " ").title(),
                "description": f"W3C capability: {cap}",
                "tags": [cap],
                "examples": [],
            })

        a2a_endpoint = ""
        for iface in descriptor.interfaces:
            if iface.get("protocol") == "a2a":
                a2a_endpoint = iface.get("endpoint", "")
                break

        streaming = "streaming" in descriptor.capabilities

        return {
            "id": descriptor.id,
            "name": descriptor.name,
            "description": descriptor.description,
            "version": descriptor.version,
            "provider": {
                "organization": descriptor.provider.get("name", "Unknown"),
                "url": descriptor.provider.get("url", ""),
            },
            "capabilities": {
                "streaming": streaming,
                "push_notifications": False,
                "extended_agent_card": False,
            },
            "skills": skills,
            "security_schemes": descriptor.authentication,
            "url": a2a_endpoint,
            "metadata": descriptor.metadata,
        }

    # -- A2AMessage <-> W3CMessage -----------------------------------------

    def from_aria_message(self, message: Any) -> W3CMessage:
        """Convert an A2A ``A2AMessage`` to a ``W3CMessage``."""
        role = getattr(message, "role", "user")
        msg_type = W3CMessageType.REQUEST if role == "user" else W3CMessageType.RESPONSE

        parts = getattr(message, "parts", [])
        if len(parts) == 1 and getattr(parts[0], "text", None):
            content_type = W3CContentType.TEXT
            body = parts[0].text
        elif any(getattr(p, "structured_data", None) for p in parts):
            content_type = W3CContentType.JSON
            body = [
                {
                    "text": getattr(p, "text", None),
                    "data": getattr(p, "structured_data", None),
                    "file_uri": getattr(p, "file_uri", None),
                    "mime_type": getattr(p, "mime_type", None),
                }
                for p in parts
            ]
        elif len(parts) > 1:
            content_type = W3CContentType.MULTIPART
            body = [
                {
                    "text": getattr(p, "text", None),
                    "file_uri": getattr(p, "file_uri", None),
                    "mime_type": getattr(p, "mime_type", None),
                }
                for p in parts
            ]
        else:
            content_type = W3CContentType.TEXT
            body = getattr(parts[0], "text", "") if parts else ""

        return W3CMessage(
            id=getattr(message, "id", str(uuid4())),
            type=msg_type,
            sender=f"urn:agent:role:{role}",
            receiver="",
            content_type=content_type,
            body=body,
            headers=getattr(message, "metadata", {}),
            correlation_id=getattr(message, "task_id", None) or getattr(message, "context_id", None),
            timestamp=getattr(message, "timestamp", datetime.now(timezone.utc)),
        )

    def to_aria_message(self, w3c_msg: W3CMessage) -> dict[str, Any]:
        """Convert a ``W3CMessage`` to an A2A-compatible message dict."""
        role = "agent" if w3c_msg.type == W3CMessageType.RESPONSE else "user"

        parts: list[dict[str, Any]] = []
        if w3c_msg.content_type == W3CContentType.TEXT:
            parts.append({"text": str(w3c_msg.body) if w3c_msg.body is not None else ""})
        elif w3c_msg.content_type == W3CContentType.JSON:
            if isinstance(w3c_msg.body, list):
                for item in w3c_msg.body:
                    parts.append({
                        "text": item.get("text"),
                        "structured_data": item.get("data"),
                        "file_uri": item.get("file_uri"),
                        "mime_type": item.get("mime_type"),
                    })
            else:
                parts.append({"structured_data": w3c_msg.body})
        elif w3c_msg.content_type == W3CContentType.MULTIPART:
            if isinstance(w3c_msg.body, list):
                for item in w3c_msg.body:
                    parts.append({
                        "text": item.get("text") if isinstance(item, dict) else str(item),
                    })
            else:
                parts.append({"text": str(w3c_msg.body)})
        else:
            parts.append({"text": str(w3c_msg.body) if w3c_msg.body is not None else ""})

        return {
            "id": w3c_msg.id,
            "role": role,
            "parts": parts,
            "context_id": w3c_msg.correlation_id,
            "task_id": w3c_msg.correlation_id,
            "timestamp": w3c_msg.timestamp.isoformat(),
            "metadata": w3c_msg.headers,
        }

    # -- MCP tool <-> W3C capability ---------------------------------------

    def from_mcp_tool(self, tool: dict[str, Any]) -> dict[str, Any]:
        """Convert an MCP tool definition to a W3C capability descriptor."""
        input_schema = tool.get("inputSchema") or tool.get("input_schema", {})
        return {
            "id": f"urn:capability:mcp:{tool.get('name', 'unknown')}",
            "name": tool.get("name", "unknown"),
            "description": tool.get("description", ""),
            "type": "tool",
            "protocol_origin": "mcp",
            "parameters": input_schema,
            "metadata": {
                k: v for k, v in tool.items()
                if k not in ("name", "description", "inputSchema", "input_schema")
            },
        }

    def to_mcp_tool(self, capability: dict[str, Any]) -> dict[str, Any]:
        """Convert a W3C capability descriptor to an MCP tool definition."""
        return {
            "name": capability.get("name", "unknown"),
            "description": capability.get("description", ""),
            "inputSchema": capability.get("parameters", {}),
        }

    # -- protocol negotiation ----------------------------------------------

    def negotiate_protocol(
        self,
        local_caps: list[str],
        remote_caps: list[str],
    ) -> str:
        """Determine the best shared protocol between local and remote agents.

        Uses a priority list: w3c > a2a > mcp > rest > websocket.
        Returns the highest-priority protocol present in both capability sets.
        Falls back to ``"rest"`` when no overlap is found.
        """
        local_set = set(local_caps)
        remote_set = set(remote_caps)
        common = local_set & remote_set

        for proto in PROTOCOL_PRIORITY:
            if proto in common:
                return proto

        return "rest"

    # -- discovery document ------------------------------------------------

    def create_discovery_document(self) -> dict[str, Any]:
        """Generate a ``/.well-known/ai-agent`` discovery document."""
        descriptor = self.get_descriptor()
        return {
            "@context": "https://www.w3.org/ns/ai-agent/v1",
            "@type": "AgentDescriptor",
            "id": descriptor.id,
            "name": descriptor.name,
            "description": descriptor.description,
            "version": descriptor.version,
            "provider": descriptor.provider,
            "capabilities": descriptor.capabilities,
            "interfaces": descriptor.interfaces,
            "authentication": descriptor.authentication,
            "metadata": descriptor.metadata,
            "discovery": {
                "well_known_path": "/.well-known/ai-agent",
                "a2a_path": "/.well-known/a2a/agent-card",
                "mcp_endpoint": f"{self.base_url}/mcp",
                "api_docs": f"{self.base_url}/docs",
            },
        }


# ---------------------------------------------------------------------------
# ProtocolBridge — routes messages across protocol boundaries
# ---------------------------------------------------------------------------


class ProtocolBridge:
    """Routes and translates messages between aria-core's supported protocols.

    Register one or more ``W3CProtocolAdapter`` instances (or compatible
    objects) keyed by protocol name.  The bridge can then translate payloads
    from one protocol to another using the registered adapters.
    """

    def __init__(self) -> None:
        self._adapters: dict[str, W3CProtocolAdapter] = {}

    def register_adapter(self, protocol_name: str, adapter: W3CProtocolAdapter) -> None:
        """Register an adapter under a protocol name."""
        self._adapters[protocol_name] = adapter

    def get_supported_protocols(self) -> list[str]:
        """Return the list of currently registered protocol names."""
        return list(self._adapters.keys())

    def translate(self, message: Any, from_protocol: str, to_protocol: str) -> Any:
        """Translate *message* from one protocol representation to another.

        Supported translation paths:

        * ``a2a -> w3c`` : A2AMessage -> W3CMessage
        * ``w3c -> a2a`` : W3CMessage -> dict (A2A-shaped)
        * ``mcp -> w3c`` : MCP tool dict -> W3C capability dict
        * ``w3c -> mcp`` : W3C capability dict -> MCP tool dict

        Raises ``ValueError`` when no adapter can handle the requested pair.
        """
        pair = (from_protocol, to_protocol)

        # Pick any registered adapter — they all share the same translation
        # logic (stateless), so we just need one.
        adapter = self._resolve_adapter(from_protocol, to_protocol)

        if pair == ("a2a", "w3c"):
            return adapter.from_aria_message(message)
        if pair == ("w3c", "a2a"):
            return adapter.to_aria_message(message)
        if pair == ("mcp", "w3c"):
            return adapter.from_mcp_tool(message)
        if pair == ("w3c", "mcp"):
            return adapter.to_mcp_tool(message)

        raise ValueError(
            f"Unsupported translation path: {from_protocol} -> {to_protocol}. "
            f"Registered protocols: {self.get_supported_protocols()}"
        )

    # -- internal ----------------------------------------------------------

    def _resolve_adapter(self, from_proto: str, to_proto: str) -> W3CProtocolAdapter:
        """Find an adapter that can handle the requested pair."""
        # Prefer an adapter registered under either the source or target name.
        for key in (from_proto, to_proto):
            if key in self._adapters:
                return self._adapters[key]

        # Fall back to the first registered adapter.
        if self._adapters:
            return next(iter(self._adapters.values()))

        raise ValueError(
            f"No adapters registered. Cannot translate {from_proto} -> {to_proto}."
        )
