"""Tests for W3C Agent Protocol compatibility layer (ARIA-308)."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from aria_core.a2a.models import (
    AgentCapabilities,
    AgentCard,
    AgentProvider,
    AgentSkill,
    A2AMessage,
    A2APart,
)
from aria_core.protocols.w3c import (
    PROTOCOL_PRIORITY,
    ProtocolBridge,
    W3CAgentDescriptor,
    W3CContentType,
    W3CMessage,
    W3CMessageType,
    W3CProtocolAdapter,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def adapter() -> W3CProtocolAdapter:
    return W3CProtocolAdapter("test-agent", "https://agent.example.com")


@pytest.fixture
def bridge(adapter: W3CProtocolAdapter) -> ProtocolBridge:
    b = ProtocolBridge()
    b.register_adapter("w3c", adapter)
    return b


@pytest.fixture
def sample_card() -> AgentCard:
    return AgentCard.for_aria_core(name="Unit Test Agent", base_url="https://test.local")


@pytest.fixture
def sample_a2a_message() -> A2AMessage:
    return A2AMessage.text("Hello from A2A", role="user")


# ---------------------------------------------------------------------------
# W3CAgentDescriptor model tests
# ---------------------------------------------------------------------------


class TestW3CAgentDescriptor:
    def test_default_id_format(self) -> None:
        desc = W3CAgentDescriptor(name="Test")
        assert desc.id.startswith("urn:agent:aria-core:")

    def test_fields_populated(self) -> None:
        desc = W3CAgentDescriptor(
            name="My Agent",
            description="Does things",
            version="2.0.0",
            capabilities=["text-generation", "tool-use"],
        )
        assert desc.name == "My Agent"
        assert desc.version == "2.0.0"
        assert len(desc.capabilities) == 2

    def test_serialization_roundtrip(self) -> None:
        desc = W3CAgentDescriptor(
            name="Roundtrip",
            provider={"name": "HLN", "url": "https://hln.dev"},
            interfaces=[{"protocol": "rest", "endpoint": "/api"}],
        )
        data = desc.model_dump(mode="json")
        restored = W3CAgentDescriptor.model_validate(data)
        assert restored.name == desc.name
        assert restored.provider == desc.provider


# ---------------------------------------------------------------------------
# W3CMessage model tests
# ---------------------------------------------------------------------------


class TestW3CMessage:
    def test_default_type_is_request(self) -> None:
        msg = W3CMessage(sender="urn:a", receiver="urn:b", body="hi")
        assert msg.type == W3CMessageType.REQUEST

    def test_timestamp_populated(self) -> None:
        msg = W3CMessage(body="check")
        assert isinstance(msg.timestamp, datetime)
        assert msg.timestamp.tzinfo is not None

    def test_correlation_id_optional(self) -> None:
        msg = W3CMessage(body="x")
        assert msg.correlation_id is None

    def test_all_message_types(self) -> None:
        for t in W3CMessageType:
            msg = W3CMessage(type=t, body="test")
            assert msg.type == t


# ---------------------------------------------------------------------------
# W3CProtocolAdapter — descriptor generation
# ---------------------------------------------------------------------------


class TestAdapterDescriptor:
    def test_get_descriptor_basic(self, adapter: W3CProtocolAdapter) -> None:
        desc = adapter.get_descriptor()
        assert desc.name == "test-agent"
        assert "text-generation" in desc.capabilities
        assert desc.provider["name"] == "Hidden Leaf Networks"

    def test_descriptor_has_four_interfaces(self, adapter: W3CProtocolAdapter) -> None:
        desc = adapter.get_descriptor()
        protocols = {i["protocol"] for i in desc.interfaces}
        assert protocols == {"a2a", "mcp", "rest", "websocket"}

    def test_descriptor_has_auth_methods(self, adapter: W3CProtocolAdapter) -> None:
        desc = adapter.get_descriptor()
        assert len(desc.authentication) == 2
        types = {a["type"] for a in desc.authentication}
        assert "bearer" in types
        assert "api_key" in types


# ---------------------------------------------------------------------------
# AgentCard <-> W3CAgentDescriptor conversion
# ---------------------------------------------------------------------------


class TestAgentCardConversion:
    def test_from_aria_agent_card(self, adapter: W3CProtocolAdapter, sample_card: AgentCard) -> None:
        desc = adapter.from_aria_agent_card(sample_card)
        assert desc.name == "Unit Test Agent"
        assert desc.provider["name"] == "Hidden Leaf Networks"
        assert len(desc.capabilities) > 0

    def test_from_card_includes_streaming(self, adapter: W3CProtocolAdapter) -> None:
        card = AgentCard(
            name="Streamer",
            capabilities=AgentCapabilities(streaming=True),
            skills=[],
        )
        desc = adapter.from_aria_agent_card(card)
        assert "streaming" in desc.capabilities

    def test_roundtrip_card_to_descriptor_to_card(
        self, adapter: W3CProtocolAdapter, sample_card: AgentCard
    ) -> None:
        desc = adapter.from_aria_agent_card(sample_card)
        card_dict = adapter.to_aria_agent_card(desc)
        assert card_dict["name"] == sample_card.name
        assert isinstance(card_dict["skills"], list)

    def test_to_aria_agent_card_structure(self, adapter: W3CProtocolAdapter) -> None:
        desc = W3CAgentDescriptor(
            name="W3C Agent",
            capabilities=["text-generation", "planning"],
            interfaces=[{"protocol": "a2a", "endpoint": "https://a2a.test"}],
        )
        card = adapter.to_aria_agent_card(desc)
        assert card["name"] == "W3C Agent"
        assert card["url"] == "https://a2a.test"
        assert len(card["skills"]) == 2
        assert card["capabilities"]["streaming"] is False


# ---------------------------------------------------------------------------
# A2AMessage <-> W3CMessage conversion
# ---------------------------------------------------------------------------


class TestMessageConversion:
    def test_from_aria_message_text(self, adapter: W3CProtocolAdapter, sample_a2a_message: A2AMessage) -> None:
        w3c = adapter.from_aria_message(sample_a2a_message)
        assert w3c.type == W3CMessageType.REQUEST
        assert w3c.content_type == W3CContentType.TEXT
        assert w3c.body == "Hello from A2A"

    def test_from_aria_message_agent_role(self, adapter: W3CProtocolAdapter) -> None:
        msg = A2AMessage.text("Response", role="agent")
        w3c = adapter.from_aria_message(msg)
        assert w3c.type == W3CMessageType.RESPONSE

    def test_from_aria_message_multipart(self, adapter: W3CProtocolAdapter) -> None:
        msg = A2AMessage(
            role="user",
            parts=[A2APart(text="Part 1"), A2APart(text="Part 2")],
        )
        w3c = adapter.from_aria_message(msg)
        assert w3c.content_type == W3CContentType.MULTIPART
        assert isinstance(w3c.body, list)
        assert len(w3c.body) == 2

    def test_from_aria_message_structured_data(self, adapter: W3CProtocolAdapter) -> None:
        msg = A2AMessage(
            role="user",
            parts=[A2APart(structured_data={"key": "value"})],
        )
        w3c = adapter.from_aria_message(msg)
        assert w3c.content_type == W3CContentType.JSON

    def test_to_aria_message_text(self, adapter: W3CProtocolAdapter) -> None:
        w3c = W3CMessage(
            type=W3CMessageType.REQUEST,
            content_type=W3CContentType.TEXT,
            body="Hello W3C",
            correlation_id="task-123",
        )
        a2a = adapter.to_aria_message(w3c)
        assert a2a["role"] == "user"
        assert a2a["parts"][0]["text"] == "Hello W3C"
        assert a2a["task_id"] == "task-123"

    def test_to_aria_message_response(self, adapter: W3CProtocolAdapter) -> None:
        w3c = W3CMessage(type=W3CMessageType.RESPONSE, body="Answer")
        a2a = adapter.to_aria_message(w3c)
        assert a2a["role"] == "agent"


# ---------------------------------------------------------------------------
# MCP tool <-> W3C capability conversion
# ---------------------------------------------------------------------------


class TestMCPConversion:
    def test_from_mcp_tool(self, adapter: W3CProtocolAdapter) -> None:
        tool = {
            "name": "create_plan",
            "description": "Create an execution plan",
            "inputSchema": {"type": "object", "properties": {"name": {"type": "string"}}},
        }
        cap = adapter.from_mcp_tool(tool)
        assert cap["name"] == "create_plan"
        assert cap["type"] == "tool"
        assert cap["protocol_origin"] == "mcp"
        assert "properties" in cap["parameters"]

    def test_to_mcp_tool(self, adapter: W3CProtocolAdapter) -> None:
        cap = {
            "name": "search",
            "description": "Search the web",
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
        }
        tool = adapter.to_mcp_tool(cap)
        assert tool["name"] == "search"
        assert "inputSchema" in tool

    def test_mcp_roundtrip(self, adapter: W3CProtocolAdapter) -> None:
        original = {
            "name": "calculate_risk",
            "description": "Score risk",
            "inputSchema": {"type": "object"},
        }
        cap = adapter.from_mcp_tool(original)
        restored = adapter.to_mcp_tool(cap)
        assert restored["name"] == original["name"]
        assert restored["description"] == original["description"]


# ---------------------------------------------------------------------------
# Protocol negotiation
# ---------------------------------------------------------------------------


class TestProtocolNegotiation:
    def test_w3c_preferred(self, adapter: W3CProtocolAdapter) -> None:
        result = adapter.negotiate_protocol(["w3c", "a2a", "rest"], ["w3c", "mcp"])
        assert result == "w3c"

    def test_a2a_fallback(self, adapter: W3CProtocolAdapter) -> None:
        result = adapter.negotiate_protocol(["a2a", "rest"], ["a2a", "mcp"])
        assert result == "a2a"

    def test_rest_default(self, adapter: W3CProtocolAdapter) -> None:
        result = adapter.negotiate_protocol(["custom-a"], ["custom-b"])
        assert result == "rest"

    def test_mcp_when_shared(self, adapter: W3CProtocolAdapter) -> None:
        result = adapter.negotiate_protocol(["mcp", "rest"], ["mcp", "websocket"])
        assert result == "mcp"


# ---------------------------------------------------------------------------
# Discovery document
# ---------------------------------------------------------------------------


class TestDiscoveryDocument:
    def test_structure(self, adapter: W3CProtocolAdapter) -> None:
        doc = adapter.create_discovery_document()
        assert doc["@context"] == "https://www.w3.org/ns/ai-agent/v1"
        assert doc["@type"] == "AgentDescriptor"
        assert "capabilities" in doc
        assert "interfaces" in doc

    def test_discovery_paths(self, adapter: W3CProtocolAdapter) -> None:
        doc = adapter.create_discovery_document()
        assert doc["discovery"]["well_known_path"] == "/.well-known/ai-agent"
        assert "mcp" in doc["discovery"]["mcp_endpoint"]


# ---------------------------------------------------------------------------
# ProtocolBridge
# ---------------------------------------------------------------------------


class TestProtocolBridge:
    def test_register_and_list(self, bridge: ProtocolBridge) -> None:
        assert "w3c" in bridge.get_supported_protocols()

    def test_translate_a2a_to_w3c(self, bridge: ProtocolBridge, sample_a2a_message: A2AMessage) -> None:
        result = bridge.translate(sample_a2a_message, "a2a", "w3c")
        assert isinstance(result, W3CMessage)
        assert result.body == "Hello from A2A"

    def test_translate_w3c_to_a2a(self, bridge: ProtocolBridge) -> None:
        msg = W3CMessage(type=W3CMessageType.RESPONSE, body="Hi", content_type=W3CContentType.TEXT)
        result = bridge.translate(msg, "w3c", "a2a")
        assert isinstance(result, dict)
        assert result["role"] == "agent"

    def test_translate_mcp_to_w3c(self, bridge: ProtocolBridge) -> None:
        tool = {"name": "test_tool", "description": "A tool", "inputSchema": {}}
        result = bridge.translate(tool, "mcp", "w3c")
        assert result["name"] == "test_tool"
        assert result["protocol_origin"] == "mcp"

    def test_translate_w3c_to_mcp(self, bridge: ProtocolBridge) -> None:
        cap = {"name": "cap1", "description": "Cap", "parameters": {}}
        result = bridge.translate(cap, "w3c", "mcp")
        assert result["name"] == "cap1"
        assert "inputSchema" in result

    def test_unsupported_translation_raises(self, bridge: ProtocolBridge) -> None:
        with pytest.raises(ValueError, match="Unsupported translation path"):
            bridge.translate({}, "graphql", "grpc")

    def test_no_adapters_raises(self) -> None:
        empty = ProtocolBridge()
        with pytest.raises(ValueError, match="No adapters registered"):
            empty.translate({}, "a2a", "w3c")

    def test_multiple_adapters(self, adapter: W3CProtocolAdapter) -> None:
        bridge = ProtocolBridge()
        adapter2 = W3CProtocolAdapter("other-agent", "https://other.example.com")
        bridge.register_adapter("w3c", adapter)
        bridge.register_adapter("a2a", adapter2)
        assert len(bridge.get_supported_protocols()) == 2
        # Translation still works — picks appropriate adapter
        msg = A2AMessage.text("Multi-adapter test")
        result = bridge.translate(msg, "a2a", "w3c")
        assert isinstance(result, W3CMessage)
