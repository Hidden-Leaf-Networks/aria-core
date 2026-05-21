"""Tests for MCP server and client."""

from __future__ import annotations

import json
from uuid import uuid4

import pytest

try:
    from mcp.server.fastmcp import FastMCP
    HAS_MCP = True
except ImportError:
    HAS_MCP = False

from aria_core.mcp.server import create_server
from aria_core.mcp.client import AriaMCPClient, MCPToolDefinition, MCPToolResult
from aria_core.persistence.memory import InMemoryProvider
from aria_core.tenant.models import DEFAULT_TENANT


pytestmark = pytest.mark.skipif(not HAS_MCP, reason="mcp SDK not installed")


@pytest.fixture
async def provider() -> InMemoryProvider:
    p = InMemoryProvider()
    await p.save_tenant(DEFAULT_TENANT)
    return p


class TestMCPServer:
    def test_create_server_returns_fastmcp(self, provider: InMemoryProvider) -> None:
        server = create_server(provider)
        assert server is not None

    async def test_server_has_tools(self, provider: InMemoryProvider) -> None:
        server = create_server(provider)
        # FastMCP registers tools via decorators — check the internal registry
        tools = server._tool_manager._tools
        tool_names = set(tools.keys())
        assert "create_plan" in tool_names
        assert "list_plans" in tool_names
        assert "calculate_risk" in tool_names
        assert "list_archetypes" in tool_names
        assert "get_pricing" in tool_names

    async def test_server_has_resources(self, provider: InMemoryProvider) -> None:
        server = create_server(provider)
        resources = server._resource_manager._resources
        assert len(resources) > 0

    async def test_server_has_prompts(self, provider: InMemoryProvider) -> None:
        server = create_server(provider)
        prompts = server._prompt_manager._prompts
        prompt_names = set(prompts.keys())
        assert "plan_builder" in prompt_names
        assert "risk_assessment" in prompt_names
        assert "agent_designer" in prompt_names


class TestMCPClient:
    def test_client_init(self) -> None:
        client = AriaMCPClient()
        assert len(client.connected_servers) == 0
        assert len(client.list_tools()) == 0

    def test_tool_definition(self) -> None:
        defn = MCPToolDefinition(
            name="test_tool",
            description="A test tool",
            input_schema={"type": "object"},
            server_name="test-server",
        )
        d = defn.to_dict()
        assert d["name"] == "test_tool"
        assert d["server_name"] == "test-server"

    def test_tool_result(self) -> None:
        result = MCPToolResult(content="hello", is_error=False)
        assert result.content == "hello"
        assert not result.is_error

    def test_tool_result_error(self) -> None:
        result = MCPToolResult(content="failed", is_error=True)
        assert result.is_error

    async def test_call_tool_not_found(self) -> None:
        client = AriaMCPClient()
        result = await client.call_tool("nonexistent", {})
        assert result.is_error
        assert "not found" in result.content

    async def test_as_skill_executor(self) -> None:
        """Client can act as a PlanEngine skill executor."""
        client = AriaMCPClient()
        executor = client.as_skill_executor()
        assert callable(executor)

        # Calling unknown tool returns failure
        result = await executor("unknown_tool", {})
        assert not result.success
