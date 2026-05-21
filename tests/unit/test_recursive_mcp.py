"""Tests for Recursive MCP — server-as-agent composition patterns."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import UUID, uuid4

import pytest

from aria_core.mcp.recursive import (
    MCPServerNode,
    MCPTopology,
    RecursiveMCPManager,
    ToolExecutor,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_node(
    id: str = "n1",
    name: str = "server-a",
    url: str = "stdio://",
    transport: str = "stdio",
    tools: list[dict[str, Any]] | None = None,
    **kw: Any,
) -> MCPServerNode:
    return MCPServerNode(
        id=id,
        name=name,
        url=url,
        transport=transport,
        tools=tools or [],
        **kw,
    )


class FakeExecutor(ToolExecutor):
    """In-memory executor for tests — returns args echoed back."""

    def __init__(self, responses: dict[str, dict[str, Any]] | None = None) -> None:
        self._responses = responses or {}
        self.calls: list[tuple[str, str, dict[str, Any]]] = []

    async def execute(
        self, node: MCPServerNode, tool_name: str, args: dict[str, Any]
    ) -> dict[str, Any]:
        self.calls.append((node.id, tool_name, args))
        key = f"{node.name}.{tool_name}"
        if key in self._responses:
            return self._responses[key]
        return {"content": f"executed {tool_name}", "is_error": False, "args": args}


# ---------------------------------------------------------------------------
# MCPServerNode model
# ---------------------------------------------------------------------------

class TestMCPServerNode:
    def test_create_minimal(self) -> None:
        node = MCPServerNode(id="a", name="alpha", url="http://localhost:8080")
        assert node.id == "a"
        assert node.transport == "stdio"
        assert node.connected is False
        assert node.tools == []

    def test_create_with_all_fields(self) -> None:
        node = MCPServerNode(
            id="b",
            name="beta",
            url="http://localhost:9090",
            transport="http",
            command="python",
            args=["-m", "server"],
            headers={"Authorization": "Bearer tok"},
            tools=[{"name": "ping", "description": "ping"}],
            connected=True,
            last_connected_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            metadata={"region": "us-east"},
        )
        assert b.transport == "http" if (b := node) else False
        assert node.headers["Authorization"] == "Bearer tok"
        assert len(node.tools) == 1
        assert node.metadata["region"] == "us-east"

    def test_serialization_roundtrip(self) -> None:
        node = _make_node(tools=[{"name": "t1"}])
        data = node.model_dump()
        restored = MCPServerNode(**data)
        assert restored.id == node.id
        assert restored.tools == node.tools


# ---------------------------------------------------------------------------
# MCPTopology model
# ---------------------------------------------------------------------------

class TestMCPTopology:
    def test_create_default(self) -> None:
        topo = MCPTopology(tenant_id=uuid4())
        assert topo.name == "default"
        assert topo.nodes == []
        assert topo.connections == []
        assert isinstance(topo.id, UUID)

    def test_create_with_nodes(self) -> None:
        n = _make_node()
        tid = uuid4()
        topo = MCPTopology(tenant_id=tid, name="prod", nodes=[n])
        assert len(topo.nodes) == 1
        assert topo.tenant_id == tid


# ---------------------------------------------------------------------------
# RecursiveMCPManager — server management
# ---------------------------------------------------------------------------

class TestServerManagement:
    def test_add_server(self) -> None:
        mgr = RecursiveMCPManager()
        node = _make_node()
        result = mgr.add_server(node)
        assert result.id == "n1"
        assert len(mgr.list_servers()) == 1

    def test_add_duplicate_raises(self) -> None:
        mgr = RecursiveMCPManager()
        mgr.add_server(_make_node(id="x"))
        with pytest.raises(ValueError, match="already registered"):
            mgr.add_server(_make_node(id="x"))

    def test_remove_server(self) -> None:
        mgr = RecursiveMCPManager()
        mgr.add_server(_make_node(id="r1"))
        assert mgr.remove_server("r1") is True
        assert mgr.list_servers() == []

    def test_remove_nonexistent(self) -> None:
        mgr = RecursiveMCPManager()
        assert mgr.remove_server("ghost") is False

    def test_remove_cleans_connections(self) -> None:
        mgr = RecursiveMCPManager()
        mgr.add_server(_make_node(id="a", name="a"))
        mgr.add_server(_make_node(id="b", name="b"))
        mgr.connect("a", "b")
        assert len(mgr.get_topology().connections) == 1
        mgr.remove_server("a")
        assert len(mgr.get_topology().connections) == 0

    def test_list_servers_empty(self) -> None:
        mgr = RecursiveMCPManager()
        assert mgr.list_servers() == []

    def test_tenant_id_string(self) -> None:
        tid = "12345678-1234-1234-1234-123456789abc"
        mgr = RecursiveMCPManager(tenant_id=tid)
        assert mgr._tenant_id == UUID(tid)


# ---------------------------------------------------------------------------
# Tool discovery
# ---------------------------------------------------------------------------

class TestToolDiscovery:
    def test_discover_tools_marks_connected(self) -> None:
        mgr = RecursiveMCPManager()
        node = _make_node(tools=[{"name": "read", "description": "read file"}])
        mgr.add_server(node)
        tools = mgr.discover_tools("n1")
        assert len(tools) == 1
        assert mgr.list_servers()[0].connected is True
        assert mgr.list_servers()[0].last_connected_at is not None

    def test_discover_tools_unknown_server(self) -> None:
        mgr = RecursiveMCPManager()
        with pytest.raises(KeyError, match="not found"):
            mgr.discover_tools("nope")

    def test_discover_all(self) -> None:
        mgr = RecursiveMCPManager()
        mgr.add_server(_make_node(id="a", name="a", tools=[{"name": "t1"}]))
        mgr.add_server(_make_node(id="b", name="b", tools=[{"name": "t2"}, {"name": "t3"}]))
        result = mgr.discover_all()
        assert len(result["a"]) == 1
        assert len(result["b"]) == 2

    def test_get_all_tools_namespaced(self) -> None:
        mgr = RecursiveMCPManager()
        mgr.add_server(_make_node(id="fs", name="filesystem", tools=[
            {"name": "read_file", "description": "read"},
            {"name": "write_file", "description": "write"},
        ]))
        mgr.add_server(_make_node(id="db", name="database", tools=[
            {"name": "query", "description": "sql query"},
        ]))
        all_tools = mgr.get_all_tools()
        names = [t["namespaced_name"] for t in all_tools]
        assert "filesystem.read_file" in names
        assert "filesystem.write_file" in names
        assert "database.query" in names
        assert len(all_tools) == 3

    def test_get_all_tools_includes_composite(self) -> None:
        mgr = RecursiveMCPManager()
        mgr.add_server(_make_node(id="a", name="a", tools=[{"name": "t1"}]))
        mgr.create_composite_tool("my_chain", [{"tool": "a.t1", "args": {}}])
        all_tools = mgr.get_all_tools()
        composite = [t for t in all_tools if t.get("composite")]
        assert len(composite) == 1
        assert composite[0]["name"] == "my_chain"


# ---------------------------------------------------------------------------
# Topology / connections
# ---------------------------------------------------------------------------

class TestTopology:
    def test_connect_servers(self) -> None:
        mgr = RecursiveMCPManager()
        mgr.add_server(_make_node(id="a", name="a"))
        mgr.add_server(_make_node(id="b", name="b"))
        edge = mgr.connect("a", "b", tool_mapping={"read": "ingest"})
        assert edge["from"] == "a"
        assert edge["to"] == "b"
        assert edge["tool_mapping"] == {"read": "ingest"}

    def test_connect_unknown_from(self) -> None:
        mgr = RecursiveMCPManager()
        mgr.add_server(_make_node(id="b", name="b"))
        with pytest.raises(KeyError):
            mgr.connect("ghost", "b")

    def test_connect_unknown_to(self) -> None:
        mgr = RecursiveMCPManager()
        mgr.add_server(_make_node(id="a", name="a"))
        with pytest.raises(KeyError):
            mgr.connect("a", "ghost")

    def test_get_topology_snapshot(self) -> None:
        mgr = RecursiveMCPManager(tenant_id=uuid4())
        mgr.add_server(_make_node(id="x", name="x"))
        mgr.add_server(_make_node(id="y", name="y"))
        mgr.connect("x", "y")
        topo = mgr.get_topology()
        assert isinstance(topo, MCPTopology)
        assert len(topo.nodes) == 2
        assert len(topo.connections) == 1
        assert topo.tenant_id == mgr._tenant_id


# ---------------------------------------------------------------------------
# Tool calling
# ---------------------------------------------------------------------------

class TestToolCalling:
    @pytest.mark.asyncio
    async def test_call_tool_routes_correctly(self) -> None:
        executor = FakeExecutor()
        mgr = RecursiveMCPManager(executor=executor)
        mgr.add_server(_make_node(id="fs", name="filesystem", tools=[{"name": "read"}]))
        result = await mgr.call_tool("filesystem.read", {"path": "/tmp"})
        assert result["content"] == "executed read"
        assert executor.calls[0] == ("fs", "read", {"path": "/tmp"})

    @pytest.mark.asyncio
    async def test_call_tool_no_namespace(self) -> None:
        mgr = RecursiveMCPManager(executor=FakeExecutor())
        result = await mgr.call_tool("bare_tool", {})
        assert "error" in result
        assert "namespaced" in result["error"]

    @pytest.mark.asyncio
    async def test_call_tool_unknown_server(self) -> None:
        mgr = RecursiveMCPManager(executor=FakeExecutor())
        result = await mgr.call_tool("ghost.tool", {})
        assert "error" in result
        assert "ghost" in result["error"]

    @pytest.mark.asyncio
    async def test_call_tool_custom_response(self) -> None:
        executor = FakeExecutor(responses={
            "analytics.parse": {"data": [1, 2, 3], "is_error": False},
        })
        mgr = RecursiveMCPManager(executor=executor)
        mgr.add_server(_make_node(id="an", name="analytics", tools=[{"name": "parse"}]))
        result = await mgr.call_tool("analytics.parse", {"input": "raw"})
        assert result["data"] == [1, 2, 3]


# ---------------------------------------------------------------------------
# Tool chaining
# ---------------------------------------------------------------------------

class TestToolChaining:
    @pytest.mark.asyncio
    async def test_chain_two_steps(self) -> None:
        executor = FakeExecutor(responses={
            "fs.read": {"content": "file data", "is_error": False},
            "analytics.parse": {"parsed": True, "is_error": False},
        })
        mgr = RecursiveMCPManager(executor=executor)
        mgr.add_server(_make_node(id="f", name="fs", tools=[{"name": "read"}]))
        mgr.add_server(_make_node(id="a", name="analytics", tools=[{"name": "parse"}]))

        results = await mgr.chain_tools([
            {"tool": "fs.read", "args": {"path": "/data.json"}},
            {"tool": "analytics.parse", "args": {"data": "$prev"}},
        ])
        assert len(results) == 2
        assert results[0]["content"] == "file data"
        assert results[1]["parsed"] is True
        # Second call should have received previous result as $prev
        _, _, second_args = executor.calls[1]
        assert second_args["data"] == results[0]

    @pytest.mark.asyncio
    async def test_chain_empty(self) -> None:
        mgr = RecursiveMCPManager(executor=FakeExecutor())
        results = await mgr.chain_tools([])
        assert results == []

    @pytest.mark.asyncio
    async def test_chain_with_initial_args(self) -> None:
        executor = FakeExecutor()
        mgr = RecursiveMCPManager(executor=executor)
        mgr.add_server(_make_node(id="s", name="svc", tools=[{"name": "run"}]))
        results = await mgr.chain_tools(
            [{"tool": "svc.run", "args": {"input": "$prev"}}],
            initial_args={"seed": 42},
        )
        assert len(results) == 1
        _, _, call_args = executor.calls[0]
        assert call_args["input"] == {"seed": 42}

    @pytest.mark.asyncio
    async def test_chain_three_steps(self) -> None:
        executor = FakeExecutor()
        mgr = RecursiveMCPManager(executor=executor)
        mgr.add_server(_make_node(id="a", name="a", tools=[{"name": "t1"}]))
        mgr.add_server(_make_node(id="b", name="b", tools=[{"name": "t2"}]))
        mgr.add_server(_make_node(id="c", name="c", tools=[{"name": "t3"}]))

        results = await mgr.chain_tools([
            {"tool": "a.t1", "args": {"x": 1}},
            {"tool": "b.t2", "args": {"prev": "$prev"}},
            {"tool": "c.t3", "args": {"prev": "$prev"}},
        ])
        assert len(results) == 3
        assert len(executor.calls) == 3


# ---------------------------------------------------------------------------
# Composite tools
# ---------------------------------------------------------------------------

class TestCompositeTools:
    def test_create_composite(self) -> None:
        mgr = RecursiveMCPManager()
        steps = [
            {"tool": "a.read", "args": {}},
            {"tool": "b.transform", "args": {"data": "$prev"}},
        ]
        result = mgr.create_composite_tool("etl_pipeline", steps)
        assert result["name"] == "etl_pipeline"
        assert result["composite"] is True
        assert result["steps"] == 2

    @pytest.mark.asyncio
    async def test_call_composite_tool(self) -> None:
        executor = FakeExecutor(responses={
            "src.extract": {"rows": 100, "is_error": False},
            "dest.load": {"loaded": True, "is_error": False},
        })
        mgr = RecursiveMCPManager(executor=executor)
        mgr.add_server(_make_node(id="s", name="src", tools=[{"name": "extract"}]))
        mgr.add_server(_make_node(id="d", name="dest", tools=[{"name": "load"}]))

        mgr.create_composite_tool("etl", [
            {"tool": "src.extract", "args": {"table": "users"}},
            {"tool": "dest.load", "args": {"data": "$prev"}},
        ])

        result = await mgr.call_tool("etl", {})
        assert result["loaded"] is True

    def test_composite_in_get_all_tools(self) -> None:
        mgr = RecursiveMCPManager()
        mgr.create_composite_tool("pipeline", [{"tool": "x.y", "args": {}}])
        tools = mgr.get_all_tools()
        assert any(t["name"] == "pipeline" for t in tools)


# ---------------------------------------------------------------------------
# $prev substitution
# ---------------------------------------------------------------------------

class TestPrevSubstitution:
    def test_direct_replacement(self) -> None:
        result = RecursiveMCPManager._substitute_prev(
            {"data": "$prev"}, {"key": "value"},
        )
        assert result["data"] == {"key": "value"}

    def test_string_interpolation(self) -> None:
        result = RecursiveMCPManager._substitute_prev(
            {"msg": "result is $prev"}, "hello",
        )
        assert result["msg"] == "result is hello"

    def test_no_replacement(self) -> None:
        result = RecursiveMCPManager._substitute_prev(
            {"x": 42, "y": "plain"}, "ignored",
        )
        assert result == {"x": 42, "y": "plain"}

    def test_prev_with_dict_in_string(self) -> None:
        result = RecursiveMCPManager._substitute_prev(
            {"q": "data=$prev"}, {"a": 1},
        )
        assert '"a": 1' in result["q"] or '"a":1' in result["q"]
