"""Recursive MCP — server-as-agent composition patterns.

Enables MCP servers to connect to OTHER MCP servers, creating tool
composition chains. An aria-core agent can discover tools from multiple
MCP servers, namespace them, and chain them together into composite
workflows.

Architecture:
    MCPServerNode  — a registered MCP server endpoint
    MCPTopology    — the graph of server nodes and their connections
    RecursiveMCPManager — orchestrates discovery, routing, and chaining

Usage:
    manager = RecursiveMCPManager(tenant_id=tid)
    node = MCPServerNode(id="fs", name="filesystem", url="stdio://", transport="stdio",
                         command="python", args=["fs_server.py"])
    manager.add_server(node)
    await manager.discover_tools("fs")
    result = await manager.call_tool("filesystem.read_file", {"path": "/tmp/x"})

    # Chain tools across servers
    results = await manager.chain_tools([
        {"tool": "filesystem.read_file", "args": {"path": "/tmp/data.json"}},
        {"tool": "analytics.parse_json", "args": {"data": "$prev"}},
    ])
"""

from __future__ import annotations

import copy
import json
from datetime import datetime, timezone
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class MCPServerNode(BaseModel):
    """A registered MCP server endpoint."""

    id: str
    name: str
    url: str
    transport: str = "stdio"  # "stdio" | "http"
    command: str | None = None
    args: list[str] = Field(default_factory=list)
    headers: dict[str, str] = Field(default_factory=dict)
    tools: list[dict[str, Any]] = Field(default_factory=list)
    connected: bool = False
    last_connected_at: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class MCPTopology(BaseModel):
    """Graph of MCP server nodes and their composition edges."""

    id: UUID = Field(default_factory=uuid4)
    tenant_id: UUID
    name: str = "default"
    nodes: list[MCPServerNode] = Field(default_factory=list)
    connections: list[dict[str, Any]] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ---------------------------------------------------------------------------
# Tool executor protocol — allows injection for testing
# ---------------------------------------------------------------------------

class ToolExecutor:
    """Executes a tool call against a real MCP server. Override for testing."""

    async def execute(
        self, node: MCPServerNode, tool_name: str, args: dict[str, Any]
    ) -> dict[str, Any]:
        """Execute *tool_name* on *node* with *args*.

        The default implementation delegates to :class:`AriaMCPClient`.
        """
        from aria_core.mcp.client import AriaMCPClient

        client = AriaMCPClient()
        if node.transport == "stdio" and node.command:
            await client.connect_stdio(
                node.command, *node.args, server_name=node.name,
            )
        elif node.transport == "http":
            await client.connect_http(
                node.url, server_name=node.name, headers=node.headers or None,
            )
        else:
            return {"error": f"Unsupported transport: {node.transport}"}

        result = await client.call_tool(f"{node.name}.{tool_name}", args)
        await client.disconnect()
        return {"content": result.content, "is_error": result.is_error}


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------

class RecursiveMCPManager:
    """Orchestrates recursive MCP server composition.

    Manages a topology of MCP servers, discovers their tools, and enables
    chaining / composition across server boundaries.
    """

    def __init__(
        self,
        tenant_id: UUID | str | None = None,
        executor: ToolExecutor | None = None,
    ) -> None:
        if isinstance(tenant_id, str):
            tenant_id = UUID(tenant_id)
        self._tenant_id: UUID = tenant_id or UUID("00000000-0000-0000-0000-000000000000")
        self._nodes: dict[str, MCPServerNode] = {}
        self._connections: list[dict[str, Any]] = []
        self._composite_tools: dict[str, list[dict[str, Any]]] = {}
        self._executor: ToolExecutor = executor or ToolExecutor()

    # -- Server management ---------------------------------------------------

    def add_server(self, node: MCPServerNode) -> MCPServerNode:
        """Register an MCP server node."""
        if node.id in self._nodes:
            raise ValueError(f"Server '{node.id}' already registered")
        self._nodes[node.id] = node
        return node

    def remove_server(self, node_id: str) -> bool:
        """Remove a server by id. Returns True if removed."""
        if node_id not in self._nodes:
            return False
        del self._nodes[node_id]
        # Remove related connections
        self._connections = [
            c for c in self._connections
            if c["from"] != node_id and c["to"] != node_id
        ]
        return True

    def list_servers(self) -> list[MCPServerNode]:
        """Return all registered server nodes."""
        return list(self._nodes.values())

    # -- Tool discovery ------------------------------------------------------

    def discover_tools(self, node_id: str) -> list[dict[str, Any]]:
        """Discover tools from a specific server.

        In production this would call the MCP server's ``list_tools``.
        Here we return what is already registered on the node (populated
        during ``add_server`` or externally).
        """
        node = self._nodes.get(node_id)
        if not node:
            raise KeyError(f"Server '{node_id}' not found")
        node.connected = True
        node.last_connected_at = datetime.now(timezone.utc)
        return node.tools

    def discover_all(self) -> dict[str, list[dict[str, Any]]]:
        """Discover tools from every registered server."""
        return {nid: self.discover_tools(nid) for nid in self._nodes}

    def get_all_tools(self) -> list[dict[str, Any]]:
        """Flat list of all tools, namespaced as ``server_name.tool_name``."""
        tools: list[dict[str, Any]] = []
        for node in self._nodes.values():
            for tool in node.tools:
                namespaced = dict(tool)
                namespaced["namespaced_name"] = f"{node.name}.{tool['name']}"
                namespaced["server_id"] = node.id
                namespaced["server_name"] = node.name
                tools.append(namespaced)
        # Include composite tools
        for name, steps in self._composite_tools.items():
            tools.append({
                "name": name,
                "namespaced_name": name,
                "description": f"Composite tool: {len(steps)} steps",
                "composite": True,
                "steps": steps,
            })
        return tools

    # -- Topology / connections ----------------------------------------------

    def connect(
        self,
        from_id: str,
        to_id: str,
        tool_mapping: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Create a composition edge between two server nodes."""
        if from_id not in self._nodes:
            raise KeyError(f"Server '{from_id}' not found")
        if to_id not in self._nodes:
            raise KeyError(f"Server '{to_id}' not found")
        edge: dict[str, Any] = {
            "from": from_id,
            "to": to_id,
            "tool_mapping": tool_mapping or {},
        }
        self._connections.append(edge)
        return edge

    def get_topology(self) -> MCPTopology:
        """Return the full topology snapshot."""
        return MCPTopology(
            tenant_id=self._tenant_id,
            nodes=list(self._nodes.values()),
            connections=list(self._connections),
        )

    # -- Tool execution ------------------------------------------------------

    async def call_tool(self, tool_name: str, args: dict[str, Any] | None = None) -> dict[str, Any]:
        """Route a namespaced tool call to the correct server and execute.

        ``tool_name`` is expected as ``server_name.tool`` (namespaced).
        Falls back to checking composite tools if no server match.
        """
        args = args or {}

        # Check composite tools first
        if tool_name in self._composite_tools:
            steps = self._composite_tools[tool_name]
            results = await self.chain_tools(steps, initial_args=args)
            return results[-1] if results else {"error": "Empty composite tool"}

        # Parse namespace
        if "." not in tool_name:
            return {"error": f"Tool '{tool_name}' must be namespaced as 'server_name.tool_name'"}

        server_name, local_tool = tool_name.split(".", 1)

        # Find node by name
        node = self._resolve_node_by_name(server_name)
        if not node:
            return {"error": f"No server named '{server_name}'"}

        return await self._executor.execute(node, local_tool, args)

    async def chain_tools(
        self,
        steps: list[dict[str, Any]],
        initial_args: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Execute a chain of tool calls sequentially.

        Each step is ``{"tool": "server.tool", "args": {...}}``.
        Inside ``args``, the string ``"$prev"`` is replaced with the
        previous step's result.
        """
        results: list[dict[str, Any]] = []
        prev_result: Any = initial_args or {}

        for step in steps:
            tool = step["tool"]
            step_args = self._substitute_prev(step.get("args", {}), prev_result)
            result = await self.call_tool(tool, step_args)
            results.append(result)
            prev_result = result

        return results

    def create_composite_tool(
        self, name: str, steps: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Register a composite tool — a named chain of other tools."""
        self._composite_tools[name] = steps
        return {
            "name": name,
            "composite": True,
            "steps": len(steps),
            "definition": steps,
        }

    # -- Helpers -------------------------------------------------------------

    def _resolve_node_by_name(self, name: str) -> MCPServerNode | None:
        for node in self._nodes.values():
            if node.name == name:
                return node
        return None

    @staticmethod
    def _substitute_prev(args: dict[str, Any], prev: Any) -> dict[str, Any]:
        """Replace ``"$prev"`` placeholders in *args* with *prev*."""
        resolved: dict[str, Any] = {}
        for key, value in args.items():
            if value == "$prev":
                resolved[key] = prev
            elif isinstance(value, str) and "$prev" in value:
                # String interpolation: "prefix $prev suffix"
                resolved[key] = value.replace("$prev", json.dumps(prev) if not isinstance(prev, str) else prev)
            else:
                resolved[key] = value
        return resolved
