"""MCP Client — consume external MCP servers as skills within aria-core.

Wraps external MCP tool calls as aria-core skill executions,
bridging the MCP ecosystem into the agent runtime.

Usage:
    client = AriaMCPClient()
    await client.connect_stdio("python", "external_server.py")
    tools = await client.list_tools()
    result = await client.call_tool("search", {"query": "hello"})
"""

from __future__ import annotations

import json
from typing import Any


class MCPToolDefinition:
    """Represents an MCP tool discovered from an external server."""

    def __init__(
        self,
        name: str,
        description: str = "",
        input_schema: dict[str, Any] | None = None,
        server_name: str = "",
    ) -> None:
        self.name = name
        self.description = description
        self.input_schema = input_schema or {}
        self.server_name = server_name

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
            "server_name": self.server_name,
        }


class MCPToolResult:
    """Result from calling an MCP tool."""

    def __init__(
        self,
        content: str,
        is_error: bool = False,
        raw: Any = None,
    ) -> None:
        self.content = content
        self.is_error = is_error
        self.raw = raw


class AriaMCPClient:
    """Client for consuming external MCP servers.

    Supports multiple concurrent server connections. Each server's tools
    are namespaced to prevent collisions.

    Connection modes:
    - stdio: Launch a subprocess (python, node, etc.)
    - http: Connect to a Streamable HTTP endpoint
    """

    def __init__(self) -> None:
        self._sessions: dict[str, Any] = {}
        self._tools: dict[str, MCPToolDefinition] = {}

    async def connect_stdio(
        self,
        command: str,
        *args: str,
        server_name: str | None = None,
    ) -> list[MCPToolDefinition]:
        """Connect to an MCP server via stdio transport.

        Args:
            command: Command to run (e.g. "python")
            args: Command arguments (e.g. "server.py")
            server_name: Name for this server connection
        """
        try:
            from mcp.client.session import ClientSession
            from mcp.client.stdio import StdioTransport
        except ImportError:
            raise RuntimeError("MCP SDK not installed. Install with: pip install 'aria-core[mcp]'")

        name = server_name or command
        transport = StdioTransport([command, *args])
        session = ClientSession(transport)
        await session.__aenter__()

        self._sessions[name] = session
        return await self._discover_tools(session, name)

    async def connect_http(
        self,
        url: str,
        server_name: str | None = None,
        headers: dict[str, str] | None = None,
    ) -> list[MCPToolDefinition]:
        """Connect to an MCP server via Streamable HTTP transport.

        Args:
            url: Server URL (e.g. "http://localhost:8080/mcp")
            server_name: Name for this server connection
            headers: Optional auth headers
        """
        try:
            from mcp.client.session import ClientSession
            from mcp.client.streamable_http import StreamableHTTPTransport
        except ImportError:
            raise RuntimeError("MCP SDK not installed. Install with: pip install 'aria-core[mcp]'")

        name = server_name or url
        transport = StreamableHTTPTransport(url, headers=headers or {})
        session = ClientSession(transport)
        await session.__aenter__()

        self._sessions[name] = session
        return await self._discover_tools(session, name)

    async def _discover_tools(
        self, session: Any, server_name: str
    ) -> list[MCPToolDefinition]:
        """Discover and register tools from a connected server."""
        result = await session.list_tools()
        tools = []
        for tool in result.tools:
            defn = MCPToolDefinition(
                name=tool.name,
                description=tool.description or "",
                input_schema=tool.inputSchema if hasattr(tool, "inputSchema") else {},
                server_name=server_name,
            )
            # Namespace: server_name.tool_name
            key = f"{server_name}.{tool.name}"
            self._tools[key] = defn
            tools.append(defn)
        return tools

    def list_tools(self) -> list[MCPToolDefinition]:
        """List all discovered tools across all connected servers."""
        return list(self._tools.values())

    def list_tools_for_server(self, server_name: str) -> list[MCPToolDefinition]:
        """List tools from a specific server."""
        return [t for t in self._tools.values() if t.server_name == server_name]

    async def call_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any] | None = None,
    ) -> MCPToolResult:
        """Call a tool on a connected MCP server.

        Args:
            tool_name: Namespaced tool name (server.tool) or just tool name
            arguments: Tool arguments
        """
        # Find the tool
        defn = self._tools.get(tool_name)
        if not defn:
            # Try without namespace
            for key, d in self._tools.items():
                if d.name == tool_name:
                    defn = d
                    break

        if not defn:
            return MCPToolResult(
                content=f"Tool '{tool_name}' not found",
                is_error=True,
            )

        session = self._sessions.get(defn.server_name)
        if not session:
            return MCPToolResult(
                content=f"Server '{defn.server_name}' not connected",
                is_error=True,
            )

        try:
            result = await session.call_tool(defn.name, arguments or {})
            # Extract text content
            content_parts = []
            for item in result.content:
                if hasattr(item, "text"):
                    content_parts.append(item.text)
                else:
                    content_parts.append(str(item))

            return MCPToolResult(
                content="\n".join(content_parts),
                is_error=result.isError if hasattr(result, "isError") else False,
                raw=result,
            )
        except Exception as e:
            return MCPToolResult(content=str(e), is_error=True)

    async def disconnect(self, server_name: str | None = None) -> None:
        """Disconnect from a server or all servers."""
        if server_name:
            session = self._sessions.pop(server_name, None)
            if session:
                await session.__aexit__(None, None, None)
            # Remove tools
            self._tools = {
                k: v for k, v in self._tools.items()
                if v.server_name != server_name
            }
        else:
            for session in self._sessions.values():
                try:
                    await session.__aexit__(None, None, None)
                except Exception:
                    pass
            self._sessions.clear()
            self._tools.clear()

    @property
    def connected_servers(self) -> list[str]:
        return list(self._sessions.keys())

    def as_skill_executor(self) -> Any:
        """Return a PlanEngine-compatible skill executor that routes to MCP tools."""
        client = self

        async def executor(
            skill_name: str, skill_args: dict[str, Any] | None
        ) -> Any:
            from aria_core.planning.plan_engine import ExecutionResult

            result = await client.call_tool(skill_name, skill_args)
            if result.is_error:
                return ExecutionResult(success=False, error=result.content)
            return ExecutionResult(
                success=True,
                result={"output": result.content},
            )

        return executor
