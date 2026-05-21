"""MCP (Model Context Protocol) integration for Aria Core.

Provides:
- AriaMCPServer: Exposes aria-core capabilities as MCP tools/resources/prompts
- AriaMCPClient: Consumes external MCP servers as skills

Usage:
    # Run as MCP server
    from aria_core.mcp import create_server
    server = create_server(provider)
    server.run(transport="streamable-http")

    # Use as MCP client
    from aria_core.mcp import AriaMCPClient
    client = AriaMCPClient("http://external-server/mcp")
    tools = await client.list_tools()
"""

from aria_core.mcp.server import create_server
from aria_core.mcp.client import AriaMCPClient

__all__ = ["create_server", "AriaMCPClient"]
