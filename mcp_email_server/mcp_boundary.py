"""Keep strict gateway validation isolated from legacy FastMCP tool behavior."""

from typing import Any

from mcp.server.fastmcp import FastMCP

from mcp_email_server.email_commands import CommandRequest, dispatch_email_command


class EmailMCP(FastMCP):
    async def list_tools(self):
        return [
            tool.model_copy(update={"inputSchema": CommandRequest.model_json_schema()})
            if tool.name == "email_command"
            else tool
            for tool in await super().list_tools()
        ]

    # FastMCP's annotation omits the native CallToolResult it supports at runtime.
    async def call_tool(self, name: str, arguments: dict[str, Any]):  # pyright: ignore[reportIncompatibleMethodOverride]
        if name == "email_command":
            # The registered handler delegates to this same implementation. The
            # SDK pre-parser would otherwise ignore extras and coerce "null" or
            # JSON-looking optional scalar strings before our handler can run.
            return await dispatch_email_command(arguments)
        return await super().call_tool(name, arguments)
