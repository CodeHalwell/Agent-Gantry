"""
End-to-end tests for MCP tool execution.

Covers the wiring added so MCP-discovered tools are executable through
``gantry.execute()`` (handler registration in ``add_mcp_server``), and the
persistent MCP session used by ``MCPClient.call_tool``.

Uses a real stdio MCP server (subprocess) built with the ``mcp`` package.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("mcp")

from agent_gantry import AgentGantry
from agent_gantry.adapters.executors.mcp_client import MCPClient
from agent_gantry.schema.config import MCPServerConfig
from agent_gantry.schema.execution import ExecutionStatus, ToolCall

SERVER_SCRIPT = """
import os

try:
    # mcp 1.x
    from mcp.server.fastmcp import FastMCP as _Server
except ImportError:
    # mcp 2.x renamed FastMCP to MCPServer (same tool()/run() surface)
    from mcp.server import MCPServer as _Server

mcp = _Server("test-server")


@mcp.tool()
def add_numbers(a: int, b: int) -> int:
    \"\"\"Add two numbers together.\"\"\"
    return a + b


@mcp.tool()
def server_pid() -> int:
    \"\"\"Return this server process's PID.\"\"\"
    return os.getpid()


if __name__ == "__main__":
    mcp.run()
"""


@pytest.fixture()
def server_config(tmp_path: Path) -> MCPServerConfig:
    script = tmp_path / "mcp_test_server.py"
    script.write_text(SERVER_SCRIPT)
    return MCPServerConfig(
        name="test-server",
        command=[sys.executable, str(script)],
        namespace="mcp_test",
    )


def _text_content(result) -> str:
    """Extract text from an MCP CallToolResult."""
    for block in result.content:
        text = getattr(block, "text", None)
        if text is not None:
            return text
    raise AssertionError(f"No text content in result: {result!r}")


@pytest.mark.asyncio
async def test_client_persistent_session_reuse(server_config: MCPServerConfig) -> None:
    """call_tool reuses one server process across calls; close() shuts it down."""
    client = MCPClient(server_config)
    try:
        first = await client.call_tool("server_pid", {})
        assert client._connected is True
        second = await client.call_tool("server_pid", {})
        # Same subprocess handled both calls — the session persisted
        assert _text_content(first) == _text_content(second)

        result = await client.call_tool("add_numbers", {"a": 2, "b": 3})
        assert _text_content(result) == "5"
    finally:
        await client.close()
    assert client._connected is False


@pytest.mark.asyncio
async def test_add_mcp_server_tools_are_executable(server_config: MCPServerConfig) -> None:
    """Tools discovered via add_mcp_server execute through gantry.execute()."""
    gantry = AgentGantry()
    try:
        count = await gantry.add_mcp_server(server_config)
        assert count == 2

        result = await gantry.execute(
            ToolCall(tool_name="add_numbers", arguments={"a": 20, "b": 22})
        )
        assert result.status == ExecutionStatus.SUCCESS, result.error
        assert _text_content(result.result) == "42"
    finally:
        await gantry.close()


@pytest.mark.asyncio
async def test_list_tools_discovery(server_config: MCPServerConfig) -> None:
    """Discovery returns ToolDefinitions with MCP source metadata."""
    client = MCPClient(server_config)
    try:
        tools = await client.list_tools()
        names = {t.name for t in tools}
        assert names == {"add_numbers", "server_pid"}
        for tool in tools:
            assert tool.namespace == "mcp_test"
            assert tool.metadata["mcp_server"] == "test-server"
    finally:
        await client.close()
