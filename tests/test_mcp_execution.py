"""
End-to-end tests for MCP tool execution.

Covers the wiring added so MCP-discovered tools are executable through
``gantry.execute()`` (handler registration in ``add_mcp_server``), and the
persistent MCP session used by ``MCPClient.call_tool``.

Uses a real stdio MCP server (subprocess) built with the ``mcp`` package.
"""

from __future__ import annotations

import asyncio
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


@mcp.tool()
def always_fails() -> int:
    \"\"\"Raise an error to produce an isError tool result.\"\"\"
    raise ValueError("intentional failure")


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
async def test_concurrent_invalidation_clears_session_state(
    server_config: MCPServerConfig,
) -> None:
    """Invalidation clears the live-session flags immediately, so a sibling
    caller's retry reconnects instead of reusing the dying transport, and
    concurrent invalidations are safe."""
    client = MCPClient(server_config)
    try:
        await client.call_tool("server_pid", {})
        assert client._connected is True

        await asyncio.gather(client._invalidate_session(), client._invalidate_session())
        # No live-session state may survive invalidation, even for the
        # caller that returned early because the fields were already cleared
        assert client._connected is False
        assert client._session is None

        # And the client reconnects cleanly afterwards
        result = await client.call_tool("add_numbers", {"a": 1, "b": 1})
        assert _text_content(result) == "2"
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_stale_owner_does_not_clear_replacement_session(
    server_config: MCPServerConfig,
) -> None:
    """A detached owner task exiting late must not tear down the state of a
    replacement session installed after invalidation. Uses a mocked transport
    with slow teardown to force the overlap deterministically."""
    from contextlib import asynccontextmanager

    client = MCPClient(server_config)

    @asynccontextmanager
    async def fake_connect():
        session = object()
        try:
            yield session
        finally:
            # Slow teardown keeps the old owner alive while a replacement
            # session is being installed
            await asyncio.sleep(0.2)

    client.connect = fake_connect

    first = await client._ensure_session()
    invalidation = asyncio.create_task(client._invalidate_session())
    await asyncio.sleep(0)  # let invalidation detach the old owner

    second = await client._ensure_session()
    assert second is not first

    await invalidation  # old owner's teardown finishes AFTER the replacement
    assert client._session is second
    assert client._connected is True

    await client.close()


@pytest.mark.asyncio
async def test_client_raises_on_iserror_result(server_config: MCPServerConfig) -> None:
    """In-band MCP tool failures (isError) surface as exceptions, and the
    session survives them — a tool error is not a broken connection."""
    client = MCPClient(server_config)
    try:
        with pytest.raises(RuntimeError, match="intentional failure"):
            await client.call_tool("always_fails", {})
        assert client._connected is True
        result = await client.call_tool("add_numbers", {"a": 1, "b": 2})
        assert _text_content(result) == "3"
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_add_mcp_server_tools_are_executable(server_config: MCPServerConfig) -> None:
    """Tools discovered via add_mcp_server execute through gantry.execute()."""
    gantry = AgentGantry()
    try:
        count = await gantry.add_mcp_server(server_config)
        assert count == 3

        result = await gantry.execute(
            ToolCall(tool_name="add_numbers", arguments={"a": 20, "b": 22})
        )
        assert result.status == ExecutionStatus.SUCCESS, result.error
        assert _text_content(result.result) == "42"

        # An isError result from the server must be recorded as a failure,
        # not passed through as a successful result.
        failed = await gantry.execute(ToolCall(tool_name="always_fails", arguments={}))
        assert failed.status == ExecutionStatus.FAILURE
        assert "intentional failure" in (failed.error or "")
    finally:
        await gantry.close()


@pytest.mark.asyncio
async def test_mcp_name_collision_keeps_existing_tool(server_config: MCPServerConfig) -> None:
    """An MCP tool sharing a qualified name with an existing tool must not
    hijack its handler: qualified-name dedup keeps the first definition, so
    replacing only the handler would validate against one tool while
    dispatching to another."""
    gantry = AgentGantry()
    try:

        @gantry.register(namespace="mcp_test")
        def add_numbers(a: int, b: int) -> int:
            """Local implementation that shadows the MCP server's tool."""
            return 999

        await gantry.add_mcp_server(server_config)

        result = await gantry.execute(
            ToolCall(tool_name="add_numbers", arguments={"a": 1, "b": 2})
        )
        assert result.status == ExecutionStatus.SUCCESS, result.error
        # The locally registered handler still owns the name
        assert result.result == 999

        # Non-colliding MCP tools from the same server work normally
        pid_result = await gantry.execute(ToolCall(tool_name="server_pid", arguments={}))
        assert pid_result.status == ExecutionStatus.SUCCESS, pid_result.error
    finally:
        await gantry.close()


@pytest.mark.asyncio
async def test_readd_reconfigured_server_refreshes_definitions(tmp_path: Path) -> None:
    """Re-adding a server under the same name with a changed config refreshes
    the stored tool definitions, not just the handlers — otherwise validation
    and authorization would keep using the old schema while calls dispatch to
    the reconfigured server."""
    script_v1 = tmp_path / "server_v1.py"
    script_v1.write_text(SERVER_SCRIPT)
    # v2 changes add_numbers' schema AND renames server_pid → server_pid_v2,
    # so the old name must disappear from the registry on re-add
    script_v2 = tmp_path / "server_v2.py"
    script_v2.write_text(
        SERVER_SCRIPT.replace(
            "Add two numbers together.", "Add two numbers together (v2)."
        ).replace("def server_pid(", "def server_pid_v2(")
    )

    def config_for(script: Path) -> MCPServerConfig:
        return MCPServerConfig(
            name="test-server",
            command=[sys.executable, str(script)],
            namespace="mcp_test",
        )

    gantry = AgentGantry()
    try:
        await gantry.add_mcp_server(config_for(script_v1))
        await gantry.add_mcp_server(config_for(script_v2))

        tools = {f"{t.namespace}.{t.name}": t for t in gantry.export_tools()}
        assert "(v2)" in tools["mcp_test.add_numbers"].description

        # Tools the reconfigured server no longer exposes are removed — their
        # handlers closed over the replaced client and would reconnect to the
        # old command
        assert "mcp_test.server_pid" not in tools
        assert "mcp_test.server_pid_v2" in tools
        stale = await gantry.execute(ToolCall(tool_name="server_pid", arguments={}))
        assert stale.status != ExecutionStatus.SUCCESS

        # And execution dispatches to the reconfigured server
        result = await gantry.execute(
            ToolCall(tool_name="add_numbers", arguments={"a": 2, "b": 2})
        )
        assert result.status == ExecutionStatus.SUCCESS, result.error
        assert _text_content(result.result) == "4"
    finally:
        await gantry.close()


@pytest.mark.asyncio
async def test_mcp_tools_executable_before_sync_with_auto_sync_off(
    server_config: MCPServerConfig,
) -> None:
    """MCP-discovered tools are executable immediately even with
    auto_sync=False — the definition must enter the registry alongside the
    handler, mirroring add_tool()'s documented guarantee."""
    from agent_gantry.schema.config import AgentGantryConfig

    gantry = AgentGantry(AgentGantryConfig(auto_sync=False))
    try:
        count = await gantry.add_mcp_server(server_config)
        assert count == 3

        result = await gantry.execute(
            ToolCall(tool_name="add_numbers", arguments={"a": 3, "b": 4})
        )
        assert result.status == ExecutionStatus.SUCCESS, result.error
        assert _text_content(result.result) == "7"
    finally:
        await gantry.close()


GANTRY_SERVER_SCRIPT = """
import asyncio

from agent_gantry import AgentGantry
from agent_gantry.servers.mcp_server import create_mcp_server

gantry = AgentGantry()


@gantry.register
def echo(text: str) -> str:
    \"\"\"Echo the given text back.\"\"\"
    return text


@gantry.register
def boom() -> str:
    \"\"\"Always fails with a server-side error.\"\"\"
    raise ValueError("server-side failure")


async def main() -> None:
    await gantry.sync()
    server = create_mcp_server(gantry, mode="static", name="gantry-e2e")
    await server.run_stdio()


if __name__ == "__main__":
    asyncio.run(main())
"""


@pytest.mark.asyncio
async def test_gantry_mcp_server_roundtrip(tmp_path: Path) -> None:
    """Full wire round trip against Gantry's own MCPServer: success content
    comes back, and a failed execution arrives as an isError result — pinning
    the assumption that the 1.x SDK wraps raised handler exceptions (and that
    the 2.x callback marks is_error itself), which unit tests can't see."""
    script = tmp_path / "gantry_mcp_server.py"
    script.write_text(GANTRY_SERVER_SCRIPT)
    client = MCPClient(
        MCPServerConfig(
            name="gantry-e2e",
            command=[sys.executable, str(script)],
            namespace="gantry_e2e",
        )
    )
    try:
        tools = await client.list_tools()
        assert {t.name for t in tools} == {"echo", "boom"}

        result = await client.call_tool("echo", {"text": "hello"})
        assert "hello" in _text_content(result)

        # The served failure must surface as an MCP error result, which the
        # client translates into an exception
        with pytest.raises(RuntimeError, match="server-side failure"):
            await client.call_tool("boom", {})

        # And the session survives the failed call
        result = await client.call_tool("echo", {"text": "still alive"})
        assert "still alive" in _text_content(result)
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_list_tools_discovery(server_config: MCPServerConfig) -> None:
    """Discovery returns ToolDefinitions with MCP source metadata."""
    client = MCPClient(server_config)
    try:
        tools = await client.list_tools()
        names = {t.name for t in tools}
        assert names == {"add_numbers", "server_pid", "always_fails"}
        for tool in tools:
            assert tool.namespace == "mcp_test"
            assert tool.metadata["mcp_server"] == "test-server"
    finally:
        await client.close()
