"""
MCP over the network, both directions, plus the hardening around discovery.

Spins up Gantry's own MCPServer over Streamable HTTP and legacy SSE in-process
(uvicorn on a loopback port) and drives it with ``MCPClient`` configured by
``url`` — the remote-server path that used to be impossible (the client was
stdio-only) — then registers the remote server's tools into a second gantry
and executes through it.
"""

from __future__ import annotations

import asyncio
import socket
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

pytest.importorskip("mcp")
uvicorn = pytest.importorskip("uvicorn")

from agent_gantry import AgentGantry
from agent_gantry.adapters.executors.mcp_client import MCPClient, sanitize_tool_name
from agent_gantry.schema.config import MCPServerConfig
from agent_gantry.schema.execution import ExecutionStatus, ToolCall
from agent_gantry.schema.mcp import MCPServerDefinition
from agent_gantry.schema.tool import ToolDefinition
from agent_gantry.servers.mcp_server import MCPServer, _render_tool_output, create_mcp_server


def _free_port() -> int:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def _text(result: Any) -> str:
    for block in result.content:
        text = getattr(block, "text", None)
        if text is not None:
            return text
    raise AssertionError(f"No text content in {result!r}")


async def _served_gantry() -> AgentGantry:
    gantry = AgentGantry()

    @gantry.register(tags=["math"])
    def add_numbers(a: int, b: int) -> int:
        """Add two integers together and return the sum."""
        return a + b

    @gantry.register(namespace="text", tags=["text"])
    def shout(text: str) -> dict[str, Any]:
        """Upper-case a piece of text and report its length."""
        return {"upper": text.upper(), "length": len(text)}

    await gantry.sync()
    return gantry


class _Served:
    """A Gantry MCP server running on a loopback port for one test."""

    def __init__(self, server: MCPServer, transport: str) -> None:
        self.server = server
        self.transport = transport
        self.port = _free_port()
        app = (
            server.streamable_http_app("/mcp")
            if transport == "streamable_http"
            else server.sse_app()
        )
        self._uvicorn = uvicorn.Server(
            uvicorn.Config(app, host="127.0.0.1", port=self.port, log_level="warning")
        )
        self._task: asyncio.Task[None] | None = None

    @property
    def url(self) -> str:
        suffix = "/mcp" if self.transport == "streamable_http" else "/sse"
        return f"http://127.0.0.1:{self.port}{suffix}"

    async def __aenter__(self) -> _Served:
        self._task = asyncio.create_task(self._uvicorn.serve())
        for _ in range(200):
            if self._uvicorn.started:
                return self
            await asyncio.sleep(0.05)
        raise RuntimeError("uvicorn did not start")

    async def __aexit__(self, *exc: Any) -> None:
        self._uvicorn.should_exit = True
        assert self._task is not None
        await asyncio.wait_for(self._task, 15)


@pytest.fixture(params=["streamable_http", "sse"])
async def served(request: pytest.FixtureRequest) -> AsyncIterator[tuple[_Served, AgentGantry]]:
    gantry = await _served_gantry()
    server = create_mcp_server(gantry, mode="hybrid", name="e2e", expose=["add_numbers"])
    async with _Served(server, request.param) as running:
        yield running, gantry
    await gantry.close()


@pytest.mark.asyncio
async def test_remote_roundtrip(served: tuple[_Served, AgentGantry]) -> None:
    """Remote client against Gantry's HTTP server: list, search, execute, errors."""
    running, _gantry = served
    client = MCPClient(MCPServerConfig(name="remote", url=running.url, transport=running.transport))
    try:
        tools = await asyncio.wait_for(client.list_tools(), 30)
        # hybrid: the pinned tool is listed directly next to the meta-tools
        assert sorted(t.name for t in tools) == [
            "add_numbers",
            "execute_tool",
            "find_relevant_tools",
        ]
        remote = next(t for t in tools if t.name == "add_numbers")
        assert remote.metadata["mcp_url"] == running.url
        assert remote.metadata["mcp_transport"] == running.transport

        found = _text(await client.call_tool("find_relevant_tools", {"query": "add two numbers"}))
        assert found.startswith("Tool: add_numbers")
        assert '"properties"' in found  # parameters are JSON, not a Python repr

        # A tool outside the default namespace is reported and addressed qualified
        found = _text(await client.call_tool("find_relevant_tools", {"query": "upper-case text"}))
        assert "Tool: text.shout" in found
        result = _text(
            await client.call_tool(
                "execute_tool", {"tool_name": "text.shout", "arguments": {"text": "hi"}}
            )
        )
        assert result == '{"upper": "HI", "length": 2}'

        assert _text(await client.call_tool("add_numbers", {"a": 2, "b": 5})) == "7"

        with pytest.raises(RuntimeError, match="not found"):
            await client.call_tool("execute_tool", {"tool_name": "nope", "arguments": {}})
        # ...and the session survives the failed call
        assert _text(await client.call_tool("add_numbers", {"a": 1, "b": 1})) == "2"
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_remote_server_tools_execute_through_a_second_gantry(
    served: tuple[_Served, AgentGantry],
) -> None:
    """add_mcp_server with a url discovers the remote tools and executes them."""
    running, _gantry = served
    consumer = AgentGantry()
    try:
        count = await consumer.add_mcp_server(
            MCPServerConfig(
                name="remote", url=running.url, transport=running.transport, namespace="remote"
            )
        )
        assert count == 3
        result = await consumer.execute(
            ToolCall(
                tool_name="execute_tool",
                namespace="remote",
                arguments={"tool_name": "add_numbers", "arguments": {"a": 40, "b": 2}},
            )
        )
        assert result.status == ExecutionStatus.SUCCESS, result.error
        assert _text(result.result) == "42"
    finally:
        await consumer.close()


# ---------------------------------------------------------------------------
# Config / definition validation
# ---------------------------------------------------------------------------


def test_server_config_requires_exactly_one_endpoint() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        MCPServerConfig(name="x")
    with pytest.raises(ValueError, match="exactly one"):
        MCPServerConfig(name="x", command=["a"], url="https://x.example/mcp")
    with pytest.raises(ValueError, match="requires 'url'"):
        MCPServerConfig(name="x", command=["a"], transport="sse")
    with pytest.raises(ValueError, match="requires 'command'"):
        MCPServerConfig(name="x", url="https://x.example/mcp", transport="stdio")
    with pytest.raises(ValueError, match="http"):
        MCPServerConfig(name="x", url="ftp://x.example/mcp")


def test_server_config_infers_transport_and_hides_secrets() -> None:
    local = MCPServerConfig(name="local", command=["npx", "srv"], env={"API_KEY": "s3cret"})
    remote = MCPServerConfig(
        name="remote", url="https://x.example/mcp", headers={"Authorization": "Bearer t0k"}
    )
    assert local.resolved_transport == "stdio"
    assert remote.resolved_transport == "streamable_http"
    assert "s3cret" not in repr(local)
    assert "t0k" not in repr(remote)
    assert remote.endpoint == "https://x.example/mcp"


def test_server_definition_roundtrips_remote_fields() -> None:
    definition = MCPServerDefinition(
        name="search",
        description="Web search and page fetching for research tasks",
        url="https://x.example/sse",
        transport="sse",
        headers={"Authorization": "Bearer t"},
    )
    config = MCPServerConfig(**definition.to_config())
    assert config.resolved_transport == "sse"
    assert config.headers == {"Authorization": "Bearer t"}
    with pytest.raises(ValueError, match="exactly one"):
        MCPServerDefinition(name="bad", description="No endpoint at all here")


@pytest.mark.asyncio
async def test_register_remote_mcp_server_is_retrievable() -> None:
    gantry = AgentGantry()
    gantry.register_mcp_server(
        name="search",
        url="https://x.example/mcp",
        description="Web search, news lookup and page fetching",
        tags=["web", "search"],
    )
    servers = await gantry.retrieve_mcp_servers("search the web for news", limit=1)
    assert [s.name for s in servers] == ["search"]
    assert servers[0].resolved_transport == "streamable_http"


# ---------------------------------------------------------------------------
# Discovery hardening: names and descriptions the wild sends
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("searchWeb", "search_web"),
        ("get-weather", "get_weather"),
        ("Browser.Navigate", "browser_navigate"),
        ("getUserByID", "get_user_by_id"),
        ("2fa", "t_2fa"),
        ("list", "list_tool"),
        ("delete", "delete_tool"),
        ("read_file", "read_file"),
        ("", "tool"),
    ],
)
def test_sanitize_tool_name(raw: str, expected: str) -> None:
    assert sanitize_tool_name(raw) == expected
    ToolDefinition(name=sanitize_tool_name(raw), description="x" * 10, parameters_schema={})


def _mock_tool(name: str, description: str | None, schema: dict[str, Any] | None = None) -> Any:
    tool = MagicMock(spec=["name", "description", "inputSchema"])
    tool.name = name
    tool.description = description
    tool.inputSchema = schema or {"type": "object", "properties": {}}
    return tool


@pytest.mark.asyncio
async def test_discovery_survives_unconventional_tools() -> None:
    """One camelCase tool or a three-character description used to fail the
    whole server's discovery with a ValidationError."""
    client = MCPClient(MCPServerConfig(name="wild", command=["srv"], namespace="wild"))
    session = AsyncMock()
    session.list_tools = AsyncMock(
        return_value=MagicMock(
            tools=[
                _mock_tool("searchWeb", "Search the public web for pages"),
                _mock_tool("add", "Add"),
                _mock_tool("echo", None),
                _mock_tool("get-weather", "w" * 5000),
                _mock_tool("getUser", "Fetch one user record by id"),
                _mock_tool("get_user", "Fetch one user record by id (snake)"),
            ]
        )
    )
    client._ensure_session = AsyncMock(return_value=session)  # type: ignore[method-assign]

    tools = await client.list_tools()
    by_name = {t.name: t for t in tools}
    assert set(by_name) == {"search_web", "add", "echo", "get_weather", "get_user", "get_user_2"}
    assert by_name["search_web"].metadata["mcp_tool_name"] == "searchWeb"
    assert "mcp_tool_name" not in by_name["add"].metadata  # unchanged names carry no alias
    assert by_name["add"].description.startswith("Add (MCP tool 'add'")
    assert by_name["echo"].description.startswith("Tool echo")
    assert len(by_name["get_weather"].description) <= 2000
    assert by_name["get_weather"].extended_description is not None
    # Both keep the name their server answers to: ``getUser`` normalised into
    # ``get_user``, and the raw ``get_user`` was pushed to ``get_user_2`` by
    # the collision, so it needs the alias just as much.
    assert by_name["get_user"].metadata["mcp_tool_name"] == "getUser"
    assert by_name["get_user_2"].metadata["mcp_tool_name"] == "get_user"
    assert all(t.namespace == "wild" for t in tools)


@pytest.mark.asyncio
async def test_a_deduplicated_name_keeps_the_servers_own_name() -> None:
    """``getUser`` normalises to ``get_user``; a raw ``get_user`` beside it is
    renamed ``get_user_2``. The rename is local, so the *server's* name has to
    travel with it — dispatching on ``get_user_2`` would call a tool the
    server does not have."""
    client = MCPClient(MCPServerConfig(name="wild", command=["srv"], namespace="wild"))
    session = AsyncMock()
    session.list_tools = AsyncMock(
        return_value=MagicMock(
            tools=[
                _mock_tool("getUser", "Fetch one user record by id"),
                _mock_tool("get_user", "Fetch one user record by id (snake)"),
            ]
        )
    )
    client._ensure_session = AsyncMock(return_value=session)  # type: ignore[method-assign]

    tools = await client.list_tools()
    assert [tool.name for tool in tools] == ["get_user", "get_user_2"]
    assert [MCPClient.server_tool_name(tool) for tool in tools] == ["getUser", "get_user"]


@pytest.mark.asyncio
async def test_a_dedup_suffix_stays_within_the_name_limit() -> None:
    """``ToolDefinition`` caps a name at 128 characters, and ``model_copy``
    does not revalidate, so the suffix has to fit inside the cap rather than
    extend past it."""
    client = MCPClient(MCPServerConfig(name="wild", command=["srv"], namespace="wild"))
    session = AsyncMock()
    session.list_tools = AsyncMock(
        return_value=MagicMock(
            tools=[
                _mock_tool("a" * 130, "A very long tool name"),
                _mock_tool("a" * 128 + "bb", "Another name sharing its prefix"),
            ]
        )
    )
    client._ensure_session = AsyncMock(return_value=session)  # type: ignore[method-assign]

    tools = await client.list_tools()
    assert len({tool.name for tool in tools}) == 2
    for tool in tools:
        assert len(tool.name) <= 128
        # still a valid definition, which model_copy would not have checked
        ToolDefinition.model_validate(tool.model_dump())


def test_a_definition_rejects_a_non_http_url() -> None:
    """``register_mcp_server`` builds a definition directly, so without this
    an ``ftp://`` endpoint registered and retrieved fine and only failed much
    later, when a client was finally built from it."""
    with pytest.raises(ValueError, match="http"):
        MCPServerDefinition(
            name="bad", description="A server with an unusable url scheme", url="ftp://x.example/y"
        )


@pytest.mark.asyncio
async def test_re_registering_a_server_drops_the_cached_client() -> None:
    """A client cached from the previous definition would keep discovering and
    executing against the old endpoint."""
    gantry = AgentGantry()
    gantry.register_mcp_server(
        name="search", url="https://old.example/mcp", description="The original search endpoint"
    )
    assert gantry._mcp_registry.get_client("search").config.url == "https://old.example/mcp"

    gantry.register_mcp_server(
        name="search", url="https://new.example/mcp", description="The replacement search endpoint"
    )
    assert gantry._mcp_registry.get_client("search").config.url == "https://new.example/mcp"

    # An unchanged re-registration keeps the existing client
    before = gantry._mcp_registry.get_client("search")
    gantry.register_mcp_server(
        name="search", url="https://new.example/mcp", description="The replacement search endpoint"
    )
    assert gantry._mcp_registry.get_client("search") is before
    await gantry.close()


@pytest.mark.asyncio
async def test_a_reconfigured_server_is_honoured_by_existing_handlers() -> None:
    """Dropping the registry's cached client is not enough: a handler created
    by an earlier discovery closed over the old client directly, so an
    already-discovered tool kept executing against the replaced endpoint."""
    gantry = AgentGantry()
    gantry.register_mcp_server(
        name="search", url="https://old.example/mcp", description="The original search endpoint"
    )
    used: list[str] = []

    def arm(client: Any) -> None:
        client.list_tools = AsyncMock(
            return_value=[
                ToolDefinition(
                    name="probe",
                    description="A probe tool used to report its endpoint",
                    parameters_schema={"type": "object", "properties": {}},
                    metadata={"mcp_server": "search"},
                )
            ]
        )

        async def call_tool(name: str, arguments: dict[str, Any], _c: Any = client) -> str:
            used.append(_c.config.url)
            return "ok"

        client.call_tool = call_tool

    arm(gantry._mcp_registry.get_client("search"))
    await gantry.discover_tools_from_server("search")
    await gantry.execute(ToolCall(tool_name="probe", arguments={}))

    gantry.register_mcp_server(
        name="search", url="https://new.example/mcp", description="The replacement search endpoint"
    )
    arm(gantry._mcp_registry.get_client("search"))
    result = await gantry.execute(ToolCall(tool_name="probe", arguments={}))

    assert result.status == ExecutionStatus.SUCCESS, result.error
    assert used == ["https://old.example/mcp", "https://new.example/mcp"]
    await gantry.close()


@pytest.mark.asyncio
async def test_a_pinned_tool_named_like_a_meta_tool_stays_reachable() -> None:
    """``_call_tool`` dispatches the meta names before consulting ``_exposed``,
    so a pinned tool claiming one was advertised and then unreachable."""
    gantry = AgentGantry()

    async def handler(**kwargs: Any) -> str:
        return "pinned!"

    await gantry.add_tool(
        ToolDefinition(
            name="execute_tool",
            description="A user tool whose name clashes with the meta-tool",
            parameters_schema={"type": "object", "properties": {}},
        ),
        handler=handler,
    )
    server = create_mcp_server(gantry, mode="hybrid", expose=["execute_tool"])

    wire_names = [tool.name for tool in await server._list_tools()]
    assert len(wire_names) == len(set(wire_names)), wire_names
    assert "find_relevant_tools" in wire_names and "execute_tool" in wire_names
    assert server._exposed["default_execute_tool"].name == "execute_tool"

    # the pinned tool is callable under its renamed wire name...
    assert await server._call_tool("default_execute_tool", {}) == [
        {"type": "text", "text": "pinned!"}
    ]
    # ...and the meta-tool still owns its own name
    with pytest.raises(ValueError, match="requires a 'tool_name'"):
        await server._call_tool("execute_tool", {})
    await gantry.close()


@pytest.mark.asyncio
async def test_handlers_dispatch_with_the_servers_own_name() -> None:
    """The normalised name is Gantry's; the server must be called by its own."""
    gantry = AgentGantry()
    client = MagicMock()
    client.config = MCPServerConfig(name="wild", command=["srv"], namespace="wild")
    client.call_tool = AsyncMock(return_value="ok")
    tool = ToolDefinition(
        name="search_web",
        namespace="wild",
        description="Search the public web for pages",
        parameters_schema={"type": "object", "properties": {"q": {"type": "string"}}},
        metadata={"mcp_server": "wild", "mcp_tool_name": "searchWeb"},
    )
    gantry._register_mcp_tool_handlers(client, [tool])
    result = await gantry.execute(
        ToolCall(tool_name="search_web", namespace="wild", arguments={"q": "x"})
    )
    assert result.status == ExecutionStatus.SUCCESS, result.error
    client.call_tool.assert_awaited_once_with("searchWeb", {"q": "x"})


# ---------------------------------------------------------------------------
# Server modes and meta-tools (no transport)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_find_relevant_tools_surfaces_tools_with_the_default_embedder() -> None:
    """The hash embedder's scores sit under ToolQuery's 0.5 default threshold;
    the meta-tool must not inherit that silent-drop cutoff."""
    gantry = await _served_gantry()
    server = create_mcp_server(gantry)
    blocks = await server._handle_find_relevant_tools({"query": "add two numbers", "limit": 500})
    assert blocks and blocks[0]["text"].startswith("Tool: add_numbers")
    assert "Parameters: {" in blocks[0]["text"]

    with pytest.raises(ValueError, match="non-empty"):
        await server._handle_find_relevant_tools({"query": "   "})
    with pytest.raises(ValueError, match="must be an object"):
        await server._handle_execute_tool({"tool_name": "add_numbers", "arguments": [1, 2]})
    await gantry.close()


@pytest.mark.asyncio
async def test_hybrid_lists_pinned_tools_and_static_qualifies_collisions() -> None:
    gantry = await _served_gantry()

    @gantry.register(namespace="other")
    def add_numbers(a: int, b: int) -> int:
        """Add two numbers in another namespace."""
        return a + b

    await gantry.sync()

    hybrid = create_mcp_server(gantry, mode="hybrid", expose=["text.shout", "missing_tool"])
    names = [t.name for t in await hybrid._list_tools()]
    assert names == ["find_relevant_tools", "execute_tool", "shout"]
    assert await hybrid._call_tool("shout", {"text": "ab"}) == [
        {"type": "text", "text": '{"upper": "AB", "length": 2}'}
    ]

    static = create_mcp_server(gantry, mode="static")
    names = sorted(t.name for t in await static._list_tools())
    # the two add_numbers collide on their bare name, so both are qualified
    assert names == ["default_add_numbers", "other_add_numbers", "shout"]
    assert await static._call_tool("other_add_numbers", {"a": 1, "b": 2}) == [
        {"type": "text", "text": "3"}
    ]

    with pytest.raises(ValueError, match="Unsupported MCP server mode"):
        MCPServer(gantry, mode="bogus")
    await gantry.close()


@pytest.mark.asyncio
async def test_the_app_serves_when_mounted_into_another_service() -> None:
    """A parent Starlette/FastAPI app does not run a *mounted* sub-app's
    lifespan, so an app that only started the SDK session manager there
    reached an uninitialised manager on the first request — precisely the
    "mount it into your service" flow the method documents."""
    starlette = pytest.importorskip("starlette")
    from starlette.applications import Starlette
    from starlette.responses import PlainTextResponse
    from starlette.routing import Mount, Route

    del starlette
    gantry = await _served_gantry()
    mounted = create_mcp_server(gantry, name="mounted").streamable_http_app("/mcp")
    host = Starlette(
        routes=[
            Route("/health", lambda request: PlainTextResponse("ok")),
            Mount("/tools", app=mounted),
        ]
    )

    port = _free_port()
    server = uvicorn.Server(
        uvicorn.Config(host, host="127.0.0.1", port=port, log_level="warning")
    )
    task = asyncio.create_task(server.serve())
    try:
        for _ in range(200):
            if server.started:
                break
            await asyncio.sleep(0.05)
        assert server.started

        client = MCPClient(
            MCPServerConfig(name="mounted", url=f"http://127.0.0.1:{port}/tools/mcp")
        )
        try:
            tools = await asyncio.wait_for(client.list_tools(), 30)
            assert sorted(t.name for t in tools) == ["execute_tool", "find_relevant_tools"]
            result = await client.call_tool(
                "execute_tool", {"tool_name": "add_numbers", "arguments": {"a": 2, "b": 3}}
            )
            assert _text(result) == "5"
        finally:
            await client.close()
    finally:
        server.should_exit = True
        await asyncio.wait_for(task, 15)
        await gantry.close()


@pytest.mark.asyncio
async def test_a_qualified_wire_name_never_takes_another_tools_name() -> None:
    """``a.x`` and ``b.x`` qualify to ``a_x``/``b_x``, which can collide with a
    real ``default.a_x``. The later one used to overwrite the earlier in the
    dispatch map, so calling one tool ran the other."""
    gantry = AgentGantry()

    async def handler(**kwargs: Any) -> dict[str, Any]:
        return kwargs

    for namespace, name in (("a", "x"), ("b", "x"), ("default", "a_x")):
        await gantry.add_tool(
            ToolDefinition(
                name=name,
                namespace=namespace,
                description=f"Tool {namespace}.{name} for the collision test",
                parameters_schema={"type": "object", "properties": {}},
            ),
            handler=handler,
        )

    server = create_mcp_server(gantry, mode="static")
    wire_names = [tool.name for tool in await server._list_tools()]
    assert len(wire_names) == len(set(wire_names)), wire_names
    # the genuine tool keeps its own name; the qualified one yields
    assert server._exposed["a_x"].namespace == "default"
    assert {name: (t.namespace, t.name) for name, t in server._exposed.items()} == {
        "a_x": ("default", "a_x"),
        "a_x_2": ("a", "x"),
        "b_x": ("b", "x"),
    }
    await gantry.close()


def test_render_tool_output_prefers_json_for_structured_results() -> None:
    assert _render_tool_output({"a": 1, "b": [1, 2]}) == '{"a": 1, "b": [1, 2]}'
    assert _render_tool_output("plain") == "plain"
    assert _render_tool_output(42) == "42"
    block = MagicMock()
    block.text = "from block"
    assert _render_tool_output([block]) == "from block"
    proxied = MagicMock()
    proxied.content = [block]
    assert _render_tool_output(proxied) == "from block"
