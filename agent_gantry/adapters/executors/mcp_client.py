"""
MCP Client adapter for Agent-Gantry.

Connects to MCP servers and converts their tools to ToolDefinition.
"""

from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

logger = logging.getLogger(__name__)

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from agent_gantry.schema.config import MCPServerConfig
from agent_gantry.schema.tool import ToolDefinition, ToolSource

# ``ToolDefinition.name`` must match ``^[a-z][a-z0-9_]*$``, but MCP tool names
# are free-form (``searchWeb``, ``get-weather``, ``Browser.Navigate`` are all
# common in the wild). Discovery used to hand the raw name to the model and
# fail validation — one unconventional tool took the whole server's discovery
# down. The name is normalised here instead; the server still sees the
# original, which ``_convert_tool`` keeps in ``metadata["mcp_tool_name"]``.
_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
_NON_IDENTIFIER = re.compile(r"[^a-z0-9_]+")
_RESERVED_TOOL_NAMES = frozenset({"register", "retrieve", "execute", "list", "delete"})
_NAME_MAX_LENGTH = 128
_DESCRIPTION_MIN_LENGTH = 10
_DESCRIPTION_MAX_LENGTH = 2000
_EXTENDED_DESCRIPTION_MAX_LENGTH = 10000


def sanitize_tool_name(raw: str) -> str:
    """Normalise an external tool name into Gantry's ``snake_case`` identifier form.

    ``searchWeb`` -> ``search_web``, ``get-weather`` -> ``get_weather``,
    ``Browser.Navigate`` -> ``browser_navigate``, ``2fa`` -> ``t_2fa``. Names
    that collide with Gantry's reserved verbs get a ``_tool`` suffix. The
    result always satisfies :class:`~agent_gantry.schema.tool.ToolDefinition`'s
    name pattern; an input with no usable characters becomes ``tool``.
    """
    name = _CAMEL_BOUNDARY.sub("_", raw.strip())
    name = _NON_IDENTIFIER.sub("_", name.lower())
    name = re.sub(r"_+", "_", name).strip("_")
    if not name:
        name = "tool"
    if not name[0].isalpha():
        name = f"t_{name}"
    if name in _RESERVED_TOOL_NAMES:
        name = f"{name}_tool"
    return name[:_NAME_MAX_LENGTH].rstrip("_") or "tool"


def _dedupe_names(names: list[str], raw_names: list[str] | None = None) -> list[str]:
    """Suffix repeated names (``get_user``, ``get_user_2``, ...) so a server whose
    ``getUser`` and ``get_user`` both normalise to one identifier keeps both.

    Which of a colliding pair keeps the bare name is decided by the *raw*
    names, sorted, not by position in the list. MCP does not promise a stable
    ``tools/list`` order, so deciding by position meant a server reordering
    its response silently swapped which upstream operation ``get_user``
    refers to — and a model calling the name it was given last time then
    reached the other tool. ``raw_names`` defaults to ``names`` for callers
    with nothing else to go on.

    The suffix is applied *within* the name-length cap rather than beyond it:
    two names that agree on their first 128 characters would otherwise produce
    an over-long identifier that ``ToolDefinition`` forbids.
    """
    sources = raw_names if raw_names is not None and len(raw_names) == len(names) else names
    # Rank each position among the entries it collides with, by raw name.
    groups: dict[str, list[tuple[str, int]]] = {}
    for index, (name, raw) in enumerate(zip(names, sources)):
        groups.setdefault(name, []).append((raw, index))
    rank: dict[int, int] = {}
    for members in groups.values():
        for order, (_raw, index) in enumerate(sorted(members)):
            rank[index] = order

    taken = set(names)
    out: list[str] = []
    for index, name in enumerate(names):
        if rank[index] == 0:
            out.append(name)
            continue
        count = rank[index] + 1
        candidate = _suffixed(name, count)
        while candidate in taken:
            count += 1
            candidate = _suffixed(name, count)
        taken.add(candidate)
        out.append(candidate)
    return out


def _suffixed(name: str, index: int) -> str:
    """``name`` with ``_<index>`` appended, trimmed to the name-length cap."""
    suffix = f"_{index}"
    return f"{name[: _NAME_MAX_LENGTH - len(suffix)].rstrip('_')}{suffix}"


def _truncate(text: str, limit: int) -> str:
    """Cut ``text`` to ``limit`` characters at a word boundary, marking the cut."""
    if len(text) <= limit:
        return text
    cut = text[: limit - 1]
    space = cut.rfind(" ")
    if space > limit // 2:
        cut = cut[:space]
    return cut.rstrip() + "…"


def _split_description(raw: str | None, tool_name: str, server_name: str) -> tuple[str, str | None]:
    """Fit an MCP description into ``(description, extended_description)``.

    A description under Gantry's 10-character floor (``"Add"``, or ``None`` for
    a tool with none) is padded with the tool's provenance rather than
    rejected; one over the 2000-character ceiling keeps its head as the
    searchable description and the full text (itself capped) as the extended
    description so nothing the server wrote is lost.
    """
    text = " ".join((raw or "").split())
    if len(text) < _DESCRIPTION_MIN_LENGTH:
        # ``tool_name`` is the server's raw name, before ``sanitize_tool_name``
        # caps it, so padding with it unbounded could push the result past
        # ``ToolDefinition``'s 2000-character ceiling. That raised inside the
        # list comprehension in ``list_tools``, taking down discovery for the
        # whole server — the failure this padding exists to avoid, reached by
        # a long name instead of a long description.
        base = text or f"Tool {_truncate(tool_name, _NAME_MAX_LENGTH)}"
        padded = (
            f"{base} (MCP tool '{_truncate(tool_name, _NAME_MAX_LENGTH)}' "
            f"from server '{_truncate(server_name, _NAME_MAX_LENGTH)}')"
        )
        return _truncate(padded, _DESCRIPTION_MAX_LENGTH), None
    if len(text) <= _DESCRIPTION_MAX_LENGTH:
        return text, None
    return (
        _truncate(text, _DESCRIPTION_MAX_LENGTH),
        _truncate(text, _EXTENDED_DESCRIPTION_MAX_LENGTH),
    )


class MCPClient:
    """
    Client for connecting to MCP servers.

    Handles:
    - Connection via stdio (subprocess)
    - MCP handshake (initialize/initialized)
    - Tool discovery (tools/list)
    - Tool execution (tools/call) over a persistent session
    - Conversion of MCP tools to ToolDefinition

    ``call_tool`` keeps one long-lived connection per client: spawning the
    server subprocess and re-running the initialize handshake per call adds
    hundreds of milliseconds (or seconds, for ``npx``-launched servers) to
    every tool execution. The connection is owned by a dedicated background
    task so the anyio cancel scopes of ``stdio_client``/``ClientSession`` are
    entered and exited in the same task — holding them open across arbitrary
    caller tasks is unsafe. Call :meth:`close` to shut the connection down.
    """

    def __init__(self, config: MCPServerConfig) -> None:
        """
        Initialize MCP client.

        Args:
            config: Configuration for the MCP server to connect to
        """
        self.config = config
        self._session: ClientSession | None = None
        self._connected = False
        # Persistent-session machinery (see class docstring)
        self._owner_task: asyncio.Task[None] | None = None
        self._close_event: asyncio.Event | None = None
        self._connect_lock: asyncio.Lock | None = None
        self._loop_id: int | None = None

    @asynccontextmanager
    async def _open_transport(self) -> AsyncIterator[tuple[Any, Any]]:
        """Open the configured transport and yield its ``(read, write)`` streams.

        ``stdio`` spawns the server subprocess; ``streamable_http`` and ``sse``
        connect to ``config.url`` with ``config.headers`` (the place for
        bearer tokens). The HTTP client modules are imported lazily so the
        stdio path never touches httpx.
        """
        transport = self.config.resolved_transport
        if transport == "stdio":
            server_params = StdioServerParameters(
                command=self.config.command[0],
                args=self.config.command[1:] + self.config.args,
                env=self.config.env or None,
            )
            async with stdio_client(server_params) as (read, write):
                yield read, write
            return

        url = self.config.url
        if not url:  # pragma: no cover - MCPServerConfig validation guarantees this
            raise RuntimeError(f"MCP server '{self.config.name}' has no url for {transport}")
        headers = dict(self.config.headers) or None
        if transport == "sse":
            from mcp.client.sse import sse_client

            async with sse_client(
                url, headers=headers, timeout=self.config.timeout_s
            ) as (read, write):
                yield read, write
            return

        # Streamable HTTP. The SDK renamed the helper between minors:
        # ``streamable_http_client`` (1.28+, the only spelling on 2.x) takes a
        # pre-built HTTP client; the older ``streamablehttp_client`` takes
        # headers/timeout directly and is deprecated where both exist.
        try:
            from mcp.client.streamable_http import (
                create_mcp_http_client,
                streamable_http_client,
            )
        except ImportError:  # pragma: no cover - SDKs predating the rename
            from mcp.client.streamable_http import streamablehttp_client

            async with streamablehttp_client(
                url, headers=headers, timeout=self.config.timeout_s
            ) as streams:
                yield streams[0], streams[1]
            return

        # The SDK's own factory, not a bare httpx import: mcp 2.x ships
        # against its own HTTP client module, so this is the only spelling
        # that works on every SDK version. Its defaults (30s connect/write,
        # 300s read for long-lived streams) stay; only the connect budget is
        # narrowed to the configured timeout, via the client's own Timeout
        # type so no HTTP library is imported here.
        http_client = create_mcp_http_client(headers=headers)
        try:
            timeout_cls = type(http_client.timeout)
            http_client.timeout = timeout_cls(self.config.timeout_s, read=300.0)
        except Exception:  # pragma: no cover - keep the SDK defaults on any surprise
            pass
        async with http_client:
            async with streamable_http_client(url, http_client=http_client) as streams:
                yield streams[0], streams[1]

    @asynccontextmanager
    async def connect(self) -> Any:
        """
        Connect to the MCP server.

        Yields:
            ClientSession for interacting with the server
        """
        async with self._open_transport() as (read, write):
            async with ClientSession(read, write) as session:
                # Initialize the session
                await session.initialize()
                self._session = session
                self._connected = True
                try:
                    yield session
                finally:
                    # Only clear our own session: _invalidate_session clears
                    # these fields eagerly, and a replacement session may
                    # already be live by the time this teardown runs.
                    if self._session is session:
                        self._session = None
                        self._connected = False

    async def _ensure_session(self) -> ClientSession:
        """
        Return a live persistent session, connecting if needed.

        The connection is opened and later closed by a dedicated owner task
        (via :meth:`connect`), so the underlying anyio cancel scopes stay in
        one task. Other tasks may safely *use* the returned session.
        """
        loop_id = id(asyncio.get_running_loop())

        if self._connected and self._session is not None and self._loop_id == loop_id:
            return self._session

        if self._loop_id is not None and self._loop_id != loop_id:
            # Session belongs to a different (likely dead) event loop. Its
            # owner task can't be awaited from here — signal it to close and
            # abandon it, then reconnect on the current loop.
            #
            # Loud on purpose: if the other loop is in fact still running,
            # its server subprocess may outlive this reset (two live loops
            # sharing one MCPClient is outside the supported contract).
            logger.warning(
                f"Abandoning MCP session for server '{self.config.name}' owned by a "
                f"different event loop; reconnecting on the current loop"
            )
            #
            # Concurrency scope: this reset runs without awaits, so tasks on
            # a single loop can never interleave inside it. Simultaneous
            # first-use from two OS threads (two live loops sharing one
            # MCPClient) is not supported — same bound as the per-loop lock
            # pattern documented in core/rate_limiter.py.
            if self._close_event is not None:
                try:
                    self._close_event.set()
                except RuntimeError:
                    # Waking the old owner's waiter schedules a callback on
                    # its (likely closed) loop, which raises "Event loop is
                    # closed" — abandon, same guard as _invalidate_session.
                    pass
            self._owner_task = None
            self._close_event = None
            self._connect_lock = None
            self._loop_id = None

        if self._connect_lock is None:
            self._connect_lock = asyncio.Lock()

        async with self._connect_lock:
            if self._connected and self._session is not None and self._loop_id == loop_id:
                return self._session

            ready = asyncio.Event()
            close_event = asyncio.Event()
            startup_error: list[BaseException] = []

            async def owner() -> None:
                session = None
                try:
                    async with self.connect() as session:
                        # connect() sets _session/_connected for the real
                        # implementation; set them explicitly so patched/mock
                        # connect() implementations work too.
                        self._session = session
                        self._connected = True
                        ready.set()
                        await close_event.wait()
                except BaseException as e:
                    startup_error.append(e)
                    ready.set()
                finally:
                    # Only clear our own session (same guard as connect()'s
                    # teardown): a detached owner can exit after invalidation
                    # has already let a replacement session go live, and
                    # clearing the shared fields then would tear down the
                    # replacement's state.
                    if session is not None and self._session is session:
                        self._session = None
                        self._connected = False

            self._close_event = close_event
            self._loop_id = loop_id
            self._owner_task = asyncio.create_task(owner())
            try:
                await ready.wait()
            except asyncio.CancelledError:
                # Caller cancelled mid-startup (e.g. a timeout around
                # discovery). Cancel and drain the owner before releasing the
                # connect lock — otherwise a retry creates a second owner and
                # the half-started first one can later install its session
                # over the replacement and orphan its subprocess where
                # close() can no longer reach it.
                task = self._owner_task
                if task is not None:
                    task.cancel()
                    try:
                        await task
                    except BaseException:
                        pass
                self._owner_task = None
                self._close_event = None
                self._loop_id = None
                raise

            if startup_error:
                self._owner_task = None
                self._close_event = None
                self._loop_id = None
                raise RuntimeError(
                    f"Failed to connect to MCP server '{self.config.name}': {startup_error[0]}"
                ) from startup_error[0]

            if self._session is None:
                # Explicit check rather than assert: asserts are stripped
                # under python -O, and proceeding with a None session would
                # fail far from the cause.
                raise RuntimeError(
                    f"MCP server '{self.config.name}' connected but produced no session"
                )
            return self._session

    async def _invalidate_session(self) -> None:
        """Drop the persistent connection so the next call reconnects."""
        same_loop = self._loop_id is None or self._loop_id == id(asyncio.get_running_loop())
        event = self._close_event
        task = self._owner_task
        self._owner_task = None
        self._close_event = None
        self._loop_id = None
        # Clear the live-session flags NOW, not when the owner task finishes
        # its teardown: concurrent callers whose calls failed against the same
        # broken session invoke _invalidate_session too, return immediately
        # (fields already cleared), and retry — the retry must reconnect, not
        # find still-truthy flags and reuse the dying transport.
        self._session = None
        self._connected = False
        if event is not None:
            if same_loop:
                event.set()
            else:
                # The event belongs to another (likely dead) loop. set() wakes
                # its waiters through that loop and can raise "Event loop is
                # closed" — signal best-effort, then abandon like
                # _ensure_session does.
                try:
                    event.set()
                except RuntimeError:
                    pass
        if task is not None and same_loop:
            try:
                await asyncio.wait_for(asyncio.shield(task), timeout=5)
            except asyncio.TimeoutError:
                # Loud on purpose: a normal close is quiet, so a timeout here
                # means the owner task is stuck and the server subprocess may
                # still be alive.
                logger.warning(
                    f"Timed out waiting for MCP session owner task to close "
                    f"(server '{self.config.name}'); its subprocess may still be running"
                )
            except Exception:
                logger.debug("Error while closing MCP session", exc_info=True)

    async def close(self) -> None:
        """Close the persistent connection (if any). Safe to call repeatedly."""
        await self._invalidate_session()

    async def list_tools(self) -> list[ToolDefinition]:
        """
        List all tools from the MCP server over the persistent session.

        Discovery seeds the persistent connection: add_mcp_server() always
        lists tools before the first call_tool(), so opening a short-lived
        connection here would spawn the subprocess and run the initialize
        handshake twice on the common add-then-call path. Callers that only
        discover must close() the client to release the connection.

        Returns:
            List of ToolDefinition objects
        """
        session = await self._ensure_session()
        try:
            result = await session.list_tools()
        except Exception:
            await self._invalidate_session()
            raise
        raw_names = [str(tool.name) for tool in result.tools]
        tools = [self._convert_tool(tool) for tool in result.tools]
        # Two raw names can normalise to one identifier (``getUser`` and
        # ``get_user``); keep both rather than letting the second overwrite
        # the first in the registry. The suffix is purely local, so the
        # server's own name has to travel with the renamed tool — a tool
        # whose raw name needed no normalising still loses it to the suffix,
        # and dispatching on ``get_user_2`` would call a tool the server does
        # not have.
        unique = _dedupe_names([tool.name for tool in tools], raw_names)
        out: list[ToolDefinition] = []
        for tool, name, raw in zip(tools, unique, raw_names):
            if tool.name == name:
                out.append(tool)
                continue
            metadata = {**tool.metadata, "mcp_tool_name": raw}
            out.append(tool.model_copy(update={"name": name, "metadata": metadata}))
        return out

    def _convert_tool(self, mcp_tool: Any) -> ToolDefinition:
        """
        Convert MCP tool to ToolDefinition.

        Args:
            mcp_tool: MCP tool object

        Returns:
            ToolDefinition object
        """
        # Extract tool information. The raw name is what the server answers
        # to; the normalised one is what Gantry (and the model) sees.
        raw_name = str(mcp_tool.name)
        name = sanitize_tool_name(raw_name)
        description, extended = _split_description(
            getattr(mcp_tool, "description", None), raw_name, self.config.name
        )

        # Convert input schema to parameters_schema. mcp 2.x renamed the
        # attribute inputSchema -> input_schema (the old spelling remains a
        # construction alias only), so read both — checking just the 1.x name
        # would silently replace every v2 tool's schema with the empty default.
        parameters_schema = (
            getattr(mcp_tool, "input_schema", None)
            or getattr(mcp_tool, "inputSchema", None)
            or {"type": "object", "properties": {}, "required": []}
        )

        metadata: dict[str, Any] = {"mcp_server": self.config.name}
        if self.config.command:
            metadata["mcp_command"] = " ".join(self.config.command)
        else:
            metadata["mcp_url"] = self.config.url
            metadata["mcp_transport"] = self.config.resolved_transport
        if name != raw_name:
            # Only recorded when it differs, so tools from well-behaved
            # servers keep byte-identical metadata (and fingerprints).
            metadata["mcp_tool_name"] = raw_name
        # Server-declared annotations (readOnlyHint, destructiveHint, ...)
        # are worth keeping for policy decisions downstream.
        annotations = getattr(mcp_tool, "annotations", None)
        if annotations is not None:
            dump = getattr(annotations, "model_dump", None)
            try:
                as_dict = dump(exclude_none=True) if callable(dump) else dict(annotations)
            except Exception:
                as_dict = None
            if as_dict:
                metadata["mcp_annotations"] = as_dict

        # Create ToolDefinition with MCP source
        return ToolDefinition(
            name=name,
            description=description,
            extended_description=extended,
            parameters_schema=parameters_schema,
            namespace=self.config.namespace,
            source=ToolSource.MCP_SERVER,
            source_uri=f"mcp://{self.config.name}",
            metadata=metadata,
        )

    @staticmethod
    def server_tool_name(tool: ToolDefinition) -> str:
        """The name the MCP server knows ``tool`` by (before normalisation)."""
        return str(tool.metadata.get("mcp_tool_name") or tool.name)

    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        """
        Execute a tool on the MCP server over the persistent session.

        The first call connects (subprocess spawn + initialize handshake);
        subsequent calls reuse the connection. On a transport error the
        session is dropped so the next call reconnects.

        Args:
            tool_name: Name of the tool to execute
            arguments: Arguments for the tool

        Returns:
            Tool execution result
        """
        session = await self._ensure_session()
        try:
            result = await session.call_tool(tool_name, arguments)
        except Exception:
            await self._invalidate_session()
            raise
        # The MCP protocol reports tool failures in-band (isError on the
        # result) rather than as transport errors, so the call above resolves
        # normally. Surface them as exceptions so the execution engine records
        # a failure (retries, health, telemetry) instead of a success. The
        # session stays valid — this is a tool error, not a broken connection.
        # Attribute is spelled isError on mcp 1.x and is_error on 2.x.
        if getattr(result, "isError", False) or getattr(result, "is_error", False):
            raise RuntimeError(self._extract_error_text(result))
        return result

    @staticmethod
    def _extract_error_text(result: Any) -> str:
        """Pull a readable message out of an error CallToolResult."""
        texts = [
            text
            for item in getattr(result, "content", None) or []
            if isinstance(text := getattr(item, "text", None), str)
        ]
        return "; ".join(texts) if texts else f"MCP tool call failed: {result!r}"


class MCPClientPool:
    """
    Pool of MCP clients for managing multiple server connections.
    """

    def __init__(self) -> None:
        """Initialize the client pool."""
        self._clients: dict[str, MCPClient] = {}
        # Clients dropped from the pool that could not be closed at the time
        # (no running loop). ``close_all`` still owns them; see
        # ``remove_server``. ``MCPRegistry`` keeps the same list for the same
        # reason.
        self._retired: list[MCPClient] = []

    def add_server(self, config: MCPServerConfig) -> MCPClient:
        """
        Add an MCP server to the pool.

        Args:
            config: Configuration for the MCP server

        Returns:
            MCPClient instance
        """
        client = MCPClient(config)
        self._clients[config.name] = client
        return client

    def get_client(self, name: str) -> MCPClient | None:
        """
        Get an MCP client by name.

        Args:
            name: Server name

        Returns:
            MCPClient instance or None
        """
        return self._clients.get(name)

    async def list_all_tools(self) -> list[ToolDefinition]:
        """
        List tools from all connected servers.

        Returns:
            List of all ToolDefinition objects from all servers
        """
        all_tools = []
        for client in self._clients.values():
            try:
                tools = await client.list_tools()
                all_tools.extend(tools)
            except Exception as e:
                # Log error but continue with other servers
                logger.error(f"Error listing tools from {client.config.name}: {e}")
        return all_tools

    def remove_server(self, name: str) -> bool:
        """
        Remove an MCP server from the pool.

        Best-effort closes the client's persistent connection (scheduled on
        the running loop when there is one). Called without one — from a
        plain thread, say — the close cannot be scheduled, so the client is
        retained for ``close_all`` instead of being dropped: forgetting it
        there stranded a live HTTP connection or stdio subprocess that
        nothing could reach again, surviving to process teardown.

        Args:
            name: Server name

        Returns:
            True if server was removed
        """
        client = self._clients.pop(name, None)
        if client is None:
            return False
        if not _schedule_client_close(client):
            self._retired.append(client)
        return True

    async def close_all(self) -> None:
        """Close all pooled clients' persistent connections.

        Closes run concurrently so shutdown is bounded by the slowest single
        client (each close can wait up to 5s on a stuck owner task), not the
        sum across servers.
        """
        # ``_retired`` holds clients dropped by ``remove_server`` with no loop
        # to close them on; this is the next async shutdown it was waiting for.
        clients = list(self._clients.values()) + self._retired
        self._retired = []
        results = await asyncio.gather(
            *(client.close() for client in clients), return_exceptions=True
        )
        for result in results:
            if isinstance(result, BaseException):
                logger.debug("Error closing MCP client", exc_info=result)


# Strong references to in-flight fire-and-forget close tasks: the event loop
# only keeps weak references, so without this a scheduled close could be
# garbage-collected before it runs.
_pending_close_tasks: set[asyncio.Task[None]] = set()


def _schedule_client_close(client: MCPClient) -> bool:
    """Schedule ``client.close()`` on the running loop, if any.

    Returns:
        True when a close was scheduled. False means there was no running
        loop to schedule it on, and the caller still owns the client — it
        must keep a reference so the connection or stdio subprocess can be
        closed at the next async shutdown rather than surviving to process
        teardown.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        logger.debug(
            "No running event loop; deferring close of MCP client '%s'",
            client.config.name,
        )
        return False
    task = loop.create_task(client.close())
    # The loop holds only weak references to tasks; without a strong external
    # reference the close task can be garbage-collected mid-flight, silently
    # skipping the close (a done-callback alone does not keep the task alive).
    _pending_close_tasks.add(task)
    task.add_done_callback(_pending_close_tasks.discard)
