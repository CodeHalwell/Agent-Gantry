"""
MCP Server implementation for Agent-Gantry.

Exposes AgentGantry as an MCP server with dynamic tool discovery.

Supports both mcp 1.x and mcp 2.x. The wire types (``Tool``, ``TextContent``)
and the stdio transport (``stdio_server`` + ``Server.run``) are identical
across both major versions; the one breaking seam is handler registration —
mcp 2.0 removed the ``@server.list_tools()`` / ``@server.call_tool()``
decorators in favour of constructor callbacks (``on_list_tools`` /
``on_call_tool``) whose handlers take ``(ctx, params)`` and return full
result models (``ListToolsResult`` / ``CallToolResult``), with no automatic
wrapping of returned content lists or raised exceptions.

Three modes:

- ``dynamic`` (default) exposes two meta-tools, ``find_relevant_tools`` and
  ``execute_tool``: the MCP client semantic-searches Gantry on demand, so its
  tool list stays two entries long however many tools are registered.
- ``static`` lists every registered tool directly.
- ``hybrid`` lists the tools named in ``expose`` directly and keeps the
  meta-tools for everything else — for the handful of tools a client should
  always see without paying for the whole registry.

Transports: ``run_stdio`` for local clients (Claude Desktop, Claude Code,
Cline), ``run_http`` for the spec's Streamable HTTP transport, and ``run_sse``
for clients that still speak the legacy SSE transport. ``streamable_http_app``
/ ``sse_app`` return the Starlette apps for mounting into an existing
service.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import re
from collections.abc import AsyncIterator, Iterable, Sequence
from typing import TYPE_CHECKING, Any

import mcp
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool

from agent_gantry.schema.query import ConversationContext, ToolQuery
from agent_gantry.utils.render import render_result

if TYPE_CHECKING:
    from agent_gantry import AgentGantry
    from agent_gantry.schema.tool import ToolDefinition

logger = logging.getLogger(__name__)

#: Bounds for the ``limit`` argument of ``find_relevant_tools`` — mirrors
#: ``ToolQuery.limit`` (``ge=1, le=50``) so an over-eager client gets a
#: clamped search rather than a validation error.
_FIND_LIMIT_MIN = 1
_FIND_LIMIT_MAX = 50

#: Characters allowed in a wire-level MCP tool name. The spec recommends
#: ``[A-Za-z0-9_.-]``; the Claude API is stricter (no dots), so dots are
#: swapped for underscores when a namespace has to be folded into a name.
_WIRE_NAME_INVALID = re.compile(r"[^A-Za-z0-9_-]+")


# mcp 2.x introduced the high-level `Client`; 1.x has no such attribute.
def _detect_mcp_v2() -> bool:
    """Detect mcp 2.x by installed version, falling back to a symbol probe.

    The major version is the contract we branch on; hasattr(mcp, "Client")
    alone is a proxy that could silently flip if the symbol is ever added to
    or removed from a different major.
    """
    try:
        from importlib.metadata import version

        return int(version("mcp").split(".")[0]) >= 2
    except Exception:
        return hasattr(mcp, "Client")


_MCP_V2 = _detect_mcp_v2()


def _text_block(text: str) -> dict[str, Any]:
    """A text content block in the shape both SDK majors accept."""
    return {"type": "text", "text": text}


def _render_tool_output(value: Any) -> str:
    """Stringify a tool result for an MCP client.

    Structured results (dicts, lists of plain values) are emitted as JSON —
    a model can parse ``{"a": 1}`` and cannot reliably parse the Python repr
    ``{'a': 1}``. Content-block lists and objects carrying ``.content`` (a
    proxied ``CallToolResult`` from an upstream MCP server) render through
    :func:`render_result`, which reads their ``.text``.
    """
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list, tuple)):
        blocks = value if isinstance(value, (list, tuple)) else None
        if blocks is not None and blocks and all(
            hasattr(item, "text") or (isinstance(item, dict) and "text" in item)
            for item in blocks
        ):
            return render_result(blocks)
        try:
            return json.dumps(value, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            return render_result(value)
    return render_result(value)


# Matches the cap the client applies to names coming the other way
# (``mcp_client._NAME_MAX_LENGTH``), and the cap on ``ToolDefinition.name``.
_WIRE_NAME_MAX_LENGTH = 128


def _fit_wire_name(candidate: str, suffix: str = "") -> str:
    """Trim ``candidate`` so ``candidate + suffix`` fits the wire-name cap.

    A definition's own name may be up to 128 characters, which is the cap, so
    qualifying it as ``namespace_name`` overruns it — and a collision suffix
    pushes it further. Every registered tool was valid, yet the listing a
    client received could be rejected whole by name validation.
    """
    room = _WIRE_NAME_MAX_LENGTH - len(suffix)
    return f"{candidate[:room].rstrip('_')}{suffix}" if suffix else candidate[:room]


def _unique_wire_name(candidate: str, taken: set[str]) -> str:
    """``candidate``, suffixed ``_2``, ``_3``… until it is not already ``taken``.

    Both the candidate and each suffixed form are kept inside the wire-name
    cap. Trimming can itself collide — two long names agreeing on their first
    128 characters — which the suffix loop then resolves.
    """
    candidate = _fit_wire_name(candidate)
    if candidate not in taken:
        return candidate
    index = 2
    while True:
        name = _fit_wire_name(candidate, f"_{index}")
        if name not in taken:
            return name
        index += 1


class _ASGIProxy:
    """Wrap a bound ASGI handler so Starlette's ``Route`` treats it as an app.

    ``Route`` wraps plain functions/methods in ``request_response`` (a
    ``Request -> Response`` contract); an object with ``__call__`` is passed
    the raw ``(scope, receive, send)`` triple, which is what the SDK's
    transport handlers expect.
    """

    def __init__(self, handler: Any) -> None:
        self._handler = handler

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        await self._handler(scope, receive, send)


class _StreamableHTTPApp:
    """ASGI app that owns the SDK session manager's lifecycle.

    ``StreamableHTTPSessionManager.handle_request`` needs the task group that
    only exists inside ``manager.run()``. Starlette does **not** run a
    *mounted* sub-application's lifespan (checked against starlette 1.6), so
    an app that started the manager from its own lifespan alone worked when
    served directly and failed the moment it was mounted into an existing
    service — the documented way to use it — reaching an uninitialised
    manager on the first request.

    So the manager is started here instead: eagerly from the lifespan when
    this app is served directly, and lazily on the first request when it is
    mounted. ``run()`` may be entered only once per manager, so both paths
    funnel through one guarded start.
    """

    def __init__(self, manager: Any) -> None:
        self._manager = manager
        # Created lazily: constructing a lock binds it to whichever loop is
        # running at construction, which need not be the serving one.
        self._lock: asyncio.Lock | None = None
        self._task: asyncio.Task[None] | None = None
        self._close: asyncio.Event | None = None
        self._closed = False

    async def start(self) -> None:
        """Enter ``manager.run()`` once, and wait until it is serving."""
        if self._closed:
            # The SDK's manager may be entered only once, so a stopped app
            # cannot be revived — say so rather than surfacing the SDK's
            # "run() can only be called once per instance" from a request.
            raise RuntimeError(
                "This MCP app has been stopped and cannot serve again; "
                "build a new one with streamable_http_app()."
            )
        if self._task is not None:
            return
        if self._lock is None:
            self._lock = asyncio.Lock()
        async with self._lock:
            if self._task is not None:
                return
            ready = asyncio.Event()
            close = asyncio.Event()
            failure: list[BaseException] = []

            async def runner() -> None:
                try:
                    async with self._manager.run():
                        ready.set()
                        await close.wait()
                except BaseException as exc:  # noqa: BLE001 - re-raised below
                    failure.append(exc)
                    ready.set()

            task = asyncio.create_task(runner())
            await ready.wait()
            if failure:
                raise RuntimeError(
                    f"MCP Streamable HTTP session manager failed to start: {failure[0]}"
                ) from failure[0]
            self._task, self._close = task, close

    async def stop(self) -> None:
        """Leave ``manager.run()``, if this app started it. Not restartable."""
        self._closed = True
        task, close = self._task, self._close
        self._task = self._close = None
        if close is not None:
            close.set()
        if task is not None:
            try:
                await asyncio.wait_for(task, timeout=5)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                task.cancel()
            except Exception:
                logger.debug("Error stopping the MCP session manager", exc_info=True)

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        await self.start()
        await self._manager.handle_request(scope, receive, send)


def _security_settings(
    allowed_hosts: Sequence[str] | None, allowed_origins: Sequence[str] | None
) -> Any:
    """Build the SDK's DNS-rebinding protection settings, or ``None`` for off.

    The SDK's own default (no settings) disables the check for backwards
    compatibility; it is switched on here as soon as the caller names a host
    or origin list.
    """
    if not allowed_hosts and not allowed_origins:
        return None
    from mcp.server.transport_security import TransportSecuritySettings

    return TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=list(allowed_hosts or []),
        allowed_origins=list(allowed_origins or []),
    )


class MCPServer:
    """
    MCP Server for Agent-Gantry.

    Supports three modes:
    - dynamic: Only expose meta-tools (find_relevant_tools, execute_tool)
    - static: Expose all tools directly
    - hybrid: Expose the tools named in ``expose`` directly + meta-tools for the rest
    """

    def __init__(
        self,
        gantry: AgentGantry,
        mode: str = "dynamic",
        name: str = "agent-gantry",
        *,
        expose: Iterable[str] | None = None,
        find_limit: int = 5,
    ) -> None:
        """
        Initialize MCP server.

        Args:
            gantry: AgentGantry instance to serve
            mode: Server mode (dynamic, static, or hybrid)
            name: Server name for identification
            expose: Tool names (``name`` or ``namespace.name``) listed
                directly in ``hybrid`` mode. Ignored in the other modes.
            find_limit: Default number of tools ``find_relevant_tools``
                returns when the client does not pass ``limit``.
        """
        if mode not in ("dynamic", "static", "hybrid"):
            raise ValueError(f"Unsupported MCP server mode: {mode!r} (dynamic, static or hybrid)")
        self.gantry = gantry
        self.mode = mode
        self.name = name
        self._expose = [str(item) for item in (expose or [])]
        self._find_limit = max(_FIND_LIMIT_MIN, min(int(find_limit), _FIND_LIMIT_MAX))
        # Wire name -> definition for tools listed directly, rebuilt on every
        # list_tools so a tool added after startup is served too.
        self._exposed: dict[str, ToolDefinition] = {}
        self._warned_missing_expose = False
        if _MCP_V2:
            self.server = Server(
                name,
                on_list_tools=self._on_list_tools_v2,
                on_call_tool=self._on_call_tool_v2,
            )
        else:
            self.server = Server(name)
            self._setup_v1_handlers()

    # ------------------------------------------------------------------
    # Tool listing / dispatch (shared across mcp versions)
    # ------------------------------------------------------------------

    async def _list_tools(self) -> list[Tool]:
        """List available tools based on mode (shared across mcp versions)."""
        if self.mode == "static":
            # Static mode: expose all tools directly
            return self._direct_tools(await self.gantry.list_tools())
        tools = self._get_meta_tools()
        if self.mode == "hybrid":
            # The meta-tools are already on the wire and ``_call_tool``
            # dispatches their names before consulting ``_exposed``, so a
            # pinned tool of the same name would be advertised and then be
            # unreachable. Reserving them renames it instead.
            tools.extend(
                self._direct_tools(
                    await self._pinned_tools(), reserved={tool.name for tool in tools}
                )
            )
        return tools

    async def _pinned_tools(self) -> list[ToolDefinition]:
        """Resolve the ``expose`` names against the registry (hybrid mode)."""
        if not self._expose:
            return []
        all_tools = await self.gantry.list_tools()
        by_bare: dict[str, list[ToolDefinition]] = {}
        by_qualified: dict[str, ToolDefinition] = {}
        for tool in all_tools:
            by_bare.setdefault(tool.name, []).append(tool)
            by_qualified[f"{tool.namespace}.{tool.name}"] = tool
        pinned: list[ToolDefinition] = []
        missing: list[str] = []
        seen: set[str] = set()
        for wanted in self._expose:
            matches = [by_qualified[wanted]] if wanted in by_qualified else by_bare.get(wanted, [])
            if not matches:
                missing.append(wanted)
            for tool in matches:
                key = f"{tool.namespace}.{tool.name}"
                if key not in seen:
                    seen.add(key)
                    pinned.append(tool)
        if missing and not self._warned_missing_expose:
            self._warned_missing_expose = True
            logger.warning(
                "MCP server %r: expose names not registered with the gantry: %s",
                self.name,
                ", ".join(missing),
            )
        return pinned

    def _direct_tools(
        self, definitions: Sequence[ToolDefinition], *, reserved: set[str] | None = None
    ) -> list[Tool]:
        """Convert definitions to wire tools, folding namespaces into names on collision.

        ``reserved`` names are already spoken for on the wire (the meta-tools
        in hybrid mode), so a definition claiming one is renamed rather than
        shadowed.
        """
        counts: dict[str, int] = {}
        for tool in definitions:
            counts[tool.name] = counts.get(tool.name, 0) + 1
        # Every *bare* name is reserved before any qualifying happens, and each
        # generated name joins the set as it is minted. A qualified name can
        # therefore never land on another tool's real name — ``a.x`` and
        # ``b.x`` qualify to ``a_x``/``b_x`` while a genuine ``default.a_x``
        # keeps ``a_x``, and the loser takes a numeric suffix. Without this,
        # the second assignment simply overwrote the first in ``_exposed`` and
        # a client calling one tool reached the other.
        taken = {tool.name for tool in definitions} | (reserved or set())
        self._exposed = {}
        wire_tools: list[Tool] = []
        for tool in definitions:
            wire_name = tool.name
            if reserved and wire_name in reserved:
                # Claims a meta-tool's name: rename unconditionally, since the
                # meta handler would otherwise win every dispatch.
                wire_name = _unique_wire_name(
                    _WIRE_NAME_INVALID.sub("_", f"{tool.namespace}_{tool.name}"), taken
                )
                taken.add(wire_name)
            elif counts[tool.name] > 1:
                # Same bare name in several namespaces: qualify all of them
                # so the mapping is deterministic whatever the listing order.
                wire_name = _unique_wire_name(
                    _WIRE_NAME_INVALID.sub("_", f"{tool.namespace}_{tool.name}"), taken
                )
                taken.add(wire_name)
            self._exposed[wire_name] = tool
            wire_tools.append(self._convert_tool(tool, wire_name=wire_name))
        return wire_tools

    async def _call_tool(self, name: str, arguments: dict[str, Any]) -> list[Any]:
        """Dispatch a tool call (shared across mcp versions)."""
        if name == "find_relevant_tools":
            return await self._handle_find_relevant_tools(arguments)
        if name == "execute_tool":
            return await self._handle_execute_tool(arguments)
        exposed = self._exposed.get(name)
        if exposed is not None:
            # Direct execution of a listed tool; pin the namespace so a
            # same-named tool elsewhere is never run instead.
            return await self._execute(exposed.name, arguments, namespace=exposed.namespace)
        # Direct tool execution by (possibly qualified) name
        return await self._handle_execute_tool({"tool_name": name, "arguments": arguments})

    def _setup_v1_handlers(self) -> None:
        """Register handlers via the mcp 1.x decorators."""

        @self.server.list_tools()  # type: ignore[misc, no-untyped-call]
        async def list_tools() -> list[Tool]:
            return await self._list_tools()

        @self.server.call_tool()  # type: ignore[misc, no-untyped-call]
        async def call_tool(name: str, arguments: dict[str, Any]) -> list[Any]:
            return await self._call_tool(name, arguments or {})

    async def _on_list_tools_v2(self, ctx: Any, params: Any) -> Any:
        """mcp 2.x list-tools handler: wrap into a full ListToolsResult."""
        import mcp.types as types

        return types.ListToolsResult(tools=await self._list_tools())

    async def _on_call_tool_v2(self, ctx: Any, params: Any) -> Any:
        """mcp 2.x call-tool handler.

        Unlike 1.x, the SDK no longer wraps returned content lists or raised
        exceptions — return a full CallToolResult and mark failures with
        ``is_error`` ourselves.
        """
        import mcp.types as types

        try:
            content = await self._call_tool(params.name, params.arguments or {})
            return types.CallToolResult(content=content)
        except Exception as e:
            text = str(e) or type(e).__name__
            if not text.startswith("Error"):
                text = f"Error: {text}"
            return types.CallToolResult(
                content=[types.TextContent(type="text", text=text)],
                is_error=True,
            )

    # ------------------------------------------------------------------
    # Meta-tools
    # ------------------------------------------------------------------

    def _get_meta_tools(self) -> list[Tool]:
        """Return the meta-tools for dynamic tool discovery."""
        return [
            Tool(
                name="find_relevant_tools",
                description=(
                    "Search for tools relevant to your current task. "
                    "Use this before calling other tools to discover what's available. "
                    "Each match reports the exact tool_name to pass to execute_tool and "
                    "its JSON Schema parameters."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "What you're trying to accomplish",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Max tools to return",
                            "default": self._find_limit,
                            "minimum": _FIND_LIMIT_MIN,
                            "maximum": _FIND_LIMIT_MAX,
                        },
                    },
                    "required": ["query"],
                },
            ),
            Tool(
                name="execute_tool",
                description=(
                    "Execute a tool by name. Use find_relevant_tools first "
                    "to discover available tools and their schemas."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "tool_name": {
                            "type": "string",
                            "description": (
                                "Name of the tool to execute, exactly as reported by "
                                "find_relevant_tools"
                            ),
                        },
                        "arguments": {
                            "type": "object",
                            "description": "Arguments for the tool, matching its parameters schema",
                        },
                    },
                    "required": ["tool_name", "arguments"],
                },
            ),
        ]

    def _convert_tool(self, tool_def: Any, *, wire_name: str | None = None) -> Tool:
        """
        Convert ToolDefinition to MCP Tool.

        Args:
            tool_def: ToolDefinition object
            wire_name: Name to expose on the wire (defaults to the tool's own)

        Returns:
            MCP Tool object
        """
        return Tool(
            name=wire_name or tool_def.name,
            description=tool_def.description,
            inputSchema=tool_def.parameters_schema,
        )

    @staticmethod
    def _reported_name(tool: ToolDefinition) -> str:
        """The name a client should pass back to ``execute_tool``.

        Qualified for tools outside the default namespace, so two servers'
        ``search`` tools stay distinguishable; ``ToolCall`` resolves the
        dotted form.
        """
        return tool.name if tool.namespace == "default" else f"{tool.namespace}.{tool.name}"

    async def _handle_find_relevant_tools(self, arguments: dict[str, Any]) -> list[dict[str, Any]]:
        """
        Handle find_relevant_tools meta-tool.

        Args:
            arguments: Query and limit parameters

        Returns:
            One text block per matching tool (or one explaining that nothing matched)
        """
        query = str(arguments.get("query") or "").strip()
        if not query:
            raise ValueError("find_relevant_tools requires a non-empty 'query'")
        try:
            limit = int(arguments.get("limit") or self._find_limit)
        except (TypeError, ValueError):
            limit = self._find_limit
        limit = max(_FIND_LIMIT_MIN, min(limit, _FIND_LIMIT_MAX))

        # Use AgentGantry's semantic routing. Threshold 0.0, like every other
        # convenience layer: ToolQuery's 0.5 default is an absolute cosine
        # cutoff that silently returns nothing for embedders whose scores sit
        # below it, and an MCP client cannot tell "no match" from "filtered".
        context = ConversationContext(query=query)
        tool_query = ToolQuery(context=context, limit=limit, score_threshold=0.0)
        result = await self.gantry.retrieve(tool_query)

        if not result.tools:
            return [_text_block(f"No tools matched {query!r}. Try rephrasing the task.")]

        # Format results for MCP
        tools_info = []
        for scored_tool in result.tools:
            tool = scored_tool.tool
            lines = [
                f"Tool: {self._reported_name(tool)}",
                f"Description: {tool.description}",
                f"Parameters: {json.dumps(tool.parameters_schema, ensure_ascii=False)}",
                f"Relevance Score: {scored_tool.semantic_score:.2f}",
            ]
            if tool.requires_confirmation:
                lines.append("Note: this tool requires human confirmation before it runs.")
            if tool.deprecated:
                lines.append(
                    "Note: deprecated"
                    + (f" — {tool.deprecation_message}" if tool.deprecation_message else "")
                    + (f"; use {tool.superseded_by}" if tool.superseded_by else "")
                    + "."
                )
            tools_info.append(_text_block("\n".join(lines) + "\n"))

        return tools_info

    async def _handle_execute_tool(self, arguments: dict[str, Any]) -> list[dict[str, Any]]:
        """
        Handle execute_tool meta-tool or direct tool execution.

        Args:
            arguments: Tool name and arguments

        Returns:
            Tool execution result
        """
        tool_name = str(arguments.get("tool_name") or "").strip()
        if not tool_name:
            raise ValueError("execute_tool requires a 'tool_name'")
        tool_arguments = arguments.get("arguments")
        if tool_arguments is None:
            tool_arguments = {}
        if not isinstance(tool_arguments, dict):
            raise ValueError("execute_tool 'arguments' must be an object")
        return await self._execute(tool_name, tool_arguments)

    async def _execute(
        self, tool_name: str, tool_arguments: dict[str, Any], *, namespace: str | None = None
    ) -> list[dict[str, Any]]:
        """Run a tool through the gantry and format the outcome for MCP."""
        from agent_gantry.schema.execution import ToolCall

        call = ToolCall(tool_name=tool_name, namespace=namespace, arguments=tool_arguments)
        result = await self.gantry.execute(call)

        # Format result for MCP
        if result.status.value == "success":
            return [_text_block(_render_tool_output(result.result))]
        # Raise so failures surface as MCP errors (isError on the result):
        # the 1.x SDK wraps raised handler exceptions itself and the 2.x
        # handler sets is_error explicitly. Returning error text here would
        # make clients record the failed execution as a success.
        raise RuntimeError(
            f"Error: {result.error or 'Unknown error'} "
            f"(tool '{tool_name}', status {result.status.value})"
        )

    # ------------------------------------------------------------------
    # Transports
    # ------------------------------------------------------------------

    async def run_stdio(self) -> None:
        """Run the server with stdio transport."""
        async with stdio_server() as (read, write):
            await self.server.run(
                read,
                write,
                self.server.create_initialization_options(),
            )

    def streamable_http_app(
        self,
        path: str = "/mcp",
        *,
        json_response: bool = False,
        stateless: bool = False,
        allowed_hosts: Sequence[str] | None = None,
        allowed_origins: Sequence[str] | None = None,
    ) -> Any:
        """Build a Starlette app serving this server over Streamable HTTP.

        Use this to mount Gantry into an existing ASGI service
        (``app.mount("/", gantry_mcp.streamable_http_app())``); ``run_http``
        serves it standalone. Either way works: the returned app starts the
        SDK's session manager from its own lifespan when served directly, and
        on the first request when mounted — a parent Starlette or FastAPI app
        does not run a mounted sub-application's lifespan, so relying on that
        alone would leave the manager uninitialised.

        That cuts both ways at shutdown: a mounted app is never told to stop
        either, so the manager it started would live until the process ends.
        The owner is published as ``app.state.mcp_session`` for a host that
        wants to release it explicitly::

            mcp_app = gantry_mcp.streamable_http_app()
            host.mount("/tools", mcp_app)

            @host.on_event("shutdown")          # or your lifespan's finally
            async def _stop_mcp() -> None:
                await mcp_app.state.mcp_session.stop()

        Stopping is final: the SDK's manager may be entered only once, so
        build a new app to serve again.

        Args:
            path: Endpoint path clients connect to (default ``/mcp``).
            json_response: Reply with plain JSON instead of an SSE stream
                where the protocol allows it.
            stateless: Create a fresh transport per request (no session
                affinity) — for horizontally scaled deployments.
            allowed_hosts: ``Host`` header values to accept (e.g.
                ``["localhost:8000", "127.0.0.1:*"]``). Setting this (or
                ``allowed_origins``) enables the SDK's DNS-rebinding
                protection.
            allowed_origins: ``Origin`` header values to accept.
        """
        from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
        from starlette.applications import Starlette
        from starlette.routing import Mount, Route

        manager = StreamableHTTPSessionManager(
            app=self.server,
            json_response=json_response,
            stateless=stateless,
            security_settings=_security_settings(allowed_hosts, allowed_origins),
        )

        handler = _StreamableHTTPApp(manager)

        @contextlib.asynccontextmanager
        async def lifespan(app: Any) -> AsyncIterator[None]:
            # Eager start when this app is served directly; harmless when it
            # is mounted and this never runs, because the first request
            # starts the manager itself.
            await handler.start()
            try:
                yield
            finally:
                await handler.stop()

        path = "/" + path.strip("/")
        routes: list[Any] = [Route(path, endpoint=handler, methods=["GET", "POST", "DELETE"])]
        if path != "/":
            # ``Mount`` serves the trailing-slash form and any sub-path
            # without a redirect round-trip for clients that add one.
            routes.append(Mount(path, app=handler))
        app = Starlette(routes=routes, lifespan=lifespan)
        # The session manager's owner, published so a host that *mounts* this
        # app can release it: a parent application does not run a mounted
        # sub-application's lifespan, so the manager this app starts on its
        # first request would otherwise live until the process ends. Call
        # ``await app.state.mcp_session.stop()`` from the host's own shutdown.
        app.state.mcp_session = handler
        return app

    def sse_app(
        self,
        sse_path: str = "/sse",
        messages_path: str = "/messages/",
        *,
        allowed_hosts: Sequence[str] | None = None,
        allowed_origins: Sequence[str] | None = None,
    ) -> Any:
        """Build a Starlette app serving this server over the legacy SSE transport.

        Prefer :meth:`streamable_http_app`; this exists for clients that have
        not moved off the 2024-11-05 SSE transport.
        """
        from mcp.server.sse import SseServerTransport
        from starlette.applications import Starlette
        from starlette.routing import Mount, Route

        messages_path = "/" + messages_path.strip("/") + "/"
        sse = SseServerTransport(
            messages_path, security_settings=_security_settings(allowed_hosts, allowed_origins)
        )

        async def handle_sse(scope: Any, receive: Any, send: Any) -> None:
            async with sse.connect_sse(scope, receive, send) as streams:
                await self.server.run(
                    streams[0], streams[1], self.server.create_initialization_options()
                )

        return Starlette(
            routes=[
                Route("/" + sse_path.strip("/"), endpoint=_ASGIProxy(handle_sse), methods=["GET"]),
                Mount(messages_path.rstrip("/"), app=sse.handle_post_message),
            ]
        )

    async def run_http(
        self,
        host: str = "127.0.0.1",
        port: int = 8000,
        path: str = "/mcp",
        *,
        json_response: bool = False,
        stateless: bool = False,
        allowed_hosts: Sequence[str] | None = None,
        allowed_origins: Sequence[str] | None = None,
        log_level: str = "info",
    ) -> None:
        """Serve over Streamable HTTP with uvicorn until cancelled.

        Args:
            host: Interface to bind. Defaults to loopback; bind ``0.0.0.0``
                deliberately, and pair it with ``allowed_hosts``.
            port: TCP port.
            path: Endpoint path (clients connect to ``http://host:port/path``).
            json_response, stateless, allowed_hosts, allowed_origins: see
                :meth:`streamable_http_app`.
            log_level: uvicorn log level.
        """
        app = self.streamable_http_app(
            path,
            json_response=json_response,
            stateless=stateless,
            allowed_hosts=allowed_hosts,
            allowed_origins=allowed_origins,
        )
        await self._serve_asgi(app, host, port, log_level)

    async def run_sse(
        self,
        host: str = "127.0.0.1",
        port: int = 8000,
        *,
        sse_path: str = "/sse",
        messages_path: str = "/messages/",
        allowed_hosts: Sequence[str] | None = None,
        allowed_origins: Sequence[str] | None = None,
        log_level: str = "info",
    ) -> None:
        """
        Run the server with the legacy SSE transport.

        Args:
            host: Host to bind to
            port: Port to listen on
            sse_path: Path clients open the event stream on
            messages_path: Path clients POST messages to
            allowed_hosts, allowed_origins: see :meth:`streamable_http_app`
            log_level: uvicorn log level
        """
        app = self.sse_app(
            sse_path,
            messages_path,
            allowed_hosts=allowed_hosts,
            allowed_origins=allowed_origins,
        )
        await self._serve_asgi(app, host, port, log_level)

    @staticmethod
    async def _serve_asgi(app: Any, host: str, port: int, log_level: str) -> None:
        """Run ``app`` with uvicorn on the current event loop."""
        try:
            import uvicorn
        except ImportError as exc:  # pragma: no cover - uvicorn ships with mcp
            raise ImportError(
                "uvicorn is required for the HTTP MCP transports. Install it with "
                "'pip install agent-gantry[mcp]'."
            ) from exc

        config = uvicorn.Config(app, host=host, port=port, log_level=log_level, lifespan="on")
        server = uvicorn.Server(config)
        await server.serve()


def create_mcp_server(
    gantry: AgentGantry,
    mode: str = "dynamic",
    name: str = "agent-gantry",
    *,
    expose: Iterable[str] | None = None,
    find_limit: int = 5,
) -> MCPServer:
    """
    Create an MCP server for AgentGantry.

    Args:
        gantry: AgentGantry instance to serve
        mode: Server mode (dynamic, static, or hybrid)
        name: Server name for identification
        expose: Tools listed directly in hybrid mode (``name`` or ``namespace.name``)
        find_limit: Default result count for ``find_relevant_tools``

    Returns:
        MCPServer instance
    """
    return MCPServer(gantry, mode, name, expose=expose, find_limit=find_limit)
