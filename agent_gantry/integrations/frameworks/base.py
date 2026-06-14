"""Shared base for native per-framework tool adapters.

Agent-Gantry's job is to *select* a small, relevant slice of tools from a large
registry. To hand those tools to a concrete agent framework (LangChain,
LlamaIndex, CrewAI, Pydantic AI, OpenAI Agents SDK, Smolagents, Haystack, Agno,
…) they must be wrapped as that framework's *native* tool object — a callable
that the framework can introspect (name, description, JSON-schema parameters)
and invoke.

Every adapter in this subpackage follows the same shape:

    toolset = GantryToolset(gantry)
    specs   = await toolset.select(query, limit=3)      # ranked ToolSpec list
    native  = [spec.to_<framework>() ...]               # framework-specific

``ToolSpec`` is the framework-neutral handle. It exposes the metadata a
framework needs plus two invocation entry points that both route through
``gantry.execute`` (so retries, timeouts, circuit breakers, and the security
policy all still apply):

- :meth:`ToolSpec.ainvoke` — async, ``**kwargs`` or a single dict.
- :meth:`ToolSpec.invoke`  — sync wrapper (errors inside a running loop).

The adapters are intentionally dependency-free at import time: the third-party
framework is imported lazily inside the ``to_*`` builder, so ``import
agent_gantry`` never requires LangChain et al. to be installed.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from agent_gantry.schema.execution import ExecutionStatus, ToolCall
from agent_gantry.schema.query import ConversationContext, ToolQuery

if TYPE_CHECKING:
    from agent_gantry.core.gantry import AgentGantry
    from agent_gantry.schema.tool import ToolDefinition


class ToolExecutionError(RuntimeError):
    """Raised when a Gantry-backed tool invocation does not succeed."""

    def __init__(self, tool_name: str, status: str, error: str | None) -> None:
        self.tool_name = tool_name
        self.status = status
        self.error = error
        super().__init__(
            f"Tool {tool_name!r} failed (status={status}): {error or 'no detail'}"
        )


@dataclass(frozen=True)
class ToolSpec:
    """Framework-neutral handle to one selected Gantry tool.

    Attributes:
        name: The tool's bare name (what the model calls).
        qualified_name: ``namespace.name`` — unique within a registry.
        description: Human/LLM-facing description.
        parameters: JSON-Schema object describing the arguments.
        requires_confirmation: Whether the underlying tool is high-risk.
        score: The semantic score from selection (0.0 if unknown).
    """

    name: str
    qualified_name: str
    description: str
    parameters: dict[str, Any]
    requires_confirmation: bool
    score: float
    _gantry: AgentGantry
    _namespace: str

    # -- invocation -------------------------------------------------------- #
    async def ainvoke(self, *args: Any, **kwargs: Any) -> Any:
        """Execute the tool through Gantry and return its raw result.

        Accepts either a single positional mapping (``ainvoke({"a": 1})``) or
        keyword arguments (``ainvoke(a=1)``). Raises :class:`ToolExecutionError`
        on any non-success status so framework error handling kicks in.
        """
        arguments = _coerce_arguments(args, kwargs)
        result = await self._gantry.execute(
            ToolCall(tool_name=self.name, arguments=arguments)
        )
        if result.status != ExecutionStatus.SUCCESS:
            raise ToolExecutionError(
                self.name, getattr(result.status, "value", str(result.status)), result.error
            )
        return result.result

    def invoke(self, *args: Any, **kwargs: Any) -> Any:
        """Synchronous wrapper around :meth:`ainvoke`.

        Safe to call from synchronous framework code (CrewAI ``_run``,
        Smolagents ``forward``, Haystack/Agno function entrypoints, …) whether
        or not an event loop is already running on the current thread. When a
        loop is running, the coroutine is executed on a dedicated worker thread
        and this call blocks for the result — mirroring how those frameworks
        invoke a synchronous tool. Prefer :meth:`ainvoke` directly in async code.
        """
        return _run_coroutine_sync(self.ainvoke(*args, **kwargs))

    def callable_for_signature(self) -> Callable[..., Any]:
        """Return a plain async function that calls this tool by keyword.

        Frameworks that build their own tool object from a function (Smolagents,
        Agno, Pydantic AI, OpenAI Agents SDK) can wrap this. The returned
        function carries ``__name__`` / ``__doc__`` so introspection works.
        """

        async def _fn(**kwargs: Any) -> Any:
            return await self.ainvoke(**kwargs)

        _fn.__name__ = self.name
        _fn.__doc__ = self.description
        return _fn


def _run_coroutine_sync(coro: Any) -> Any:
    """Run an awaitable to completion from synchronous code, loop-or-not.

    If no event loop runs on the current thread, use :func:`asyncio.run`.
    Otherwise (we're inside a running loop — e.g. a framework invoked our sync
    tool from within its async agent loop), run the coroutine on a dedicated
    worker thread with its own loop and block for the result. This avoids the
    "coroutine attached to a different loop" / "loop already running" errors
    that a naive ``asyncio.run`` would raise.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(lambda: asyncio.run(coro)).result()


def _coerce_arguments(args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
    """Normalize ``(*args, **kwargs)`` into a single argument dict."""
    if args:
        if len(args) == 1 and isinstance(args[0], dict) and not kwargs:
            return dict(args[0])
        raise TypeError(
            "Tool invocation accepts a single mapping positional argument or "
            "keyword arguments, not both / multiple positionals."
        )
    return dict(kwargs)


def spec_from_tool(
    gantry: AgentGantry, tool: ToolDefinition, score: float = 0.0
) -> ToolSpec:
    """Build a :class:`ToolSpec` from a Gantry :class:`ToolDefinition`."""
    qualified = getattr(tool, "qualified_name", None) or f"{tool.namespace}.{tool.name}"
    params = tool.parameters_schema or {"type": "object", "properties": {}}
    return ToolSpec(
        name=tool.name,
        qualified_name=qualified,
        description=tool.description or tool.name,
        parameters=params,
        requires_confirmation=bool(getattr(tool, "requires_confirmation", False)),
        score=float(score),
        _gantry=gantry,
        _namespace=tool.namespace,
    )


class GantryToolset:
    """Framework-neutral entry point: select tools, then export to a framework.

    Example::

        toolset = GantryToolset(gantry)
        specs = await toolset.select("send an email", limit=3)
        lc_tools = await toolset.for_langchain("send an email", limit=3)
    """

    def __init__(self, gantry: AgentGantry, *, default_limit: int = 3) -> None:
        self._gantry = gantry
        self._default_limit = default_limit

    async def select(
        self,
        query: str,
        *,
        limit: int | None = None,
        score_threshold: float = 0.0,
        namespaces: list[str] | None = None,
        tools_already_used: list[str] | None = None,
    ) -> list[ToolSpec]:
        """Run semantic selection and return ranked :class:`ToolSpec` handles.

        ``score_threshold`` defaults to ``0.0`` (no filtering) — matching the
        high-level convenience API and avoiding the silent-drop trap of the raw
        ``ToolQuery`` 0.5 default.
        """
        context = ConversationContext(
            query=query,
            tools_already_used=list(tools_already_used or []),
        )
        result = await self._gantry.retrieve(
            ToolQuery(
                context=context,
                limit=limit or self._default_limit,
                score_threshold=score_threshold,
                namespaces=namespaces,
            )
        )
        return [
            spec_from_tool(self._gantry, st.tool, st.semantic_score) for st in result.tools
        ]
