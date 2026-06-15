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
- :meth:`ToolSpec.invoke`  — sync wrapper, safe to call even from inside a
  running event loop (it offloads to a shared worker thread and blocks).

The adapters are intentionally dependency-free at import time: the third-party
framework is imported lazily inside the ``to_*`` builder, so ``import
agent_gantry`` never requires LangChain et al. to be installed.
"""

from __future__ import annotations

import asyncio
import inspect
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

        ``None``-valued arguments for *optional* parameters are dropped: several
        frameworks (CrewAI, Pydantic-schema validators, …) materialize every
        declared optional field as ``None`` when the model didn't supply it,
        but the tool's JSON schema types those params (e.g. ``string``) and the
        executor rejects ``None``. Dropping them lets the tool's own default
        apply — ``None`` for a required param is kept so the error stays clear.
        """
        arguments = _coerce_arguments(args, kwargs)
        required = set(self.parameters.get("required") or [])
        arguments = {
            k: v for k, v in arguments.items() if v is not None or k in required
        }
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

    def callable_for_signature(
        self, *, union_optional: bool = False, type_matched_defaults: bool = False
    ) -> Callable[..., Any]:
        """Return a plain async function that calls this tool by keyword.

        Frameworks that build their own tool object from a function (Smolagents,
        Agno, Pydantic AI, OpenAI Agents SDK, AutoGen) can wrap this. The
        returned function carries ``__name__`` / ``__doc__`` **and a real
        ``__signature__``** derived from :attr:`parameters`, so frameworks that
        introspect the signature to build the LLM tool schema see the actual
        parameters instead of a bare ``**kwargs`` (which would surface as a
        no-argument tool).

        ``union_optional`` (Semantic Kernel) and ``type_matched_defaults``
        (Google ADK) are opt-in signature tweaks — see :meth:`python_signature`.
        """

        async def _fn(**kwargs: Any) -> Any:
            return await self.ainvoke(**kwargs)

        _fn.__name__ = self.name
        _fn.__doc__ = self.description
        _fn.__signature__ = self.python_signature(  # type: ignore[attr-defined]
            union_optional=union_optional, type_matched_defaults=type_matched_defaults
        )
        _fn.__annotations__ = {
            p.name: p.annotation
            for p in _fn.__signature__.parameters.values()
            if p.annotation is not inspect.Parameter.empty
        }
        return _fn

    def python_signature(
        self, *, union_optional: bool = False, type_matched_defaults: bool = False
    ) -> inspect.Signature:
        """Build an :class:`inspect.Signature` from the JSON-Schema parameters.

        Each property becomes a keyword-only parameter; required properties have
        no default. By default optional properties default to ``None``. Two
        opt-in modes adapt the signature for stricter frameworks:

        - ``union_optional``: annotate optional params ``T | None`` (Semantic
          Kernel infers required-ness from the annotation, not the default).
        - ``type_matched_defaults``: default optional params to a type-matched
          empty value (``""`` / ``0`` / ``False`` / ``[]`` / ``{}``) instead of
          ``None`` (Google ADK's automatic function calling rejects both union
          types and a ``None`` default whose type mismatches the annotation).
        """
        properties = self.parameters.get("properties") or {}
        required = set(self.parameters.get("required") or [])
        params: list[inspect.Parameter] = []
        for name, prop in properties.items():
            json_type = prop.get("type") if isinstance(prop, dict) else None
            annotation = _json_type_to_python(json_type)
            if name in required:
                default = inspect.Parameter.empty
            elif union_optional:
                # `T | None` — valid at runtime on the project's floor (3.10+,
                # enforced by ruff UP) and the form SK uses to infer optionality.
                annotation = annotation | None
                default = None
            elif type_matched_defaults:
                default = _typed_default(json_type)
            else:
                default = None
            params.append(
                inspect.Parameter(
                    name,
                    inspect.Parameter.KEYWORD_ONLY,
                    default=default,
                    annotation=annotation,
                )
            )
        return inspect.Signature(params)


_JSON_TO_PYTHON: dict[str, Any] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "array": list,
    "object": dict,
}

# Scalar "empty" default per JSON-Schema type, used to mark an optional
# parameter without a None default (which strict validators like Google ADK
# reject as incompatible with a non-Optional annotation).
_SCALAR_DEFAULTS: dict[str, Any] = {
    "string": "",
    "integer": 0,
    "number": 0.0,
    "boolean": False,
}


def _typed_default(json_type: Any) -> Any:
    """Return a fresh, type-matched empty default for an optional parameter."""
    if isinstance(json_type, list):  # e.g. ["string", "null"]
        json_type = next((t for t in json_type if t != "null"), None)
    if json_type == "array":
        return []
    if json_type == "object":
        return {}
    return _SCALAR_DEFAULTS.get(json_type, "")


def _json_type_to_python(json_type: Any) -> Any:
    """Map a JSON-Schema ``type`` to a Python annotation (default ``str``)."""
    if isinstance(json_type, list):  # e.g. ["string", "null"]
        json_type = next((t for t in json_type if t != "null"), None)
    return _JSON_TO_PYTHON.get(json_type, str)


# A single shared worker thread for running coroutines from sync framework
# callbacks while an event loop is active on the calling thread. Reused across
# invocations so we don't pay the spawn/teardown cost of a fresh pool each call.
_SYNC_BRIDGE_POOL: Any = None


def _run_coroutine_sync(coro: Any) -> Any:
    """Run an awaitable to completion from synchronous code, loop-or-not.

    If no event loop runs on the current thread, use :func:`asyncio.run`.
    Otherwise (we're inside a running loop — e.g. a framework invoked our sync
    tool from within its async agent loop), run the coroutine on a shared
    worker thread with its own loop and block for the result. This avoids the
    "coroutine attached to a different loop" / "loop already running" errors
    that a naive ``asyncio.run`` would raise.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    global _SYNC_BRIDGE_POOL
    if _SYNC_BRIDGE_POOL is None:
        import concurrent.futures

        _SYNC_BRIDGE_POOL = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="gantry-sync-bridge"
        )
    return _SYNC_BRIDGE_POOL.submit(lambda: asyncio.run(coro)).result()


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
        # turn the specs into a framework's native objects via that framework's
        # adapter (each exposes a ``convert`` classmethod):
        from agent_gantry.langchain import LangChainAdapter
        lc_tools = [LangChainAdapter.convert(s) for s in specs]
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
