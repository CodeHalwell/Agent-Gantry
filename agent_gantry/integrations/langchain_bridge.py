"""
LangChain bridge for Agent-Gantry.

Provides first-class integration between Agent-Gantry's semantic tool routing
and LangChain (1.x). The bridge converts Gantry tool definitions into native
LangChain ``StructuredTool`` instances that any LangChain or LangGraph agent
can consume directly, while routing every invocation back through Gantry's
executor so the security policy, retries, circuit breakers, and telemetry
all still apply.

Key class:
    - :class:`GantryToolBridge` — wraps Gantry tools as LangChain tools.

Usage::

    from agent_gantry import AgentGantry
    from agent_gantry.core.security import SecurityPolicy
    from agent_gantry.integrations.langchain_bridge import GantryToolBridge

    gantry = AgentGantry()

    @gantry.register
    def get_weather(city: str) -> str:
        '''Get current weather for a city.'''
        return f"Weather in {city}: Sunny, 22C"

    await gantry.sync()

    bridge = GantryToolBridge(
        gantry,
        security_policy=SecurityPolicy(require_confirmation=["delete_*"]),
    )
    tools = await bridge.get_tools("What's the weather?", limit=3)

    # Pass directly to any LangChain / LangGraph agent:
    from langchain.agents import create_agent
    agent = create_agent(model="openai:gpt-4o", tools=tools)
    result = await agent.ainvoke({"messages": [{"role": "user", "content": "..."}]})
"""

from __future__ import annotations

import json
import logging
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field, create_model

from agent_gantry.core.security import (
    ConfirmationRequiredError,
    PermissionDeniedError,
    SecurityPolicy,
)
from agent_gantry.schema.execution import ToolCall
from agent_gantry.schema.tool import ToolCapability

if TYPE_CHECKING:
    from agent_gantry.core.gantry import AgentGantry
    from agent_gantry.schema.query import RetrievalResult
    from agent_gantry.schema.tool import ToolDefinition

logger = logging.getLogger(__name__)


_APPROVAL_REQUIRED_CAPS: frozenset[ToolCapability] = frozenset(
    {
        ToolCapability.WRITE_DATA,
        ToolCapability.DELETE_DATA,
        ToolCapability.EXECUTE_CODE,
        ToolCapability.FINANCIAL,
        ToolCapability.PII_ACCESS,
    }
)


ApprovalCallback = Callable[
    ["ToolDefinition", dict[str, Any]],
    "bool | Awaitable[bool]",
]
"""Callback invoked when a tool requires human approval before execution.

Receives the ``ToolDefinition`` and the call arguments. Return ``True`` to
allow the call to proceed, ``False`` to deny (raises
:class:`PermissionDeniedError`). May be sync or async.
"""


def _require_langchain_installed(caller: str) -> None:
    """Raise a descriptive ImportError when langchain-core is not installed."""
    try:
        import langchain_core  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            f"{caller}() requires the 'langchain-core' package. "
            "Install with: pip install 'agent-gantry[agent-frameworks]'"
        ) from exc


def _json_type_to_python(json_type: str) -> Any:
    """Map a JSON Schema type string to a Python type for Pydantic fields."""
    mapping: dict[str, Any] = {
        "string": str,
        "integer": int,
        "number": float,
        "boolean": bool,
        "array": list,
        "object": dict,
    }
    return mapping.get(json_type, str)


def _build_args_schema(tool_def: ToolDefinition) -> type[BaseModel]:
    """Construct a Pydantic ``args_schema`` from a Gantry tool's JSON schema.

    LangChain's ``StructuredTool`` uses an ``args_schema`` (a Pydantic model
    class) to validate inputs and to render the tool definition for the LLM.
    Building it dynamically from Gantry's ``parameters_schema`` keeps the
    field names, descriptions, and required/optional state in sync with the
    underlying tool registration.

    Optional parameters are typed as ``py_type | None`` only when the
    schema either omits a default or explicitly defaults to ``None`` —
    surfacing ``None`` as a valid value to LangChain. Optional parameters
    with concrete defaults (e.g. ``{"default": "celsius"}``) keep their
    declared type so the generated schema doesn't advertise a spurious
    ``null`` shape that could confuse the model or downstream tools.
    """
    params_schema = tool_def.parameters_schema or {}
    properties: dict[str, dict[str, Any]] = params_schema.get("properties", {})
    required: set[str] = set(params_schema.get("required", []))

    _missing = object()
    fields: dict[str, tuple[Any, Any]] = {}
    for name, info in properties.items():
        py_type = _json_type_to_python(info.get("type", "string"))
        description = info.get("description") or f"Parameter: {name}"
        if name in required:
            fields[name] = (py_type, Field(..., description=description))
            continue

        default = info.get("default", _missing)
        # Allow None only when the schema either omits a default or
        # explicitly defaults to null. Concrete defaults keep their
        # declared type so the LLM-facing schema isn't widened to nullable.
        if default is _missing or default is None:
            field_type = py_type | None
            field_default = None
        else:
            field_type = py_type
            field_default = default
        fields[name] = (field_type, Field(default=field_default, description=description))

    # Pydantic refuses to build empty models with create_model; use a sentinel
    # field-less subclass so StructuredTool still has a valid args_schema.
    if not fields:
        return create_model(f"{tool_def.name}_NoArgs", __base__=BaseModel)

    return create_model(f"{tool_def.name}_Args", __base__=BaseModel, **fields)


def _tool_requires_approval(tool_def: ToolDefinition) -> bool:
    """Return True if the tool's capabilities mark it as approval-gated."""
    return bool(set(tool_def.capabilities) & _APPROVAL_REQUIRED_CAPS)


def _cache_key(tool_def: ToolDefinition) -> str:
    return f"{tool_def.namespace}:{tool_def.name}:{tool_def.version}"


class GantryToolBridge:
    """
    Bridge between Agent-Gantry and LangChain (and LangGraph).

    Retrieves semantically relevant tools from Gantry and wraps them as
    LangChain ``StructuredTool`` instances. The wrapped tools route every
    invocation back through ``gantry.execute(...)`` so retries, circuit
    breakers, security policy, and telemetry all flow through uniformly.

    Args:
        gantry: The AgentGantry instance providing tool retrieval and execution.
        score_threshold: Default minimum relevance score for tool selection.
        security_policy: Optional :class:`SecurityPolicy` enforced before each
            tool invocation. When a tool matches a ``require_confirmation``
            pattern, the bridge consults ``approval_callback`` (if set) or
            raises :class:`ConfirmationRequiredError` if none is configured.
        approval_callback: Optional async-or-sync callable consulted when a
            tool requires human approval (either because the
            ``security_policy`` flagged it, or because the tool's capabilities
            include destructive ones such as ``WRITE_DATA`` / ``DELETE_DATA``
            / ``EXECUTE_CODE`` / ``FINANCIAL`` / ``PII_ACCESS``). Receives the
            ``ToolDefinition`` and the call arguments and must return a bool.
        capability_approval: When ``True`` (default), tools whose capabilities
            include any of the destructive set are also gated by
            ``approval_callback``, mirroring the AF bridge's
            ``approval_mode="always_require"`` behaviour. Set ``False`` to
            only gate tools matched by ``security_policy.require_confirmation``.
    """

    def __init__(
        self,
        gantry: AgentGantry,
        *,
        score_threshold: float = 0.3,
        security_policy: SecurityPolicy | None = None,
        approval_callback: ApprovalCallback | None = None,
        capability_approval: bool = True,
    ) -> None:
        self._gantry = gantry
        self._score_threshold = score_threshold
        self._security_policy = security_policy
        self._approval_callback = approval_callback
        self._capability_approval = capability_approval
        self._tool_cache: dict[str, Any] = {}

    async def _retrieve(
        self,
        query: str,
        *,
        limit: int = 5,
        score_threshold: float | None = None,
        **query_kwargs: Any,
    ) -> RetrievalResult:
        from agent_gantry.schema.query import ConversationContext, ToolQuery

        threshold = (
            score_threshold if score_threshold is not None else self._score_threshold
        )

        context_fields = set(ConversationContext.model_fields.keys()) - {"query"}
        context_kwargs = {
            k: v for k, v in query_kwargs.items() if k in context_fields
        }
        tool_query_fields = set(ToolQuery.model_fields.keys()) - {
            "context",
            "limit",
            "score_threshold",
        }
        tq_kwargs = {k: v for k, v in query_kwargs.items() if k in tool_query_fields}

        return await self._gantry.retrieve(
            ToolQuery(
                context=ConversationContext(query=query, **context_kwargs),
                limit=limit,
                score_threshold=threshold,
                **tq_kwargs,
            )
        )

    async def _consult_approval(
        self,
        tool_def: ToolDefinition,
        arguments: dict[str, Any],
        reason: str,
    ) -> None:
        """Invoke the approval callback (if any), or raise."""
        if self._approval_callback is None:
            raise ConfirmationRequiredError(
                f"Tool {tool_def.name} requires human approval ({reason}); "
                "no approval_callback configured on GantryToolBridge."
            )

        result = self._approval_callback(tool_def, arguments)
        if hasattr(result, "__await__"):
            result = await result  # type: ignore[assignment]
        if not result:
            raise PermissionDeniedError(
                f"Tool {tool_def.name} denied by approval_callback ({reason})."
            )

    async def _gate(
        self, tool_def: ToolDefinition, arguments: dict[str, Any]
    ) -> None:
        """Run security-policy + capability-approval checks before execution.

        A single tool invocation triggers at most one approval prompt even
        when both gates apply: the policy check runs first, and if it
        already consulted the approval callback we skip the capability
        check rather than asking the human a second time for the same call.
        """
        approval_consulted = False

        if self._security_policy is not None:
            try:
                self._security_policy.check_permission(tool_def.name, arguments)
            except ConfirmationRequiredError as err:
                logger.info(
                    "GantryToolBridge: '%s' requires approval (policy: %s)",
                    tool_def.name,
                    err,
                )
                await self._consult_approval(tool_def, arguments, "policy match")
                approval_consulted = True
            except PermissionDeniedError:
                logger.warning(
                    "GantryToolBridge: denied execution of '%s' by policy",
                    tool_def.name,
                )
                raise

        if (
            not approval_consulted
            and self._capability_approval
            and _tool_requires_approval(tool_def)
        ):
            await self._consult_approval(
                tool_def, arguments, "destructive capability"
            )

    def _wrap_tool(self, tool_def: ToolDefinition) -> Any:
        """Build a LangChain ``StructuredTool`` for a single Gantry tool."""
        _require_langchain_installed("GantryToolBridge")
        from langchain_core.tools import StructuredTool

        args_schema = _build_args_schema(tool_def)
        gantry = self._gantry
        bridge = self
        tool_name = tool_def.name
        tool_desc = tool_def.description or f"Gantry tool: {tool_name}"

        async def _arun(**kwargs: Any) -> Any:
            await bridge._gate(tool_def, kwargs)
            telemetry = getattr(gantry, "_telemetry", None)
            span_cm = None
            if telemetry is not None:
                try:
                    span_cm = telemetry.span(
                        "langchain_tool_invocation", {"tool_name": tool_name}
                    )
                except Exception:  # pragma: no cover - telemetry best-effort
                    logger.debug(
                        "GantryToolBridge: telemetry.span failed",
                        exc_info=True,
                    )
                    span_cm = None

            async def _invoke() -> Any:
                result = await gantry.execute(
                    ToolCall(tool_name=tool_name, arguments=kwargs)
                )
                status = (
                    result.status.value
                    if hasattr(result.status, "value")
                    else str(result.status)
                )
                if status == "success":
                    val = result.result
                    return val if isinstance(val, str) else json.dumps(val)
                # Surfacing the error via exception lets LangChain agents
                # handle/retry; bare strings would be returned as the tool's
                # observation and the agent might continue blindly.
                raise RuntimeError(
                    f"{tool_name} failed: {result.error or 'tool execution failed'}"
                )

            if span_cm is None:
                return await _invoke()
            async with span_cm:
                return await _invoke()

        def _run(**kwargs: Any) -> Any:
            # Sync entrypoint: LangChain calls this when the agent isn't async.
            # Delegate to the async path via asyncio.run when there is no
            # running loop; otherwise fall back to a fresh event loop on a
            # thread to avoid "asyncio.run() cannot be called from a running
            # event loop" inside LangGraph's threadpool.
            import asyncio

            try:
                asyncio.get_running_loop()
            except RuntimeError:
                return asyncio.run(_arun(**kwargs))

            import threading

            container: dict[str, Any] = {}

            def _runner() -> None:
                try:
                    container["result"] = asyncio.run(_arun(**kwargs))
                except BaseException as exc:  # pragma: no cover - re-raised below
                    container["error"] = exc

            thread = threading.Thread(target=_runner, daemon=True)
            thread.start()
            thread.join()
            if "error" in container:
                raise container["error"]
            return container["result"]

        return StructuredTool.from_function(
            func=_run,
            coroutine=_arun,
            name=tool_name,
            description=tool_desc,
            args_schema=args_schema,
        )

    def wrap_tool(self, tool_def: ToolDefinition, *, cache: bool = True) -> Any:
        """Wrap a single ``ToolDefinition`` as a LangChain ``StructuredTool``."""
        if not cache:
            return self._wrap_tool(tool_def)
        key = _cache_key(tool_def)
        existing = self._tool_cache.get(key)
        if existing is not None:
            return existing
        wrapped = self._wrap_tool(tool_def)
        self._tool_cache[key] = wrapped
        return wrapped

    def wrap_tools(
        self, tool_defs: list[ToolDefinition], *, cache: bool = True
    ) -> list[Any]:
        """Wrap a list of ``ToolDefinition`` objects."""
        return [self.wrap_tool(td, cache=cache) for td in tool_defs]

    async def get_tools(
        self,
        query: str,
        *,
        limit: int = 5,
        score_threshold: float | None = None,
        cache: bool = True,
        **query_kwargs: Any,
    ) -> list[Any]:
        """Retrieve top-k tools for ``query`` and return them as LangChain tools."""
        result = await self._retrieve(
            query, limit=limit, score_threshold=score_threshold, **query_kwargs
        )
        return [self.wrap_tool(scored.tool, cache=cache) for scored in result.tools]

    async def get_tools_with_scores(
        self,
        query: str,
        *,
        limit: int = 5,
        score_threshold: float | None = None,
        cache: bool = True,
        **query_kwargs: Any,
    ) -> list[tuple[Any, float]]:
        """Retrieve top-k tools alongside their relevance scores."""
        result = await self._retrieve(
            query, limit=limit, score_threshold=score_threshold, **query_kwargs
        )
        return [
            (self.wrap_tool(scored.tool, cache=cache), scored.final_score)
            for scored in result.tools
        ]

    async def build_agent(
        self,
        model: Any,
        query: str,
        *,
        limit: int = 5,
        score_threshold: float | None = None,
        extra_tools: list[Any] | None = None,
        **create_agent_kwargs: Any,
    ) -> Any:
        """Build a ``langchain.agents.create_agent`` agent wired with retrieved tools.

        Convenience wrapper for the common case where you want a ready-to-run
        LangChain agent backed by Gantry-selected tools. Returns whatever
        ``langchain.agents.create_agent`` returns (a ``CompiledStateGraph``
        in LangChain 1.x).

        Args:
            model: The model passed to ``create_agent`` — either an
                initialised LangChain chat model or a provider string such
                as ``"openai:gpt-4o"``.
            query: The user query used to seed semantic tool retrieval.
            limit: Maximum tools to surface from Gantry.
            score_threshold: Override the bridge default score threshold.
            extra_tools: Additional LangChain tools to include alongside the
                Gantry-retrieved set (e.g. tools that aren't registered with
                Gantry).
            **create_agent_kwargs: Forwarded to ``create_agent``.

        Raises:
            ModuleNotFoundError: If ``langchain`` is not installed.
            ImportError: If ``langchain`` is installed but ``create_agent``
                cannot be imported (e.g. an older release without that
                symbol). The original error is propagated unchanged so the
                version-mismatch hint is visible.
        """
        try:
            from langchain.agents import create_agent
        except ModuleNotFoundError as exc:  # pragma: no cover - surfaced via tests
            raise ModuleNotFoundError(
                "build_agent() requires the 'langchain' package. "
                "Install with: pip install 'agent-gantry[agent-frameworks]'"
            ) from exc
        # Any other ImportError (e.g. "cannot import name 'create_agent' from
        # 'langchain.agents'" on older LangChain releases) propagates with its
        # original message — that is more actionable than a synthesised
        # "package not installed" hint.

        tools = await self.get_tools(
            query, limit=limit, score_threshold=score_threshold
        )
        if extra_tools:
            tools = [*tools, *extra_tools]
        return create_agent(model=model, tools=tools, **create_agent_kwargs)

    def clear_cache(self) -> None:
        """Drop all cached wrapped tools — useful when tool definitions change."""
        self._tool_cache.clear()


__all__ = [
    "ApprovalCallback",
    "GantryToolBridge",
]
