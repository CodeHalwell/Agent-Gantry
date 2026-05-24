"""
Microsoft Agent Framework bridge for Agent-Gantry.

Provides seamless integration between Agent-Gantry's semantic tool routing
and Microsoft Agent Framework (1.0 GA) agents. The bridge converts Gantry
tool definitions into Python callables that AF agents can invoke directly,
enabling dynamic tool selection that reduces token usage in multi-agent systems.

Key classes:
    - ``GantryToolBridge``: Main bridge that wraps Gantry tools for AF agents.

Usage:
    from agent_gantry import AgentGantry
    from agent_gantry.integrations.agent_framework_bridge import GantryToolBridge

    gantry = AgentGantry()

    @gantry.register
    def get_weather(city: str) -> str:
        '''Get current weather for a city.'''
        return f"Weather in {city}: Sunny, 22C"

    await gantry.sync()

    bridge = GantryToolBridge(gantry)
    tools = await bridge.get_tools("What's the weather?", limit=3)

    # Pass directly to any AF agent:
    from agent_framework import Agent
    agent = Agent(client, "...", name="Assistant", tools=tools)
    result = await agent.run("What's the weather in London?")
"""

from __future__ import annotations

import inspect
import json
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Annotated, Any

from pydantic import Field

from agent_gantry.schema.execution import ToolCall
from agent_gantry.schema.tool import ToolCapability

if TYPE_CHECKING:
    from agent_gantry.core.gantry import AgentGantry
    from agent_gantry.schema.query import RetrievalResult
    from agent_gantry.schema.tool import ToolDefinition

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RetrievalCandidate:
    """A single tool that was considered during a retrieval round.

    Carries the qualified name and the raw semantic score, plus a flag
    indicating whether the candidate survived the configured score
    threshold and was therefore eligible for injection into the LLM
    prompt.
    """

    name: str
    qualified_name: str
    score: float
    kept: bool


@dataclass
class RetrievalDecision:
    """Structured record of *what just happened* during a retrieval round.

    The dataclass is intentionally cheap to construct and pickle so it
    can be attached to telemetry spans, dumped to logs, and inspected
    from middleware. ``GantryContextProvider.last_selection`` exposes
    the most-recent decision; the same shape is returned by
    :meth:`GantryContextProvider.dry_run_retrieve`.

    Fields:
        query: The query string that drove this retrieval.
        candidates: All candidates returned by the gantry, in score order.
            Each carries a ``kept`` flag indicating whether it passed the
            threshold filter.
        injected: Names of tools that were ultimately injected into the
            LLM tool list (top-K of candidates after thresholding, plus
            always_include / required pins / skills, in injection order).
        threshold: The effective ``score_threshold`` applied. ``None``
            for the relative-threshold mode; ``effective_threshold`` carries
            the resolved numeric cutoff for that case.
        threshold_mode: ``"absolute"`` for a fixed cosine cutoff,
            ``"relative:<frac>"`` for the relative mode.
        effective_threshold: The numeric cutoff actually applied for this
            round (the input ``threshold`` for absolute mode, or the
            resolved ``frac * top_score`` for relative mode).
    """

    query: str = ""
    candidates: list[RetrievalCandidate] = field(default_factory=list)
    injected: list[str] = field(default_factory=list)
    threshold: float | str | None = None
    threshold_mode: str = "absolute"
    effective_threshold: float | None = None

    @property
    def kept(self) -> list[RetrievalCandidate]:
        """Candidates that passed the threshold filter."""
        return [c for c in self.candidates if c.kept]

    @property
    def dropped(self) -> list[RetrievalCandidate]:
        """Candidates that were dropped by the threshold filter."""
        return [c for c in self.candidates if not c.kept]

    def summary(self, top: int = 5) -> str:
        """One-line ``INFO``-style summary, truncated to ``top`` candidates."""
        head = self.candidates[:top]
        rendered = ", ".join(f"{c.name}:{c.score:.2f}" for c in head)
        q = (self.query or "").strip().replace("\n", " ")
        if len(q) > 60:
            q = q[:60] + "…"
        return f'query="{q}" → top{min(top, len(head))}: [{rendered}]'

    def as_span_attributes(self) -> dict[str, Any]:
        """Flat attribute dict suitable for telemetry spans."""
        return {
            "query": self.query,
            "threshold_mode": self.threshold_mode,
            "effective_threshold": self.effective_threshold,
            "candidate_count": len(self.candidates),
            "injected_count": len(self.injected),
            "candidates": [c.qualified_name for c in self.candidates],
            "scores": [c.score for c in self.candidates],
            "kept": [c.qualified_name for c in self.candidates if c.kept],
            "injected": list(self.injected),
        }


def _parse_threshold(
    threshold: float | str | None,
) -> tuple[str, float | None]:
    """Decode a ``score_threshold`` value into ``(mode, numeric)``.

    Accepts:
    - ``None`` → ``("absolute", None)`` (no filtering, equivalent to 0.0).
    - ``float`` → ``("absolute", <value>)``.
    - ``"relative:<frac>"`` → ``("relative", <frac>)`` where ``<frac>``
      is the multiplier applied to the top candidate's score to produce
      the cutoff for the round.
    """
    if threshold is None:
        return "absolute", None
    if isinstance(threshold, (int, float)):
        return "absolute", float(threshold)
    if isinstance(threshold, str):
        s = threshold.strip().lower()
        if s.startswith("relative:"):
            try:
                frac = float(s.split(":", 1)[1])
            except ValueError as exc:
                raise ValueError(
                    f"Invalid relative score_threshold {threshold!r}: "
                    "expected 'relative:<float>' (e.g. 'relative:0.8')."
                ) from exc
            if not 0.0 <= frac <= 1.0:
                raise ValueError(
                    f"Relative score_threshold fraction must be in [0.0, 1.0], "
                    f"got {frac}."
                )
            return "relative", frac
        # Bare numeric strings — be forgiving.
        try:
            return "absolute", float(s)
        except ValueError as exc:
            raise ValueError(
                f"Invalid score_threshold {threshold!r}: expected a float "
                f"or 'relative:<float>'."
            ) from exc
    raise TypeError(
        f"score_threshold must be float, str, or None; got {type(threshold).__name__}."
    )


# Mirror of the ``le`` constraint on ``ToolQuery.limit`` (see
# ``agent_gantry/schema/query.py``). Kept as a module-level constant so the
# bridge can clamp its relative-mode over-fetch without re-introspecting
# the Pydantic model at call time. If the schema's upper bound changes,
# update this constant.
_TOOL_QUERY_MAX_LIMIT = 50


# Capabilities that indicate a tool is potentially destructive / requires
# explicit human approval in production. Mapped to AF's
# ``approval_mode="always_require"`` so that AF surfaces an approval event
# before the tool actually runs.
_APPROVAL_REQUIRED_CAPS: frozenset[ToolCapability] = frozenset(
    {
        ToolCapability.WRITE_DATA,
        ToolCapability.DELETE_DATA,
        ToolCapability.EXECUTE_CODE,
        ToolCapability.FINANCIAL,
        ToolCapability.PII_ACCESS,
    }
)


def _get_af_version() -> tuple[int, int, int] | None:
    """Return the installed agent-framework version as (major, minor, patch), or ``None``."""
    try:
        import importlib.metadata as _im
        raw = _im.version("agent-framework")
        parts = raw.split(".")
        return (int(parts[0]), int(parts[1]), int(parts[2]) if len(parts) > 2 else 0)
    except Exception:
        return None


def disable_af_instrumentation() -> bool:
    """Disable Microsoft Agent Framework's default instrumentation (AF >=1.6.0).

    AF 1.6.0 enables ``asyncio.ContextVar``-based telemetry by default.  When
    two ``Agent.run()`` coroutines run concurrently (e.g. via
    ``asyncio.gather()`` or ``TaskGroup``), CPython raises::

        ValueError: <Token …> was created in a different Context

    because each ``gather`` coroutine runs in an isolated child asyncio context
    and AF tries to reset the token from the parent context.

    Calling this helper once — before building any agent — disables AF's
    built-in instrumentation for the lifetime of the process, avoiding the
    crash.  It is a no-op when AF is not installed or when the version is
    earlier than 1.6.0 (which has no default instrumentation to disable).

    To retain per-invocation observability after calling this function, attach
    :class:`~agent_gantry.integrations.agent_framework_middleware.GantryObservabilityMiddleware`
    to your agents; it records timing and success signals without the
    ContextVar clash.

    Returns:
        ``True`` if instrumentation was successfully disabled, ``False`` if it
        was already disabled, not applicable, or the AF version predates 1.6.0.

    Example::

        # Sequential workflows are unaffected — only needed for concurrent
        # asyncio.gather() / TaskGroup usage with AF >=1.6.0.
        from agent_gantry import disable_af_instrumentation
        disable_af_instrumentation()

        bridge = GantryToolBridge(gantry)
        agents = await asyncio.gather(
            bridge.as_agent(client, "query-a", name="A", instructions="…"),
            bridge.as_agent(client, "query-b", name="B", instructions="…"),
        )

    Source: https://pypi.org/pypi/agent-framework/json (1.6.0 release notes —
    "Enable instrumentation by default")
    """
    ver = _get_af_version()
    if ver is None or ver < (1, 6, 0):
        return False
    try:
        from agent_framework import telemetry as _af_telemetry  # type: ignore[import-not-found]

        _disable = getattr(_af_telemetry, "disable_instrumentation", None)
        if callable(_disable):
            _disable()
            logger.debug(
                "disable_af_instrumentation: called agent_framework.telemetry"
                ".disable_instrumentation() (AF %d.%d.%d)",
                *ver,
            )
            return True
        logger.warning(
            "disable_af_instrumentation: agent_framework.telemetry has no "
            "'disable_instrumentation' callable (AF %d.%d.%d). "
            "The ContextVar concurrency workaround could not be applied.",
            *ver,
        )
        return False
    except Exception:
        logger.debug(
            "disable_af_instrumentation: failed to import or call "
            "agent_framework.telemetry.disable_instrumentation",
            exc_info=True,
        )
        return False


# Private alias so GantryToolBridge.__init__ can call the module-level helper
# without the local parameter `disable_af_instrumentation` shadowing it.
_disable_af_instrumentation = disable_af_instrumentation


def _require_af_installed(caller: str) -> None:
    """Raise a descriptive ImportError when agent-framework is not installed."""
    try:
        import agent_framework as _  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            f"{caller}() requires the 'agent-framework' package. "
            "Install with: pip install 'agent-gantry[agent-frameworks]'"
        ) from exc


def _try_import_af_tool() -> Any | None:
    """Return ``agent_framework.tool`` if the package is installed, else None.

    The bridge degrades gracefully: when AF is not installed (unit tests,
    LangChain-only users, etc.) ``_build_callable_for_tool`` returns a bare
    typed Python callable, which AF 1.0 still auto-wraps into a FunctionTool
    when passed into ``Agent(tools=[...])``. When AF *is* installed we return
    a genuine ``FunctionTool`` so that ``approval_mode``, ``max_invocations``
    and other GA-only metadata flow through to the agent.
    """
    try:
        from agent_framework import tool as af_tool

        return af_tool
    except Exception:  # pragma: no cover - exercised in environments without AF
        return None


def _tool_approval_mode(tool_def: ToolDefinition) -> str | None:
    """Map Gantry ``ToolCapability`` set to AF ``approval_mode``.

    Destructive / sensitive capabilities elevate the tool to
    ``"always_require"`` so AF pauses for human approval before invocation.
    Everything else returns ``None`` (AF defaults to ``"never_require"``).
    """
    caps = set(tool_def.capabilities)
    if caps & _APPROVAL_REQUIRED_CAPS:
        return "always_require"
    return None


def _build_callable_for_tool(
    tool_def: ToolDefinition,
    gantry: AgentGantry,
    *,
    as_function_tool: bool | None = None,
) -> Any:
    """
    Build a Python callable wrapping a Gantry tool for Microsoft Agent Framework.

    The callable is created with proper type annotations (using ``Annotated``
    and Pydantic ``Field``) so that AF can auto-generate the correct function
    schema for the LLM. This avoids sending raw JSON schemas and lets AF's
    native function-tool infrastructure handle serialisation.

    When ``agent-framework`` is importable and ``as_function_tool`` is not
    ``False``, the callable is wrapped with ``@agent_framework.tool`` so AF
    receives a real ``FunctionTool`` with full GA metadata
    (``approval_mode`` derived from Gantry capabilities, description,
    name). When AF is not installed, a plain typed async callable is
    returned; AF 1.0 still auto-wraps those at agent construction time.

    Args:
        tool_def: The Gantry ToolDefinition to wrap.
        gantry: The AgentGantry instance for execution.
        as_function_tool: If ``True``, always wrap with ``@agent_framework.tool``
            (raises ImportError when AF isn't available). If ``False``, always
            return a bare callable. ``None`` (default) auto-detects.

    Returns:
        Either an ``agent_framework.FunctionTool`` or a bare async callable,
        both accepted by ``Agent(tools=[...])``.
    """
    tool_name = tool_def.name
    tool_desc = tool_def.description
    params_schema = tool_def.parameters_schema

    # Extract parameter info from the JSON Schema
    properties = params_schema.get("properties", {})
    required_params = set(params_schema.get("required", []))

    async def _execute(**kwargs: Any) -> str:
        # Surface any underlying exception as a structured error string. If
        # ``gantry.execute`` itself raises (e.g. a cross-event-loop
        # asyncio.Lock blowup, a connection error, an unexpected validation
        # error in the executor) the exception would otherwise propagate
        # into Agent Framework's tool runner, which replaces it with the
        # opaque ``"Error: Function failed."`` string when
        # ``include_detailed_errors`` is off (the default). That makes
        # debugging extremely hard for integrators. We convert any exception
        # into a JSON ``{"error": "..."}`` payload so the model — and the
        # human reading the trace — sees the real root cause.
        try:
            result = await gantry.execute(
                ToolCall(tool_name=tool_name, arguments=kwargs)
            )
        except Exception as exc:
            return json.dumps(
                {"error": f"{type(exc).__name__}: {exc}"}
            )
        if result.status.value == "success":
            val = result.result
            return val if isinstance(val, str) else json.dumps(val)
        # Failed ToolResult: prefer the recorded error/error_type, otherwise
        # fall back to a clear default — never the ambiguous original
        # "Tool execution failed" with no context.
        error_text = result.error or "tool execution failed (no error message)"
        if result.error_type and result.error_type not in error_text:
            error_text = f"{result.error_type}: {error_text}"
        return json.dumps({"error": error_text})

    # Build the wrapper with proper annotations for AF
    # AF inspects __name__, __doc__, and __annotations__ to generate schemas.
    # Two separate named functions avoid the mypy "conditional function variant"
    # error that arises when the same name is assigned in both if/else branches
    # with incompatible signatures.
    async def _wrapper_no_args() -> str:
        return await _execute()

    async def _wrapper_with_args(*args: Any, **kwargs: Any) -> str:
        param_names = list(properties.keys())
        if args:
            if len(args) > len(param_names):
                raise TypeError(
                    f"{tool_name}() takes at most {len(param_names)} positional "
                    f"arguments but {len(args)} were given"
                )
            for idx, value in enumerate(args):
                p_name = param_names[idx]
                if p_name not in kwargs:
                    kwargs[p_name] = value
        return await _execute(**kwargs)

    wrapper: Callable[..., Awaitable[str]]
    if len(properties) == 0:
        wrapper = _wrapper_no_args
    else:
        wrapper = _wrapper_with_args
        new_params = []
        for p_name, p_info in properties.items():
            p_desc = p_info.get("description", f"Parameter: {p_name}")
            p_type = _json_type_to_python(p_info.get("type", "string"))
            default = (
                inspect.Parameter.empty
                if p_name in required_params
                else p_info.get("default")
            )
            new_params.append(
                inspect.Parameter(
                    p_name,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    annotation=Annotated[p_type, Field(description=p_desc)],
                    default=default,
                )
            )
        wrapper.__signature__ = inspect.Signature(parameters=new_params)  # type: ignore[attr-defined]

    wrapper.__name__ = tool_name
    wrapper.__qualname__ = tool_name
    wrapper.__doc__ = tool_desc

    # Optionally upgrade to a real AF FunctionTool so approval_mode and the
    # rest of the GA metadata flows through. AF 1.0 also accepts bare
    # callables (it auto-wraps them at Agent(tools=...) time), so the
    # fallback path remains fully functional for environments without AF.
    if as_function_tool is False:
        return wrapper

    af_tool = _try_import_af_tool()
    if af_tool is None:
        if as_function_tool is True:
            raise ImportError(
                "as_function_tool=True requires the 'agent-framework' package. "
                "Install with: pip install 'agent-gantry[agent-frameworks]'"
            )
        return wrapper

    approval_mode = _tool_approval_mode(tool_def)
    decorated = af_tool(
        wrapper,
        name=tool_name,
        description=tool_desc,
        approval_mode=approval_mode,
    )
    return decorated


def _json_type_to_python(json_type: str) -> type:
    """Map JSON Schema type strings to Python types."""
    mapping: dict[str, type] = {
        "string": str,
        "integer": int,
        "number": float,
        "boolean": bool,
        "array": list,
        "object": dict,
    }
    return mapping.get(json_type, str)


def _cache_key(tool_def: ToolDefinition) -> str:
    """Build a namespace-qualified cache key for a tool definition."""
    return f"{tool_def.namespace}:{tool_def.name}"


class GantryToolBridge:
    """
    Bridge between Agent-Gantry and Microsoft Agent Framework.

    Retrieves semantically relevant tools from Gantry and wraps them as
    Python callables that AF agents can use directly. This is the primary
    integration point for production multi-agent systems where token savings
    from semantic routing are critical.

    The bridge supports two usage patterns:

    1. **Query-time retrieval** (recommended for token savings):
       Retrieve only the relevant tools for each query, minimising the
       tool definitions sent to the LLM.

       .. code-block:: python

           bridge = GantryToolBridge(gantry)
           tools = await bridge.get_tools("book a flight", limit=3)
           agent = client.as_agent(tools=tools, ...)

    2. **Pre-built tool set**:
       Build tools once from specific Gantry tool definitions for agents
       that need a fixed tool set.

       .. code-block:: python

           bridge = GantryToolBridge(gantry)
           tools = bridge.wrap_tools(my_tool_definitions)
           agent = client.as_agent(tools=tools, ...)

    Args:
        gantry: The AgentGantry instance providing tool retrieval and execution.
        score_threshold: Minimum relevance score for tool selection (default: 0.0).
            Accepts a ``float`` (absolute cosine cutoff), the string
            ``"relative:<frac>"`` (e.g. ``"relative:0.8"`` retains anything
            within 80% of the top score), or ``None`` (no filtering).
    """

    def __init__(
        self,
        gantry: AgentGantry,
        *,
        score_threshold: float | str | None = 0.0,
        as_function_tool: bool | None = None,
        disable_af_instrumentation: bool = False,
    ) -> None:
        """Initialise the bridge.

        Args:
            gantry: The AgentGantry instance providing tool retrieval and execution.
            score_threshold: Minimum relevance score for tool selection
                (default: ``0.0``; see class docstring for the relative mode).
            as_function_tool: Whether wrapped tools should be elevated to
                ``agent_framework.FunctionTool`` via the ``@tool`` decorator.
                ``None`` (default) = auto-detect (wrap if AF is importable);
                ``True`` = force wrapping (raise if AF is missing);
                ``False`` = always return bare callables. Defaults produce the
                most idiomatic AF behaviour without introducing a hard dep.
            disable_af_instrumentation: When ``True``, call
                :func:`disable_af_instrumentation` at construction time to
                suppress AF >=1.6.0's default ContextVar-based telemetry.
                Required for concurrent workflows (``asyncio.gather()`` /
                ``TaskGroup``) on AF 1.6.0 to prevent::

                    ValueError: <Token …> was created in a different Context

                Safe to pass when AF <1.6.0 is installed (becomes a no-op).
                Sequential single-agent flows are unaffected and do NOT need
                this flag. Defaults to ``False``.
        """
        # Validate the threshold eagerly so misconfiguration surfaces at
        # construction time rather than on the first retrieval round.
        _parse_threshold(score_threshold)
        self._gantry = gantry
        self._score_threshold = score_threshold
        self._as_function_tool = as_function_tool
        self._tool_cache: dict[str, Any] = {}
        if disable_af_instrumentation:
            _disable_af_instrumentation()

    async def _retrieve(
        self,
        query: str,
        *,
        limit: int = 5,
        score_threshold: float | str | None = None,
        **query_kwargs: Any,
    ) -> RetrievalResult:
        """Shared retrieval logic for get_tools and get_tools_with_scores.

        The relative threshold mode (``"relative:<frac>"``) cannot be
        applied at the vector-store level (the cutoff is data-dependent),
        so we issue the underlying gantry query without a score cutoff
        and apply the filter post-hoc in :meth:`_apply_threshold`.
        """
        from agent_gantry.schema.query import ConversationContext, ToolQuery

        threshold = (
            score_threshold if score_threshold is not None else self._score_threshold
        )
        mode, _numeric = _parse_threshold(threshold)

        # Separate context-level kwargs from query-level kwargs
        context_fields = set(ConversationContext.model_fields.keys()) - {"query"}
        context_kwargs = {k: v for k, v in query_kwargs.items() if k in context_fields}
        tool_query_fields = set(ToolQuery.model_fields.keys()) - {"context", "limit", "score_threshold"}
        tq_kwargs = {k: v for k, v in query_kwargs.items() if k in tool_query_fields}

        # Always push 0.0 to the underlying store and apply the threshold
        # post-hoc in :meth:`_apply_threshold`. This costs a few extra
        # candidates from the vector store (the router already over-fetches
        # query.limit * 4 anyway) but lets the decision record list the
        # tools that were dropped — without that, "score_threshold filtered
        # every candidate" warnings are blind and the relative-threshold
        # mode is impossible to implement.
        #
        # Relative mode also needs the full candidate pool so the cutoff
        # is computed correctly, so we over-fetch by 4×. Clamp to
        # ``ToolQuery.limit``'s upper bound (50) — without the clamp,
        # callers using ``limit >= 13`` with a relative threshold hit a
        # Pydantic validation error on the ``ToolQuery`` construction
        # below.
        if mode == "relative":
            store_limit = min(_TOOL_QUERY_MAX_LIMIT, max(limit * 4, limit))
        else:
            store_limit = limit

        return await self._gantry.retrieve(
            ToolQuery(
                context=ConversationContext(query=query, **context_kwargs),
                limit=store_limit,
                score_threshold=0.0,
                **tq_kwargs,
            )
        )

    def _apply_threshold(
        self,
        scored: list[Any],
        *,
        threshold: float | str | None,
        limit: int,
    ) -> tuple[list[Any], RetrievalDecision]:
        """Filter scored tools by threshold and return a decision record.

        ``scored`` is the list of ``ScoredTool`` from
        :class:`RetrievalResult.tools`. The returned tuple is
        ``(kept_top_K, decision)`` where ``kept_top_K`` honours ``limit``.
        """
        effective = threshold if threshold is not None else self._score_threshold
        mode, numeric = _parse_threshold(effective)

        if not scored:
            return [], RetrievalDecision(
                threshold=effective,
                threshold_mode=mode
                if mode == "absolute"
                else f"relative:{numeric}",
                effective_threshold=None,
            )

        if mode == "relative" and numeric is not None:
            top_score = max(st.semantic_score for st in scored)
            # ``ScoredTool.semantic_score`` is Pydantic-clamped to ``[0, 1]``
            # so this is the common path. Guard against the degenerate
            # ``top_score <= 0`` case anyway — multiplying a negative top
            # by a positive fraction produces a cutoff *above* the top
            # score and silently drops every candidate including the
            # best one. When that happens, fall back to ``0.0`` so the
            # ranking is preserved and the (zero-scoring) candidates are
            # at least visible to the caller via the decision record.
            cutoff = top_score * numeric if top_score > 0 else 0.0
        else:
            cutoff = numeric or 0.0

        candidates: list[RetrievalCandidate] = []
        kept: list[Any] = []
        for st in scored:
            score = float(st.semantic_score)
            keep = score >= cutoff
            candidates.append(
                RetrievalCandidate(
                    name=st.tool.name,
                    qualified_name=f"{st.tool.namespace}.{st.tool.name}",
                    score=score,
                    kept=keep,
                )
            )
            if keep:
                kept.append(st)

        kept_top = kept[:limit]
        decision = RetrievalDecision(
            candidates=candidates,
            threshold=effective,
            threshold_mode=mode
            if mode == "absolute"
            else f"relative:{numeric}",
            effective_threshold=cutoff,
        )
        return kept_top, decision

    def _get_or_build(
        self,
        tool_def: ToolDefinition,
        cache: bool,
    ) -> Any:
        """Look up a cached wrapper or build a new one."""
        key = _cache_key(tool_def)
        if cache and key in self._tool_cache:
            return self._tool_cache[key]
        wrapper = _build_callable_for_tool(
            tool_def,
            self._gantry,
            as_function_tool=self._as_function_tool,
        )
        if cache:
            self._tool_cache[key] = wrapper
        return wrapper

    async def get_tools(
        self,
        query: str,
        *,
        limit: int = 5,
        score_threshold: float | str | None = None,
        cache: bool = True,
        **query_kwargs: Any,
    ) -> list[Any]:
        """
        Retrieve semantically relevant tools as AF-compatible callables.

        This is the primary method for dynamic tool selection. It queries
        Gantry's semantic router, selects the top-k tools, and wraps each
        as a Python async callable with proper type annotations that
        Microsoft Agent Framework can introspect.

        Args:
            query: The user query or task description to match tools against.
            limit: Maximum number of tools to return (default: 5).
            score_threshold: Override the bridge's default score threshold.
                Accepts a ``float`` (absolute cosine cutoff), the string
                ``"relative:<frac>"`` (e.g. ``"relative:0.8"`` keeps
                anything within 80% of the top score), or ``None`` (use
                the bridge default).
            cache: Whether to reuse previously wrapped callables for the same
                   tool (avoids re-creating wrappers). Default: True.
            **query_kwargs: Additional keyword arguments passed through to
                ``ToolQuery`` (e.g. ``namespaces``, ``required_capabilities``,
                ``sources``, ``exclude_deprecated``, ``enable_reranking``) and
                to ``ConversationContext`` (e.g. ``user_capabilities``,
                ``conversation_summary``, ``recent_messages``).

        Returns:
            List of async callables suitable for AF agent ``tools=[...]``.
        """
        tools, _ = await self.get_tools_with_decision(
            query,
            limit=limit,
            score_threshold=score_threshold,
            cache=cache,
            **query_kwargs,
        )
        return tools

    async def get_tools_with_decision(
        self,
        query: str,
        *,
        limit: int = 5,
        score_threshold: float | str | None = None,
        cache: bool = True,
        **query_kwargs: Any,
    ) -> tuple[list[Any], RetrievalDecision]:
        """Same as :meth:`get_tools` but also returns the decision record.

        Returns the ranked candidate list (passed and dropped by the
        threshold filter), the final injected list, and the effective
        threshold — enough to make the routing path self-diagnosable
        from outside the library. See :class:`RetrievalDecision`.

        Wraps the retrieval in a ``gantry.bridge_retrieval`` telemetry
        span carrying the candidate list and scores as attributes so
        OpenTelemetry consumers see the ranked decision in their
        tracing backend.
        """
        from agent_gantry.utils.async_utils import AsyncNoopContext

        telemetry = getattr(self._gantry, "_telemetry", None)
        span_attrs: dict[str, Any] = {
            "query": query,
            "limit": limit,
            "score_threshold": (
                score_threshold
                if score_threshold is not None
                else self._score_threshold
            ),
        }
        span_cm = (
            telemetry.span("gantry.bridge_retrieval", span_attrs)
            if telemetry
            else AsyncNoopContext()
        )

        async with span_cm:
            result = await self._retrieve(
                query, limit=limit, score_threshold=score_threshold, **query_kwargs
            )

            # Threshold filtering happens post-hoc so the decision record can
            # see both kept and dropped candidates regardless of whether the
            # threshold was relative or absolute.
            kept_top, decision = self._apply_threshold(
                list(result.tools), threshold=score_threshold, limit=limit
            )
            decision.query = query
            tools = [self._get_or_build(st.tool, cache) for st in kept_top]
            decision.injected = [st.tool.name for st in kept_top]

            # Enrich the span with the structured decision. Telemetry
            # adapters in this codebase store the same dict instance they
            # were handed at span open, so mutating it now persists the
            # candidates list onto the recorded span.
            try:
                span_attrs.update(decision.as_span_attributes())
            except Exception:  # pragma: no cover - defensive
                pass

            # If threshold filtered everything out, surface a WARNING so
            # users don't see "empty surface" without context. Always log
            # the configured threshold (and the resolved cutoff for the
            # relative mode) so the message explains *which* cutoff
            # dropped the candidates, not just the mode name.
            if result.tools and not kept_top:
                if decision.threshold_mode.startswith("relative") and (
                    decision.effective_threshold is not None
                ):
                    threshold_repr = (
                        f"{decision.threshold} "
                        f"(cutoff={decision.effective_threshold:.3f})"
                    )
                else:
                    threshold_repr = repr(decision.threshold)
                logger.warning(
                    "GantryToolBridge: score_threshold %s filtered out all %d "
                    "candidates for query %r. Top scores: %s",
                    threshold_repr,
                    len(result.tools),
                    query[:80],
                    ", ".join(
                        f"{c.name}:{c.score:.3f}" for c in decision.candidates[:5]
                    ),
                )

            logger.debug(
                "GantryToolBridge: selected %d/%d tools for query '%s'",
                len(tools),
                result.candidate_count,
                query[:50],
            )
            return tools, decision

    def wrap_tools(
        self,
        tool_definitions: list[ToolDefinition],
        *,
        cache: bool = True,
    ) -> list[Any]:
        """
        Wrap specific Gantry tool definitions as AF-compatible callables.

        Use this when you already have the tool definitions and want to
        create the callable wrappers without going through semantic retrieval.

        Args:
            tool_definitions: List of ToolDefinition objects to wrap.
            cache: Whether to cache/reuse wrappers (default: True).

        Returns:
            List of async callables suitable for AF agent ``tools=[...]``.
        """
        return [self._get_or_build(td, cache) for td in tool_definitions]

    def wrap_single(self, tool_def: ToolDefinition) -> Any:
        """
        Wrap a single Gantry tool definition as an AF-compatible callable.

        Args:
            tool_def: The ToolDefinition to wrap.

        Returns:
            An async callable suitable for AF agent ``tools=[...]``.
        """
        return self._get_or_build(tool_def, cache=True)

    def clear_cache(self) -> None:
        """Clear the cached tool wrappers."""
        self._tool_cache.clear()

    async def get_tools_with_scores(
        self,
        query: str,
        *,
        limit: int = 5,
        score_threshold: float | str | None = None,
        cache: bool = True,
        **query_kwargs: Any,
    ) -> list[tuple[Any, float]]:
        """
        Retrieve tools with their relevance scores for observability.

        Same as ``get_tools`` but also returns the semantic relevance score
        for each tool, useful for debugging and monitoring token savings.

        Args:
            query: The user query to match tools against.
            limit: Maximum number of tools to return.
            score_threshold: Override the bridge's default score threshold.
            cache: Whether to reuse previously wrapped callables for the same
                   tool (avoids re-creating wrappers). Default: True.
            **query_kwargs: Additional keyword arguments passed through to
                ``ToolQuery`` and ``ConversationContext``.

        Returns:
            List of (callable, score) tuples.
        """
        result = await self._retrieve(
            query, limit=limit, score_threshold=score_threshold, **query_kwargs
        )

        kept_top, _ = self._apply_threshold(
            list(result.tools), threshold=score_threshold, limit=limit
        )

        return [
            (self._get_or_build(st.tool, cache), st.final_score)
            for st in kept_top
        ]

    # ------------------------------------------------------------------
    # Agent construction helpers
    # ------------------------------------------------------------------

    async def build_agent(
        self,
        client: Any,
        query: str,
        *,
        name: str,
        instructions: str,
        limit: int = 5,
        score_threshold: float | None = None,
        middleware: Any = None,
        cache: bool = True,
        extra_tools: list[Any] | None = None,
        **query_kwargs: Any,
    ) -> Any:
        """Retrieve relevant tools and construct an AF ``Agent`` in one call.

        This is the idiomatic one-liner for single-agent flows where the
        tool set is determined by a single query (for example, the first
        user turn). For multi-turn conversations, use ``get_tools`` and
        keep the resulting agent across turns, or re-run this helper per
        turn if you want tools to adapt to the latest user message.

        Args:
            client: Any AF chat client (e.g. ``OpenAIChatClient``,
                ``AzureOpenAIChatClient``, ``AnthropicChatClient``,
                ``GeminiChatClient`` — added in agent-framework 1.1.0).
            query: The user query whose tools will be retrieved.
            name: Name to give the constructed agent.
            instructions: System instructions for the agent.
            limit: Top-K tools to retrieve from Gantry.
            score_threshold: Override the bridge-level threshold.
            middleware: Optional AF middleware sequence. Accepts
                ``GantryApprovalMiddleware`` and any AF-native middleware.
            cache: Whether to reuse cached wrappers.
            extra_tools: Additional static tools (AF ``FunctionTool``,
                bare callables, …) to append after the Gantry-selected tools.
            **query_kwargs: Forwarded to ``ToolQuery`` / ``ConversationContext``.

        Returns:
            An ``agent_framework.Agent`` instance.
        """
        _require_af_installed("build_agent")
        from agent_framework import Agent

        tools = await self.get_tools(
            query,
            limit=limit,
            score_threshold=score_threshold,
            cache=cache,
            **query_kwargs,
        )
        if extra_tools:
            tools = tools + list(extra_tools)

        agent_kwargs: dict[str, Any] = {
            "name": name,
            "tools": tools,
        }
        if middleware is not None:
            agent_kwargs["middleware"] = middleware
        return Agent(client, instructions, **agent_kwargs)

    async def as_agent(
        self,
        client: Any,
        query: str,
        *,
        name: str,
        instructions: str,
        limit: int = 5,
        score_threshold: float | None = None,
        middleware: Any = None,
        cache: bool = True,
        extra_tools: list[Any] | None = None,
        query_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Retrieve relevant tools and construct a bare AF ``Agent(client, ...)`` directly.

        Unlike :meth:`build_agent` which uses ``client.as_agent()``, this
        constructs ``Agent`` via its constructor so the result is a first-class
        ``Agent`` object suitable for direct use with ``WorkflowBuilder``,
        ``WorkflowAgent``, and other multi-agent orchestration patterns.

        .. code-block:: python

            from agent_framework.openai import OpenAIChatClient
            from agent_framework.orchestrations import HandoffBuilder

            client = OpenAIChatClient()
            bridge = GantryToolBridge(gantry)

            triage  = await bridge.as_agent(client, "triage",   name="Triage",   instructions="Route the request.")
            billing = await bridge.as_agent(client, "billing",  name="Billing",  instructions="Handle billing questions.")
            support = await bridge.as_agent(client, "support",  name="Support",  instructions="Handle support tickets.")

            workflow = (
                HandoffBuilder(name="CustomerService")
                .participants([triage, billing, support])
                .with_start_agent(triage)
                .add_handoff(source=triage, targets=[billing], description="billing enquiries")
                .add_handoff(source=triage, targets=[support], description="support tickets")
                .build()
            )
            result = await triage.run("I need help with my invoice")

        Args:
            client: Any AF chat client (``OpenAIChatClient``, ``AzureOpenAIChatClient``,
                ``AnthropicChatClient``, ``GeminiChatClient`` — added in
                agent-framework 1.1.0, …).
            query: The user query used to semantically select tools.
            name: Name for the agent.
            instructions: System instructions for the agent.
            limit: Top-K tools to retrieve from Gantry.
            score_threshold: Override the bridge-level threshold.
            middleware: Optional AF middleware sequence.
            cache: Whether to reuse cached tool wrappers.
            extra_tools: Additional static tools to append after Gantry-selected tools.
            query_kwargs: Extra keyword arguments forwarded to :meth:`get_tools`
                (e.g. ``conversation_context``, ``namespace``).
            **kwargs: Additional keyword arguments forwarded to ``Agent()``.

        Returns:
            A bare ``agent_framework.Agent`` instance.
        """
        _require_af_installed("as_agent")
        from agent_framework import Agent

        tools = await self.get_tools(
            query,
            limit=limit,
            score_threshold=score_threshold,
            cache=cache,
            **(query_kwargs or {}),
        )
        if extra_tools:
            tools = tools + list(extra_tools)

        agent_kwargs: dict[str, Any] = {
            "name": name,
            "tools": tools,
            **kwargs,
        }
        if middleware is not None:
            agent_kwargs["middleware"] = middleware

        return Agent(client, instructions, **agent_kwargs)

    async def build_sequential_workflow(
        self,
        agent_specs: list[dict[str, Any]],
        *,
        cache: bool = True,
    ) -> Any:
        """Build a sequential multi-agent workflow from Gantry-equipped agent specs.

        Constructs each agent via :meth:`as_agent`, then wires them together
        with ``SequentialBuilder`` so they execute one after another. Each agent
        receives the previous agent's output as its input.

        .. code-block:: python

            from agent_framework.openai import OpenAIChatClient

            client = OpenAIChatClient()
            bridge = GantryToolBridge(gantry)

            workflow = await bridge.build_sequential_workflow(
                agent_specs=[
                    dict(client=client, query="gather info",   name="Gather",    instructions="Collect user details."),
                    dict(client=client, query="resolve issue", name="Resolver",  instructions="Resolve the issue."),
                    dict(client=client, query="send summary",  name="Summarise", instructions="Summarise the outcome."),
                ],
            )

        For handoff-style routing (one agent decides which specialist to call
        next), use ``agent_framework.orchestrations.HandoffBuilder`` directly
        after building agents with :meth:`as_agent`.

        Args:
            agent_specs: List of dicts passed as keyword arguments to :meth:`as_agent`.
                Required keys: ``client``, ``query``, ``name``, ``instructions``.
                Optional keys: ``limit``, ``score_threshold``, ``middleware``,
                ``extra_tools``, plus any extra ``Agent()`` kwargs.
            cache: Whether to reuse cached tool wrappers.

        Returns:
            The workflow object produced by ``SequentialBuilder.build()``.
        """
        _require_af_installed("build_sequential_workflow")
        from agent_framework.orchestrations import SequentialBuilder

        ordered: list[Any] = []
        for spec in agent_specs:
            spec = dict(spec)
            agent = await self.as_agent(cache=cache, **spec)
            ordered.append(agent)

        if not ordered:
            raise ValueError("agent_specs must contain at least one agent.")

        return SequentialBuilder(participants=ordered).build()

    async def build_handoff_workflow(
        self,
        agent_specs: list[dict[str, Any]],
        *,
        handoffs: list[tuple[str, list[str], str]] | None = None,
        start_agent_name: str | None = None,
        workflow_name: str = "gantry_workflow",
        cache: bool = True,
    ) -> Any:
        """Build a handoff-style multi-agent workflow from Gantry-equipped agent specs.

        Constructs each agent via :meth:`as_agent`, then wires them together
        with ``HandoffBuilder``. The start agent receives the initial user
        message and decides which specialist to hand off to.

        .. code-block:: python

            from agent_framework.openai import OpenAIChatClient

            client = OpenAIChatClient()
            bridge = GantryToolBridge(gantry)

            workflow = await bridge.build_handoff_workflow(
                agent_specs=[
                    dict(client=client, query="triage customer",     name="Triage",   instructions="Route the customer."),
                    dict(client=client, query="billing invoices",     name="Billing",  instructions="Handle billing."),
                    dict(client=client, query="support tickets bugs", name="Support",  instructions="Handle support."),
                ],
                handoffs=[
                    ("Triage", ["Billing"], "billing enquiries"),
                    ("Triage", ["Support"], "support tickets"),
                ],
                start_agent_name="Triage",
                workflow_name="CustomerService",
            )

        Args:
            agent_specs: List of dicts passed as keyword arguments to :meth:`as_agent`.
                Required keys: ``client``, ``query``, ``name``, ``instructions``.
            handoffs: List of ``(source_name, [target_names], description)`` tuples
                describing the handoff edges between agents.
            start_agent_name: Name of the first agent to receive the user message.
                Defaults to the first entry in ``agent_specs``.
            workflow_name: Name for the handoff workflow (default: ``"gantry_workflow"``).
            cache: Whether to reuse cached tool wrappers.

        Returns:
            The workflow object produced by ``HandoffBuilder.build()``.
        """
        _require_af_installed("build_handoff_workflow")
        from agent_framework.orchestrations import HandoffBuilder

        built: dict[str, Any] = {}
        ordered: list[Any] = []
        for spec in agent_specs:
            spec = dict(spec)
            agent_name: str = spec["name"]
            agent = await self.as_agent(cache=cache, **spec)
            built[agent_name] = agent
            ordered.append(agent)

        if not ordered:
            raise ValueError("agent_specs must contain at least one agent.")

        start_name = start_agent_name or agent_specs[0]["name"]
        if start_name not in built:
            raise ValueError(
                f"start_agent_name '{start_name}' not found in agent_specs. "
                f"Known names: {sorted(built)}"
            )

        hb = (
            HandoffBuilder(name=workflow_name)
            .participants(ordered)
            .with_start_agent(built[start_name])
        )

        for source_name, target_names, description in (handoffs or []):
            if source_name not in built:
                raise ValueError(
                    f"Handoff source '{source_name}' not found. Known: {sorted(built)}"
                )
            unknown_targets = [t for t in target_names if t not in built]
            if unknown_targets:
                raise ValueError(
                    f"Handoff target(s) {unknown_targets} not found. Known: {sorted(built)}"
                )
            hb = hb.add_handoff(
                source=built[source_name],
                targets=[built[t] for t in target_names],
                description=description,
            )

        return hb.build()

    async def build_workflow(
        self,
        agent_specs: list[dict[str, Any]],
        *,
        edges: list[tuple[str, str] | tuple[str, str, Any]] | None = None,
        chain: bool = False,
        workflow_name: str | None = None,
        cache: bool = True,
    ) -> Any:
        """Build a multi-agent ``WorkflowAgent`` from Gantry-equipped agent specs.

        Constructs each agent via :meth:`as_agent`, wraps them in
        ``AgentExecutor`` nodes (required by ``WorkflowBuilder``), then wires
        them into a ``WorkflowAgent``.

        For sequential pipelines pass ``chain=True``.  For routing with
        hand-off semantics prefer :meth:`build_handoff_workflow`, which uses
        ``HandoffBuilder`` and avoids the ``AgentExecutor`` indirection.

        .. code-block:: python

            from agent_framework.openai import OpenAIChatClient

            client = OpenAIChatClient()
            bridge = GantryToolBridge(gantry)

            # Linear chain
            wa = await bridge.build_workflow(
                agent_specs=[
                    dict(client=client, query="gather info",   name="Gather",   instructions="Collect user details."),
                    dict(client=client, query="resolve issue", name="Resolver", instructions="Resolve the issue."),
                ],
                chain=True,
            )

            # Fan-out with explicit edges
            wa = await bridge.build_workflow(
                agent_specs=[
                    dict(client=client, query="triage",   name="Triage",   instructions="Route."),
                    dict(client=client, query="billing",  name="Billing",  instructions="Handle billing."),
                    dict(client=client, query="support",  name="Support",  instructions="Handle support."),
                ],
                edges=[("Triage", "Billing"), ("Triage", "Support")],
            )

        Args:
            agent_specs: List of dicts passed as keyword arguments to :meth:`as_agent`.
                Required keys: ``client``, ``query``, ``name``, ``instructions``.
            edges: List of ``(source_name, target_name)`` or
                ``(source_name, target_name, condition)`` tuples. The optional
                third element is forwarded to ``WorkflowBuilder.add_edge()`` as
                ``condition``, enabling logic-based routing. Ignored when
                ``chain=True``.
            chain: If ``True``, wire all agents as a linear chain. Overrides
                ``edges``.
            workflow_name: Optional name forwarded to ``WorkflowAgent``.
            cache: Whether to reuse cached tool wrappers.

        Returns:
            A ``WorkflowAgent`` wrapping the constructed ``Workflow``.
        """
        _require_af_installed("build_workflow")
        from agent_framework import AgentExecutor, WorkflowAgent, WorkflowBuilder

        built_agents: dict[str, Any] = {}
        ordered_agents: list[Any] = []
        for spec in agent_specs:
            spec = dict(spec)
            agent_name: str = spec["name"]
            agent = await self.as_agent(cache=cache, **spec)
            built_agents[agent_name] = agent
            ordered_agents.append(agent)

        if not ordered_agents:
            raise ValueError("agent_specs must contain at least one agent.")

        # WorkflowBuilder operates on AgentExecutor nodes, not bare Agent objects.
        built_executors: dict[str, Any] = {
            name: AgentExecutor(agent, id=name)
            for name, agent in built_agents.items()
        }
        ordered_executors = [built_executors[spec["name"]] for spec in agent_specs]

        if not chain and edges:
            unknown = {
                name
                for edge in edges
                for name in (edge[0], edge[1])
                if name not in built_executors
            }
            if unknown:
                raise ValueError(
                    f"build_workflow() edges reference unknown agent name(s): "
                    f"{sorted(unknown)}. Known names: {sorted(built_executors)}"
                )

        builder = WorkflowBuilder(start_executor=ordered_executors[0])

        if chain:
            for i in range(len(ordered_executors) - 1):
                builder.add_edge(ordered_executors[i], ordered_executors[i + 1])
        else:
            for edge in (edges or []):
                source_name, target_name = edge[0], edge[1]
                condition = edge[2] if len(edge) > 2 else None  # type: ignore[misc]
                builder.add_edge(
                    built_executors[source_name],
                    built_executors[target_name],
                    condition=condition,
                )

        workflow = builder.build()
        wa_kwargs: dict[str, Any] = {}
        if workflow_name is not None:
            wa_kwargs["name"] = workflow_name
        return WorkflowAgent(workflow, **wa_kwargs)

    def as_tool_list(self, tool_defs: list[ToolDefinition]) -> list[Any]:
        """Alias for :meth:`wrap_tools` with a name that reads naturally in
        orchestration code where the returned list will be spread across
        multiple agents (e.g. ``Agent(tools=bridge.as_tool_list([...]))``).
        """
        return self.wrap_tools(tool_defs)
