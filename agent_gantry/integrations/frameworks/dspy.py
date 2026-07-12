"""DSPy native tool adapter for Agent-Gantry.

Selects a relevant slice of Gantry tools and wraps each as a ``dspy.Tool`` —
the tool object DSPy's agentic module, ``dspy.ReAct``, introspects (name /
description / JSON-Schema args) and invokes. The ``dspy`` import is lazy so
``import agent_gantry`` never requires DSPy to be installed.

Public entry point: :class:`DSPyAdapter`.

Why a *synchronous* wrapped function, not :meth:`ToolSpec.callable_for_signature`
-----------------------------------------------------------------------------
``dspy.Tool`` supports both a synchronous ``__call__`` (the path
``dspy.ReAct.forward`` uses — i.e. what runs when you call ``react(...)``, the
usage shown in ``dspy.ReAct``'s own docstring and ``Module.__call__``) and an
async ``acall`` (the path ``ReAct.aforward`` uses — ``await
react.acall(...)``). DSPy's own bundled MCP bridge
(``dspy.utils.mcp.convert_mcp_tool``) wraps MCP's inherently-async calls with
an *async* function, but that only works because MCP users are expected to
always invoke through ``await react.acall(...)``. Verified against the
installed dspy 3.2.1: calling the *same* async-wrapped tool through the
default, synchronous ``react(...)`` entry point does not raise all the way
out — ``ReAct.forward`` catches the resulting ``ValueError`` (DSPy's message:
"you are calling ``__call__`` on an async tool, please use ``acall`` instead
or enable async-to-sync conversion with
``dspy.configure(allow_tool_async_sync_conversion=True)``") and records it as
a plain "Execution error" *observation* string, then carries on — so the
failure is silent and the agent limps along on a broken trajectory rather
than raising loudly. Since ``react(question=...)`` (not
``await react.acall(...)``) is the call convention most users reach for
first, this adapter instead wraps every tool with a **synchronous** function
backed by :meth:`ToolSpec.invoke` (the loop-safe sync bridge in ``base.py``,
already used this way for CrewAI/Agno/Haystack/Smolagents). A sync wrapper
returns a plain value either way, so it works correctly under both
``dspy.Tool.__call__`` (``react(...)``) and ``dspy.Tool.acall`` (``await
react.acall(...)`` — a non-coroutine result is simply returned, no await
needed) with zero DSPy configuration required. The real ``__signature__`` is
still borrowed from :meth:`ToolSpec.callable_for_signature` (mirroring
``agno.py``'s ``_spec_to_agno``) so introspection sees the actual parameters;
only the function *body* differs (sync bridge vs. ``await ainvoke``).

Gantry's own name/description/JSON-Schema parameters are passed straight
through via ``dspy.adapters.types.tool.convert_input_schema_to_tool_args`` —
the same helper DSPy's own MCP and LangChain tool bridges use to convert an
externally-owned JSON-Schema into ``dspy.Tool``'s ``args``/``arg_types``/
``arg_desc`` — so nothing is lossily re-inferred from the wrapper function's
bare ``**kwargs`` signature.

Per-call dynamic tier, not per-turn
------------------------------------
``dspy.ReAct.__init__`` freezes ``self.tools`` into a plain dict and bakes
each tool's name/description into the ``react_signature`` instructions string
once, at construction time — there is no public API to swap either mid-run.
``dspy.utils.callback.BaseCallback`` does expose ``on_tool_start``/
``on_tool_end`` and ``on_module_start``/``on_module_end`` hooks, but they fire
for observability *around* an already-selected tool call or an
already-completed top-level ``forward()``/``aforward()`` call — never
*before* the model decides which tool to call next. That is unlike Strands'
``BeforeModelCallEvent`` or Google ADK's ``before_model_callback``, which fire
immediately before each model call and are read by the framework's own
tool-selection step (see ``strands_live.py`` / ``google_adk_live.py``). DSPy
ships no equivalent, so there is no genuine per-turn re-selection hook here.
:meth:`DSPyAdapter.agent_builder` therefore follows the same
*per-top-level-call* tier as ``CrewAIAdapter.agent_builder`` /
``AgnoAdapter.agent_builder`` / ``SmolagentsAdapter.agent_builder`` (see
``live_wrappers.py``): it rebuilds a fresh ``dspy.ReAct`` for each call,
re-selecting tools for that call's query. Between calls the tool surface
tracks the new query; *within* a single ``ReAct`` run (i.e. across its
internal reasoning iterations) it stays fixed — the deepest DSPy permits.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from agent_gantry.integrations.frameworks.base import (
    DEFAULT_TOOL_LIMIT,
    BaseFrameworkAdapter,
    GantryToolset,
    ToolSpec,
)

if TYPE_CHECKING:
    from agent_gantry.core.gantry import AgentGantry

_INSTALL_HINT = "DSPy support requires `dspy`. Install it with `pip install dspy`."


def _spec_to_dspy(spec: ToolSpec) -> Any:
    """Wrap a :class:`ToolSpec` as a ``dspy.Tool``.

    The ``dspy`` import happens here, lazily, so callers without DSPy
    installed only hit the error when they actually export a tool. Gantry's
    own name/description/JSON-Schema parameters are passed straight through
    via ``convert_input_schema_to_tool_args`` (DSPy's own schema-to-``Tool``
    bridge, also used by its MCP/LangChain tool converters) so nothing is
    re-inferred from the wrapper function. See the module docstring for why
    the wrapped function is a *synchronous* bridge (:meth:`ToolSpec.invoke`)
    rather than :meth:`ToolSpec.callable_for_signature`'s async wrapper.

    Raises:
        ImportError: If ``dspy`` is not installed.
    """
    try:
        # Both names are imported from the same submodule -- not the top-level
        # `dspy.Tool` re-export -- mirroring dspy.utils.mcp.convert_mcp_tool's
        # own import line exactly (DSPy's own MCP tool bridge uses this same
        # pair from this same module).
        from dspy.adapters.types.tool import Tool as DSPyTool
        from dspy.adapters.types.tool import convert_input_schema_to_tool_args
    except ImportError as exc:  # pragma: no cover - exercised via stub
        raise ImportError(_INSTALL_HINT) from exc

    args, arg_types, arg_desc = convert_input_schema_to_tool_args(spec.parameters)

    # Borrow the real __signature__/__annotations__ from the async wrapper
    # (same JSON-Schema-derived signature every other adapter uses) but call
    # through the sync bridge instead of awaiting -- see module docstring.
    async_fn = spec.callable_for_signature()

    def _fn(**kwargs: Any) -> Any:
        return spec.invoke(**kwargs)

    _fn.__name__ = spec.name
    _fn.__doc__ = spec.description
    _fn.__signature__ = async_fn.__signature__  # type: ignore[attr-defined]
    _fn.__annotations__ = dict(getattr(async_fn, "__annotations__", {}))

    return DSPyTool(
        func=_fn,
        name=spec.name,
        desc=spec.description,
        args=args,
        arg_types=arg_types,
        arg_desc=arg_desc,
    )


async def _for_dspy(
    gantry: AgentGantry,
    query: str,
    *,
    limit: int = DEFAULT_TOOL_LIMIT,
    **select_kwargs: Any,
) -> list[Any]:
    """Select tools for ``query`` and return them as ``dspy.Tool``s."""
    specs = await GantryToolset(gantry).select(query, limit=limit, **select_kwargs)
    return [_spec_to_dspy(s) for s in specs]


class GantryLiveDSPyReAct:
    """Rebuild a fresh ``dspy.ReAct`` per call, tools re-selected by Gantry.

    ``dspy.ReAct`` fixes its tool list at construction (see the module
    docstring for why), so this builder constructs a *new* ``dspy.ReAct`` for
    every call via :meth:`build`, each time wiring in the tools Gantry selects
    for that call's query. The task ``signature`` (and ``max_iters``/any extra
    ``dspy.ReAct`` kwargs) are configured once on the constructor and reused
    for every rebuild.

    Obtain one via ``DSPyAdapter(gantry).agent_builder(signature, ...)``.

    Args:
        gantry: The :class:`~agent_gantry.core.gantry.AgentGantry` to select from.
        signature: DSPy task signature — an ``"input -> output"`` string or a
            ``dspy.Signature`` subclass, forwarded to ``dspy.ReAct``.
        max_iters: Max reasoning/tool-call iterations per ``dspy.ReAct`` run.
        limit: Max tools to surface per call. Defaults to ``DEFAULT_TOOL_LIMIT``.
        score_threshold: Minimum semantic relevance score. Defaults to ``0.0``.
        **react_kwargs: Extra kwargs forwarded to ``dspy.ReAct``.
    """

    def __init__(
        self,
        gantry: AgentGantry,
        signature: Any,
        *,
        max_iters: int = 20,
        limit: int = DEFAULT_TOOL_LIMIT,
        score_threshold: float = 0.0,
        namespaces: list[str] | None = None,
        required: list[str] | None = None,
        always_include: list[str] | None = None,
        **react_kwargs: Any,
    ) -> None:
        self._gantry = gantry
        self._signature = signature
        self._max_iters = max_iters
        self._limit = limit
        self._score_threshold = score_threshold
        self._namespaces = namespaces
        self._required = required
        self._always_include = always_include
        self._react_kwargs = react_kwargs

    async def select_tools(self, query: str) -> list[Any]:
        """Re-select this call's ``dspy.Tool`` list for ``query``."""
        return await _for_dspy(
            self._gantry,
            query,
            limit=self._limit,
            score_threshold=self._score_threshold,
            namespaces=self._namespaces,
            required=self._required,
            always_include=self._always_include,
        )

    async def build(self, query: str) -> Any:
        """Build a fresh ``dspy.ReAct`` whose tools are selected for ``query``.

        Raises:
            ImportError: If ``dspy`` is not installed.
        """
        try:
            from dspy import ReAct
        except ImportError as exc:  # pragma: no cover - exercised via importorskip
            raise ImportError(_INSTALL_HINT) from exc

        tools = await self.select_tools(query)
        return ReAct(self._signature, tools=tools, max_iters=self._max_iters, **self._react_kwargs)


class DSPyAdapter(BaseFrameworkAdapter):
    """Route Gantry-selected tools into DSPy.

    Static slice (``dspy.Tool`` objects) plus a per-call live builder
    (``dspy.ReAct`` fixes its tools at construction with no runtime
    re-selection hook, so the live path rebuilds a fresh ``dspy.ReAct`` per
    call — see the module docstring for why). Every call routes through
    ``gantry.execute`` so retries, timeouts, circuit breakers, and the
    security policy all apply::

        from agent_gantry import AgentGantry
        from agent_gantry.dspy import DSPyAdapter
        import dspy

        dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))

        gantry = AgentGantry()
        # ... register tools, await gantry.sync() ...

        tools = await DSPyAdapter(gantry).select("what's the weather in Tokyo?", limit=3)
        react = dspy.ReAct("question -> answer", tools=tools)
        pred = react(question="what's the weather in Tokyo?")
    """

    live_tier = "per-call"

    @staticmethod
    def convert(spec: ToolSpec) -> Any:
        """Wrap a single :class:`ToolSpec` as a ``dspy.Tool``."""
        return _spec_to_dspy(spec)

    def live(
        self,
        *,
        limit: int | None = None,
        score_threshold: float = 0.0,
        namespaces: list[str] | None = None,
        required: list[str] | None = None,
        always_include: list[str] | None = None,
        **framework_kwargs: Any,
    ) -> Any:
        """Per-call uniform entry point: delegates to :meth:`agent_builder`.

        ``dspy.ReAct`` fixes its tools at construction (no mid-run hook), so
        the live object is a builder — the deepest DSPy allows. Requires
        ``signature=`` (the DSPy task signature) in ``framework_kwargs``;
        ``max_iters`` and any other ``dspy.ReAct`` kwargs pass through too.
        ``required``/``always_include`` are re-applied on every rebuild (see
        :meth:`~agent_gantry.integrations.frameworks.base.GantryToolset.select`).
        Returns a :class:`GantryLiveDSPyReAct`; call ``await builder.build(query)``
        per task to get a fresh ``dspy.ReAct`` with tools re-selected for that
        query.
        """
        return self.agent_builder(
            limit=limit,
            score_threshold=score_threshold,
            namespaces=namespaces,
            required=required,
            always_include=always_include,
            **framework_kwargs,
        )

    def agent_builder(
        self,
        signature: Any,
        *,
        max_iters: int = 20,
        limit: int | None = None,
        score_threshold: float = 0.0,
        namespaces: list[str] | None = None,
        required: list[str] | None = None,
        always_include: list[str] | None = None,
        **react_kwargs: Any,
    ) -> Any:
        """Return a builder that rebuilds a fresh ``dspy.ReAct`` per call with re-selected tools.

        Named ``agent_builder`` for consistency with ``CrewAIAdapter``/
        ``AgnoAdapter``/``SmolagentsAdapter`` — the same "tools fixed at
        construction, rebuild per call" tier (see the module docstring); not
        ``react(...)``, since the returned object's ``.build(query)`` method
        is what actually produces the ``dspy.ReAct`` instance.

        ``signature`` and ``react_kwargs`` (``max_iters``/...) are forwarded
        to the builder. Call ``await builder.build(query)`` per task.
        """
        return GantryLiveDSPyReAct(
            self._gantry,
            signature,
            max_iters=max_iters,
            limit=self._default_limit if limit is None else limit,
            score_threshold=score_threshold,
            namespaces=namespaces,
            required=required,
            always_include=always_include,
            **react_kwargs,
        )


__all__ = ["DSPyAdapter", "GantryLiveDSPyReAct"]
