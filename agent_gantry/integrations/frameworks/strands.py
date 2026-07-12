"""AWS Strands Agents native tool adapter for Agent-Gantry.

Selects a relevant slice of Gantry tools and wraps each as a Strands
``DecoratedFunctionTool`` — the native tool object a Strands ``Agent``
introspects (name / description / JSON-Schema input) and invokes. The
``strands`` import is lazy so ``import agent_gantry`` never requires Strands
Agents to be installed.

Public entry point: :class:`StrandsAdapter`.

Strands genuinely supports **per-turn** dynamic tool re-selection: it fires a
``BeforeModelCallEvent`` hook immediately before every model call, and only
reads ``agent.tool_registry`` for the tool specs sent to the model *after*
that hook runs (see
:mod:`agent_gantry.integrations.frameworks.strands_live`). That is deeper than
the per-top-level-call rebuild used for frameworks that fix their tool list at
agent construction (CrewAI, Agno, Haystack, Smolagents) — it mirrors the depth
of Google ADK's ``before_model_callback``.
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


def _spec_to_strands(spec: ToolSpec) -> Any:
    """Wrap a :class:`ToolSpec` as a Strands ``DecoratedFunctionTool``.

    Builds an async callable with a real ``__signature__`` via
    :meth:`ToolSpec.callable_for_signature` (so Strands' ``@tool`` decorator
    validates and advertises the actual parameters, not a bare ``**kwargs``
    no-argument tool), then decorates it with explicit
    ``name``/``description``/``inputSchema`` overrides — all three are
    supported directly by ``strands.tool()`` — so Gantry's own metadata,
    including per-parameter descriptions, wins over whatever the decorator
    would otherwise infer from the wrapper's docstring and its own
    Pydantic-derived schema.

    The ``strands`` import happens here, lazily, so callers without Strands
    Agents installed only hit the error when they actually export a tool.

    Raises:
        ImportError: If ``strands-agents`` is not installed.
    """
    try:
        from strands import tool as strands_tool
    except ImportError as exc:  # pragma: no cover - exercised via stub
        raise ImportError(
            "Strands Agents support requires `strands-agents`. "
            "Install it with `pip install strands-agents`."
        ) from exc

    async_fn = spec.callable_for_signature()
    return strands_tool(
        name=spec.name,
        description=spec.description,
        inputSchema={"json": spec.parameters},
    )(async_fn)


async def _for_strands(
    gantry: AgentGantry,
    query: str,
    *,
    limit: int = DEFAULT_TOOL_LIMIT,
    **select_kwargs: Any,
) -> list[Any]:
    """Select tools for ``query`` and return them as Strands ``DecoratedFunctionTool``s."""
    specs = await GantryToolset(gantry).select(query, limit=limit, **select_kwargs)
    return [_spec_to_strands(s) for s in specs]


class StrandsAdapter(BaseFrameworkAdapter):
    """Route Gantry-selected tools into AWS Strands Agents.

    Static slice (``DecoratedFunctionTool`` objects for ``Agent(tools=[...])``)
    plus a genuine per-turn live hook — Strands fires a ``BeforeModelCallEvent``
    before every model call and only reads the tool registry afterward, so a
    hook that swaps the registry during that event changes what the model sees
    on the very call about to happen (see
    :mod:`agent_gantry.integrations.frameworks.strands_live`). Every call routes
    through ``gantry.execute`` so retries, timeouts, circuit breakers, and the
    security policy all apply::

        from agent_gantry import AgentGantry
        from agent_gantry.strands import StrandsAdapter
        from strands import Agent

        gantry = AgentGantry()
        # ... register tools, await gantry.sync() ...

        tools = await StrandsAdapter(gantry).select("email the quarterly report", limit=3)
        agent = Agent(tools=tools)
    """

    @staticmethod
    def convert(spec: ToolSpec) -> Any:
        """Wrap a single :class:`ToolSpec` as a Strands ``DecoratedFunctionTool``."""
        return _spec_to_strands(spec)

    def tool_hook(self, *, limit: int | None = None, score_threshold: float = 0.0) -> Any:
        """Build a ``HookProvider`` that re-selects Gantry tools before every model call.

        Pass the result straight to ``Agent(hooks=[...])`` (or
        ``agent.add_hook(hook)`` on an already-built agent). Each turn it
        re-runs semantic selection against the latest user message and swaps
        the agent's tool registry in place — the deepest re-selection tier
        Strands supports.
        """
        from agent_gantry.integrations.frameworks.strands_live import (
            GantryStrandsToolHook,
        )

        return GantryStrandsToolHook(
            self._gantry,
            limit=self._default_limit if limit is None else limit,
            score_threshold=score_threshold,
        )

    def agent(
        self,
        *,
        limit: int | None = None,
        score_threshold: float = 0.0,
        **agent_kwargs: Any,
    ) -> Any:
        """Build a ``strands.Agent`` wired for per-turn dynamic tool selection.

        Constructs ``Agent(tools=[], hooks=[<tool_hook>], **agent_kwargs)`` —
        the agent ships with no statically registered tools; the hook injects
        the relevant slice before every model call (see :meth:`tool_hook`).
        ``agent_kwargs`` (``model``, ``system_prompt``, ...) are forwarded.
        """
        from agent_gantry.integrations.frameworks.strands_live import (
            _gantry_strands_agent,
        )

        return _gantry_strands_agent(
            self._gantry,
            limit=self._default_limit if limit is None else limit,
            score_threshold=score_threshold,
            **agent_kwargs,
        )
