"""Google ADK native tool adapter for Agent-Gantry.

Selects a relevant slice of Gantry tools and wraps each as a Google Agent
Development Kit ``FunctionTool`` — the native tool object an ADK agent
introspects (from the callable's name, docstring and signature) and invokes.
The ``google.adk`` import is lazy so ``import agent_gantry`` never requires ADK
to be installed.

Public entry point: :class:`GoogleADKAdapter`.
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


def _spec_to_google_adk(spec: ToolSpec) -> Any:
    """Wrap a :class:`ToolSpec` as a Google ADK ``FunctionTool``.

    ADK builds the LLM tool schema by introspecting the wrapped callable's
    name, docstring and signature, so we hand it
    :meth:`ToolSpec.callable_for_signature` — an async function that carries a
    real ``__signature__`` derived from the tool's JSON schema (not a bare
    ``**kwargs``) and routes through ``gantry.execute``.

    Raises:
        ImportError: If ``google-adk`` is not installed.
    """
    try:
        from google.adk.tools import FunctionTool
    except ImportError as exc:  # pragma: no cover - exercised via stub
        raise ImportError(
            "Google ADK support requires `google-adk`. Install it with `pip install google-adk`."
        ) from exc

    # ADK's automatic function calling rejects `T | None` and `None`-typed
    # defaults, so opt into type-matched empty defaults for optional params.
    return FunctionTool(func=spec.callable_for_signature(type_matched_defaults=True))


async def _for_google_adk(
    gantry: AgentGantry,
    query: str,
    *,
    limit: int = DEFAULT_TOOL_LIMIT,
    **select_kwargs: Any,
) -> list[Any]:
    """Select tools for ``query`` and return them as ADK ``FunctionTool``s."""
    specs = await GantryToolset(gantry).select(query, limit=limit, **select_kwargs)
    return [_spec_to_google_adk(s) for s in specs]


class GoogleADKAdapter(BaseFrameworkAdapter):
    """Route Gantry-selected tools into Google ADK.

    Static slice (``google.adk.tools.FunctionTool`` objects) plus deep per-turn
    live wiring (re-selects tools before every model request). Every call routes
    through ``gantry.execute``.
    """

    live_tier = "per-turn"

    @staticmethod
    def convert(spec: ToolSpec) -> Any:
        """Wrap a single :class:`ToolSpec` as a Google ADK ``FunctionTool``."""
        return _spec_to_google_adk(spec)

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
        """Per-turn uniform entry point: delegates to :meth:`before_model_callback`.

        Returns an ``async (callback_context, llm_request) -> None`` callback
        — plug it into ``Agent(tools=[], before_model_callback=<result>)``.
        ``required``/``always_include`` are re-applied on every model request
        (see
        :meth:`~agent_gantry.integrations.frameworks.base.GantryToolset.select`).
        No other ``framework_kwargs`` are required (unlike LangGraph/OpenAI
        Agents/Semantic Kernel, this hook is agent-agnostic).
        """
        return self.before_model_callback(
            limit=limit,
            score_threshold=score_threshold,
            namespaces=namespaces,
            required=required,
            always_include=always_include,
            **framework_kwargs,
        )

    def before_model_callback(
        self,
        *,
        limit: int | None = None,
        score_threshold: float = 0.0,
        namespaces: list[str] | None = None,
        required: list[str] | None = None,
        always_include: list[str] | None = None,
    ) -> Any:
        """Build an ADK ``before_model_callback`` that injects Gantry tools per turn."""
        from agent_gantry.integrations.frameworks.google_adk_live import (
            _gantry_before_model_callback,
        )

        return _gantry_before_model_callback(
            self._gantry,
            limit=self._default_limit if limit is None else limit,
            score_threshold=score_threshold,
            namespaces=namespaces,
            required=required,
            always_include=always_include,
        )

    def agent(
        self,
        *,
        model: Any,
        name: str,
        instruction: str = "",
        limit: int | None = None,
        score_threshold: float = 0.0,
        namespaces: list[str] | None = None,
        required: list[str] | None = None,
        always_include: list[str] | None = None,
        **agent_kwargs: Any,
    ) -> Any:
        """Build an ADK ``Agent`` wired for per-turn dynamic tool selection (tools=[] + callback)."""
        from agent_gantry.integrations.frameworks.google_adk_live import (
            _gantry_adk_agent,
        )

        return _gantry_adk_agent(
            self._gantry,
            model=model,
            name=name,
            instruction=instruction,
            limit=self._default_limit if limit is None else limit,
            score_threshold=score_threshold,
            namespaces=namespaces,
            required=required,
            always_include=always_include,
            **agent_kwargs,
        )
