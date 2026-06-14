"""Google ADK native tool adapter for Agent-Gantry.

Selects a relevant slice of Gantry tools and wraps each as a Google Agent
Development Kit ``FunctionTool`` — the native tool object an ADK agent
introspects (from the callable's name, docstring and signature) and invokes.
The ``google.adk`` import is lazy so ``import agent_gantry`` never requires ADK
to be installed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from agent_gantry.integrations.frameworks.base import GantryToolset, ToolSpec

if TYPE_CHECKING:
    from agent_gantry.core.gantry import AgentGantry


def spec_to_google_adk(spec: ToolSpec) -> Any:
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
            "Google ADK support requires `google-adk`. "
            "Install it with `pip install google-adk`."
        ) from exc

    # ADK's automatic function calling rejects `T | None` and `None`-typed
    # defaults, so opt into type-matched empty defaults for optional params.
    return FunctionTool(func=spec.callable_for_signature(type_matched_defaults=True))


async def for_google_adk(
    gantry: AgentGantry,
    query: str,
    *,
    limit: int = 3,
    **select_kwargs: Any,
) -> list[Any]:
    """Select tools for ``query`` and return them as ADK ``FunctionTool``s."""
    specs = await GantryToolset(gantry).select(query, limit=limit, **select_kwargs)
    return [spec_to_google_adk(s) for s in specs]
