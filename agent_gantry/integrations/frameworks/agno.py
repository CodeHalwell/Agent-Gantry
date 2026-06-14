"""Agno (formerly Phidata) native tool adapter for Agent-Gantry.

Selects a relevant slice of Gantry tools and wraps each as an Agno
``Function`` — the native tool object an Agno agent introspects (name /
description / JSON-schema parameters) and invokes. The ``agno`` import is lazy
so ``import agent_gantry`` never requires Agno to be installed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from agent_gantry.integrations.frameworks.base import GantryToolset, ToolSpec

if TYPE_CHECKING:
    from agent_gantry.core.gantry import AgentGantry


def spec_to_agno(spec: ToolSpec) -> Any:
    """Wrap a :class:`ToolSpec` as an Agno ``Function``.

    Agno calls the ``entrypoint`` with the tool arguments as keyword arguments,
    so the entrypoint is a sync wrapper that routes through ``spec.invoke`` (and
    therefore ``gantry.execute``). The ``agno`` import happens here, lazily, so
    callers without Agno installed only hit the error when they actually export
    a tool.
    """
    try:
        from agno.tools.function import Function
    except ImportError as exc:  # pragma: no cover - exercised via stub
        raise ImportError(
            "Agno support requires `agno`. Install it with `pip install agno`."
        ) from exc

    def _entrypoint(**kwargs: Any) -> Any:
        return spec.invoke(**kwargs)

    _entrypoint.__name__ = spec.name
    _entrypoint.__doc__ = spec.description

    return Function(
        name=spec.name,
        description=spec.description,
        parameters=spec.parameters,
        entrypoint=_entrypoint,
    )


async def for_agno(
    gantry: AgentGantry,
    query: str,
    *,
    limit: int = 3,
    **select_kwargs: Any,
) -> list[Any]:
    """Select tools for ``query`` and return them as Agno ``Function``s."""
    specs = await GantryToolset(gantry).select(query, limit=limit, **select_kwargs)
    return [spec_to_agno(s) for s in specs]
