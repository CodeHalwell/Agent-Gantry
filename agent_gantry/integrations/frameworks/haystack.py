"""Haystack 2.x native tool adapter for Agent-Gantry.

Selects a relevant slice of Gantry tools and wraps each as a Haystack
``haystack.tools.Tool`` — the native tool object Haystack components introspect
(name / description / JSON-schema parameters) and invoke via a plain callable.
The ``haystack`` import is lazy so ``import agent_gantry`` never requires
Haystack to be installed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from agent_gantry.integrations.frameworks.base import GantryToolset, ToolSpec

if TYPE_CHECKING:
    from agent_gantry.core.gantry import AgentGantry


def spec_to_haystack(spec: ToolSpec) -> Any:
    """Wrap a :class:`ToolSpec` as a Haystack ``Tool``.

    The ``haystack`` import happens here, lazily, so callers without Haystack
    installed only hit the error when they actually export a tool. Haystack
    calls ``function`` with the tool arguments as keyword arguments, so the
    wrapper routes those through ``spec.invoke`` (and thus ``gantry.execute``).
    """
    try:
        from haystack.tools import Tool
    except ImportError as exc:  # pragma: no cover - exercised via stub
        raise ImportError(
            "Haystack support requires `haystack-ai`. "
            "Install it with `pip install haystack-ai`."
        ) from exc

    def _function(**kwargs: Any) -> Any:
        return spec.invoke(**kwargs)

    return Tool(
        name=spec.name,
        description=spec.description,
        parameters=spec.parameters,
        function=_function,
    )


async def for_haystack(
    gantry: AgentGantry,
    query: str,
    *,
    limit: int = 3,
    **select_kwargs: Any,
) -> list[Any]:
    """Select tools for ``query`` and return them as Haystack ``Tool``s."""
    specs = await GantryToolset(gantry).select(query, limit=limit, **select_kwargs)
    return [spec_to_haystack(s) for s in specs]
