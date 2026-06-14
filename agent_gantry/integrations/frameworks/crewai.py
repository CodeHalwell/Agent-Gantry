"""CrewAI native tool adapter for Agent-Gantry.

Selects a relevant slice of Gantry tools and wraps each as a CrewAI
``BaseTool`` — the native tool object CrewAI agents introspect
(name / description) and invoke via ``_run``. The ``crewai`` import is lazy so
``import agent_gantry`` never requires CrewAI to be installed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from agent_gantry.integrations.frameworks.base import GantryToolset, ToolSpec

if TYPE_CHECKING:
    from agent_gantry.core.gantry import AgentGantry


def spec_to_crewai(spec: ToolSpec) -> Any:
    """Wrap a :class:`ToolSpec` as a CrewAI ``BaseTool``.

    The ``crewai`` import happens here, lazily, so callers without CrewAI
    installed only hit the error when they actually export a tool. A subclass of
    ``BaseTool`` is built dynamically whose ``name`` / ``description`` come from
    the spec and whose ``_run`` routes through ``gantry.execute``.
    """
    try:
        from crewai.tools import BaseTool
    except ImportError as exc:  # pragma: no cover - exercised via stub
        raise ImportError(
            "CrewAI support requires `crewai`. "
            "Install it with `pip install crewai`."
        ) from exc

    def _run(self: Any, **kwargs: Any) -> Any:
        return spec.invoke(**kwargs)

    tool_cls = type(
        "GantryCrewAITool",
        (BaseTool,),
        {
            "name": spec.name,
            "description": spec.description,
            "_run": _run,
        },
    )
    return tool_cls()


async def for_crewai(
    gantry: AgentGantry,
    query: str,
    *,
    limit: int = 3,
    **select_kwargs: Any,
) -> list[Any]:
    """Select tools for ``query`` and return them as CrewAI ``BaseTool``s."""
    specs = await GantryToolset(gantry).select(query, limit=limit, **select_kwargs)
    return [spec_to_crewai(s) for s in specs]
