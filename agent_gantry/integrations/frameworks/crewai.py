"""CrewAI native tool adapter for Agent-Gantry.

Selects a relevant slice of Gantry tools and wraps each as a CrewAI
``BaseTool`` — the native tool object CrewAI agents introspect
(name / description) and invoke via ``_run``. The ``crewai`` import is lazy so
``import agent_gantry`` never requires CrewAI to be installed.

Public entry point: :class:`CrewAIAdapter`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from agent_gantry.integrations.frameworks.base import GantryToolset, ToolSpec

if TYPE_CHECKING:
    from agent_gantry.core.gantry import AgentGantry


def _spec_to_crewai(spec: ToolSpec) -> Any:
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

    # CrewAI's BaseTool is a Pydantic v2 model: ``name`` / ``description`` are
    # declared fields and MUST be set via the constructor, not as bare class
    # attributes (Pydantic rejects un-annotated field overrides). ``_run`` is
    # the abstract method we implement; it closes over ``spec``. We also pass a
    # generated ``args_schema`` so CrewAI surfaces the real parameters to the
    # LLM rather than a no-argument tool.
    class GantryCrewAITool(BaseTool):  # type: ignore[misc, valid-type]
        def _run(self, **kwargs: Any) -> Any:
            return spec.invoke(**kwargs)

    kwargs: dict[str, Any] = {"name": spec.name, "description": spec.description}
    args_schema = _build_args_schema(spec)
    if args_schema is not None:
        kwargs["args_schema"] = args_schema
    return GantryCrewAITool(**kwargs)


def _build_args_schema(spec: ToolSpec) -> Any:
    """Build a Pydantic args model from the spec's JSON-Schema parameters.

    Returns ``None`` when there are no properties (CrewAI then uses its own
    default empty schema). Best-effort: if Pydantic model creation fails for any
    reason, fall back to ``None`` so the tool is still usable (sans typed args).
    """
    properties = spec.parameters.get("properties") or {}
    if not properties:
        return None
    try:
        from pydantic import Field, create_model

        from agent_gantry.integrations.frameworks.base import _json_type_to_python

        required = set(spec.parameters.get("required") or [])
        fields: dict[str, Any] = {}
        for name, prop in properties.items():
            annotation = _json_type_to_python(
                prop.get("type") if isinstance(prop, dict) else None
            )
            description = prop.get("description", "") if isinstance(prop, dict) else ""
            if name in required:
                fields[name] = (annotation, Field(..., description=description))
            else:
                fields[name] = (annotation | None, Field(default=None, description=description))
        return create_model(f"{spec.name}_Args", **fields)
    except Exception:  # noqa: BLE001 - schema is best-effort
        return None


async def _for_crewai(
    gantry: AgentGantry,
    query: str,
    *,
    limit: int = 3,
    **select_kwargs: Any,
) -> list[Any]:
    """Select tools for ``query`` and return them as CrewAI ``BaseTool``s."""
    specs = await GantryToolset(gantry).select(query, limit=limit, **select_kwargs)
    return [_spec_to_crewai(s) for s in specs]


class CrewAIAdapter:
    """Route Gantry-selected tools into CrewAI.

    Static slice (``crewai.tools.BaseTool`` objects) plus per-call live helpers
    (CrewAI fixes an agent's tools at construction, so the live path rebuilds a
    fresh agent per call). Every call routes through ``gantry.execute``.
    """

    def __init__(self, gantry: AgentGantry, *, default_limit: int = 3) -> None:
        self._gantry = gantry
        self._default_limit = default_limit

    @staticmethod
    def convert(spec: ToolSpec) -> Any:
        """Wrap a single :class:`ToolSpec` as a CrewAI ``BaseTool``."""
        return _spec_to_crewai(spec)

    async def select(self, query: str, *, limit: int | None = None, **select_kwargs: Any) -> list[Any]:
        """Select tools for ``query`` as CrewAI ``BaseTool``s (static slice)."""
        return await _for_crewai(self._gantry, query, limit=self._default_limit if limit is None else limit, **select_kwargs)

    async def live_tools(self, query: str, *, limit: int = 5, score_threshold: float = 0.0) -> list[Any]:
        """Re-select CrewAI ``BaseTool``s for THIS call's ``query`` (per-call selection)."""
        return await _for_crewai(self._gantry, query, limit=limit, score_threshold=score_threshold)

    def agent_builder(self, *, limit: int = 5, score_threshold: float = 0.0, **agent_kwargs: Any) -> Any:
        """Return a builder that rebuilds a fresh ``crewai.Agent`` per call with re-selected tools.

        ``agent_kwargs`` (role/goal/backstory/llm/...) are forwarded to the builder.
        Call ``await builder.build(query)`` per task.
        """
        from agent_gantry.integrations.frameworks.live_wrappers import GantryLiveCrewAgent
        return GantryLiveCrewAgent(self._gantry, limit=limit, score_threshold=score_threshold, **agent_kwargs)
