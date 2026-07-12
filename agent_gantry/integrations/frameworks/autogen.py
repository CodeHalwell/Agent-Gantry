"""AutoGen / AG2 native tool adapter for Agent-Gantry.

Selects a relevant slice of Gantry tools and exposes each as a plain Python
callable that AutoGen / AG2 can register. AutoGen registers a tool by
associating a callable with a *caller* agent (which proposes the call) and an
*executor* agent (which runs it) via
``autogen.register_function(func, *, caller, executor, name, description)``.

The ``autogen`` import is lazy (only inside :func:`_register_with_autogen`) so
``import agent_gantry`` never requires AutoGen to be installed, and the
schema-only helpers (:func:`_spec_to_autogen`, :func:`_for_autogen`) work without
the framework present.

Public entry point: :class:`AutoGenAdapter`.
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


def _spec_to_autogen(spec: ToolSpec) -> dict[str, Any]:
    """Describe a :class:`ToolSpec` as an AutoGen-registrable mapping.

    Returns the metadata plus a fresh async callable (from
    :meth:`ToolSpec.callable_for_signature`) that executes through Gantry. No
    framework import is needed to build this mapping.
    """
    return {
        "name": spec.name,
        "description": spec.description,
        "callable": spec.callable_for_signature(),
    }


async def _for_autogen(
    gantry: AgentGantry,
    query: str,
    *,
    limit: int = DEFAULT_TOOL_LIMIT,
    **select_kwargs: Any,
) -> list[dict[str, Any]]:
    """Select tools for ``query`` and return AutoGen-registrable mappings."""
    specs = await GantryToolset(gantry).select(query, limit=limit, **select_kwargs)
    return [_spec_to_autogen(s) for s in specs]


async def _register_with_autogen(
    gantry: AgentGantry,
    query: str,
    *,
    caller: Any,
    executor: Any,
    limit: int = DEFAULT_TOOL_LIMIT,
    **select_kwargs: Any,
) -> list[str]:
    """Select tools and register each with AutoGen's caller/executor agents.

    The ``autogen`` import happens here, lazily, so callers without AutoGen
    installed only hit the error when they actually register tools. Each
    selected tool is wired up via ``autogen.register_function``.

    Returns the list of registered tool names.

    Raises:
        ImportError: If ``autogen`` (pyautogen) is not installed.
    """
    try:
        from autogen import register_function
    except ImportError as exc:  # pragma: no cover - exercised via fake module
        raise ImportError(
            "AutoGen support requires `autogen`. Install it with `pip install pyautogen`."
        ) from exc

    specs = await GantryToolset(gantry).select(query, limit=limit, **select_kwargs)
    names: list[str] = []
    for spec in specs:
        register_function(
            spec.callable_for_signature(),
            caller=caller,
            executor=executor,
            name=spec.name,
            description=spec.description,
        )
        names.append(spec.name)
    return names


class AutoGenAdapter(BaseFrameworkAdapter):
    """Route Gantry-selected tools into AutoGen / AG2.

    Static slice (registrable callable mappings), direct registration with
    caller/executor agents, and a deep per-turn live ``Workbench``. Every call
    routes through ``gantry.execute``.
    """

    live_tier = "per-turn"

    @staticmethod
    def convert(spec: ToolSpec) -> dict[str, Any]:
        """Describe a single :class:`ToolSpec` as an AutoGen-registrable mapping."""
        return _spec_to_autogen(spec)

    def live(
        self,
        *,
        limit: int | None = None,
        score_threshold: float = 0.0,
        namespaces: list[str] | None = None,
        **framework_kwargs: Any,
    ) -> Any:
        """Per-turn uniform entry point: delegates to :meth:`workbench`.

        Returns a ``GantryWorkbench`` (an ``autogen_core.tools.Workbench``
        subclass) — plug it into the agent that consumes ``Workbench``
        instances in your installed AutoGen/AG2 version. Update its query
        between turns via ``.set_query(...)``. No ``framework_kwargs`` are
        required; ``query`` may be passed as one to seed the first turn.
        """
        return self.workbench(
            limit=limit, score_threshold=score_threshold, namespaces=namespaces, **framework_kwargs
        )

    async def register(
        self,
        query: str,
        *,
        caller: Any,
        executor: Any,
        limit: int | None = None,
        **select_kwargs: Any,
    ) -> list[str]:
        """Select tools and register each with AutoGen's caller/executor agents; return the names."""
        return await _register_with_autogen(
            self._gantry,
            query,
            caller=caller,
            executor=executor,
            limit=self._default_limit if limit is None else limit,
            **select_kwargs,
        )

    def workbench(
        self,
        *,
        query: str = "",
        limit: int | None = None,
        score_threshold: float = 0.0,
        namespaces: list[str] | None = None,
    ) -> Any:
        """Build a live ``Workbench`` for per-turn dynamic tool provision (``AssistantAgent``)."""
        from agent_gantry.integrations.frameworks.autogen_live import _gantry_workbench

        return _gantry_workbench(
            self._gantry,
            query=query,
            limit=self._default_limit if limit is None else limit,
            score_threshold=score_threshold,
            namespaces=namespaces,
        )
