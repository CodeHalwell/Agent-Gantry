"""AutoGen / AG2 native tool adapter for Agent-Gantry.

Selects a relevant slice of Gantry tools and exposes each as a plain Python
callable that AutoGen / AG2 can register. AutoGen registers a tool by
associating a callable with a *caller* agent (which proposes the call) and an
*executor* agent (which runs it) via
``autogen.register_function(func, *, caller, executor, name, description)``.

The ``autogen`` import is lazy (only inside :func:`register_with_autogen`) so
``import agent_gantry`` never requires AutoGen to be installed, and the
schema-only helpers (:func:`spec_to_autogen`, :func:`for_autogen`) work without
the framework present.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from agent_gantry.integrations.frameworks.base import GantryToolset, ToolSpec

if TYPE_CHECKING:
    from agent_gantry.core.gantry import AgentGantry


def spec_to_autogen(spec: ToolSpec) -> dict[str, Any]:
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


async def for_autogen(
    gantry: AgentGantry,
    query: str,
    *,
    limit: int = 3,
    **select_kwargs: Any,
) -> list[dict[str, Any]]:
    """Select tools for ``query`` and return AutoGen-registrable mappings."""
    specs = await GantryToolset(gantry).select(query, limit=limit, **select_kwargs)
    return [spec_to_autogen(s) for s in specs]


async def register_with_autogen(
    gantry: AgentGantry,
    query: str,
    *,
    caller: Any,
    executor: Any,
    limit: int = 3,
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
            "AutoGen support requires `autogen`. "
            "Install it with `pip install pyautogen`."
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
