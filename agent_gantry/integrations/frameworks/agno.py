"""Agno (formerly Phidata) native tool adapter for Agent-Gantry.

Selects a relevant slice of Gantry tools and wraps each as an Agno
``Function`` — the native tool object an Agno agent introspects (name /
description / JSON-schema parameters) and invokes. The ``agno`` import is lazy
so ``import agent_gantry`` never requires Agno to be installed.

Public entry point: :class:`AgnoAdapter`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from agent_gantry.integrations.frameworks.base import GantryToolset, ToolSpec

if TYPE_CHECKING:
    from agent_gantry.core.gantry import AgentGantry


def _spec_to_agno(spec: ToolSpec) -> Any:
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

    async_fn = spec.callable_for_signature()

    def _entrypoint(**kwargs: Any) -> Any:
        return spec.invoke(**kwargs)

    _entrypoint.__name__ = spec.name
    _entrypoint.__doc__ = spec.description
    # Copy the real signature so Agno introspection surfaces the actual
    # parameters instead of a bare **kwargs (no-argument) tool.
    _entrypoint.__signature__ = async_fn.__signature__  # type: ignore[attr-defined]
    _entrypoint.__annotations__ = dict(getattr(async_fn, "__annotations__", {}))

    return Function(
        name=spec.name,
        description=spec.description,
        parameters=spec.parameters,
        entrypoint=_entrypoint,
    )


async def _for_agno(
    gantry: AgentGantry,
    query: str,
    *,
    limit: int = 3,
    **select_kwargs: Any,
) -> list[Any]:
    """Select tools for ``query`` and return them as Agno ``Function``s."""
    specs = await GantryToolset(gantry).select(query, limit=limit, **select_kwargs)
    return [_spec_to_agno(s) for s in specs]


class AgnoAdapter:
    """Route Gantry-selected tools into Agno.

    Static slice (``agno.tools.function.Function`` objects) plus a per-call live
    builder (Agno fixes tools at construction). Every call routes through ``gantry.execute``.
    """

    def __init__(self, gantry: AgentGantry, *, default_limit: int = 3) -> None:
        self._gantry = gantry
        self._default_limit = default_limit

    @staticmethod
    def convert(spec: ToolSpec) -> Any:
        """Wrap a single :class:`ToolSpec` as an Agno ``Function``."""
        return _spec_to_agno(spec)

    async def select(
        self, query: str, *, limit: int | None = None, **select_kwargs: Any
    ) -> list[Any]:
        """Select tools for ``query`` as Agno ``Function``s (static slice)."""
        return await _for_agno(
            self._gantry,
            query,
            limit=self._default_limit if limit is None else limit,
            **select_kwargs,
        )

    def agent_builder(
        self,
        *,
        limit: int = 5,
        score_threshold: float = 0.0,
        **agent_kwargs: Any,
    ) -> Any:
        """Return a builder that rebuilds a fresh ``agno.agent.Agent`` per call with re-selected tools.

        ``agent_kwargs`` (model/...) are forwarded. Call ``await builder.build(query)`` per run.
        """
        from agent_gantry.integrations.frameworks.live_wrappers import (
            GantryLiveAgnoAgent,
        )

        return GantryLiveAgnoAgent(
            self._gantry,
            limit=limit,
            score_threshold=score_threshold,
            **agent_kwargs,
        )
