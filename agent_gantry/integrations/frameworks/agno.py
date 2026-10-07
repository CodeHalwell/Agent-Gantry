"""Agno (formerly Phidata) native tool adapter for Agent-Gantry.

Selects a relevant slice of Gantry tools and wraps each as an Agno
``Function`` — the native tool object an Agno agent introspects (name /
description / JSON-schema parameters) and invokes. The ``agno`` import is lazy
so ``import agent_gantry`` never requires Agno to be installed.

Public entry point: :class:`AgnoAdapter`.
"""

from __future__ import annotations

from typing import Any

from agent_gantry.integrations.frameworks.base import BaseFrameworkAdapter, ToolSpec

_INSTALL_HINT = "Agno support requires `agno`. Install it with `pip install agno` (or `uv add agno`)."


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
        raise ImportError(_INSTALL_HINT) from exc

    # A sync callable with the real signature (so Agno introspection surfaces
    # the actual parameters, not a bare ``**kwargs`` no-argument tool) that maps
    # a renamed parameter back to the schema's property before the tool runs.
    # The advertised schema carries the same aliases as the signature, as the
    # ADK and Strands adapters do; schema and signature disagreeing made a
    # property such as ``user-id`` impossible to supply.
    return Function(
        name=spec.name,
        description=spec.description,
        parameters=spec.aliased_parameters(),
        entrypoint=spec.sync_callable_for_signature(),
    )


class AgnoAdapter(BaseFrameworkAdapter):
    """Route Gantry-selected tools into Agno.

    Static slice (``agno.tools.function.Function`` objects) plus a per-call live
    builder (Agno fixes tools at construction). Every call routes through
    ``gantry.execute``. :meth:`live` returns the :meth:`agent_builder` builder;
    call ``await builder.build(query)`` before each new run.
    """

    live_tier = "per-call"
    _live_delegate = "agent_builder"

    @staticmethod
    def convert(spec: ToolSpec) -> Any:
        """Wrap a single :class:`ToolSpec` as an Agno ``Function``."""
        return _spec_to_agno(spec)

    def agent_builder(
        self,
        *,
        limit: int | None = None,
        score_threshold: float = 0.0,
        namespaces: list[str] | None = None,
        required: list[str] | None = None,
        always_include: list[str] | None = None,
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
            **self._selection_kwargs(limit, score_threshold, namespaces, required, always_include),
            **agent_kwargs,
        )
