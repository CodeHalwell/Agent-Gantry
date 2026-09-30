"""Haystack 2.x native tool adapter for Agent-Gantry.

Selects a relevant slice of Gantry tools and wraps each as a Haystack
``haystack.tools.Tool`` — the native tool object Haystack components introspect
(name / description / JSON-schema parameters) and invoke via a plain callable.
The ``haystack`` import is lazy so ``import agent_gantry`` never requires
Haystack to be installed.

Public entry point: :class:`HaystackAdapter`.
"""

from __future__ import annotations

from typing import Any

from agent_gantry.integrations.frameworks.base import BaseFrameworkAdapter, ToolSpec

_INSTALL_HINT = (
    "Haystack support requires `haystack-ai`. "
    "Install it with `pip install haystack-ai` (or `uv add haystack-ai`)."
)


def _spec_to_haystack(spec: ToolSpec) -> Any:
    """Wrap a :class:`ToolSpec` as a Haystack ``Tool``.

    The ``haystack`` import happens here, lazily, so callers without Haystack
    installed only hit the error when they actually export a tool. Haystack
    calls ``function`` with the tool arguments as keyword arguments, so the
    wrapper routes those through ``spec.invoke`` (and thus ``gantry.execute``).
    """
    try:
        from haystack.tools import Tool
    except ImportError as exc:  # pragma: no cover - exercised via stub
        raise ImportError(_INSTALL_HINT) from exc

    def _function(**kwargs: Any) -> Any:
        return spec.invoke(**kwargs)

    return Tool(
        name=spec.name,
        description=spec.description,
        parameters=spec.parameters,
        function=_function,
    )


class HaystackAdapter(BaseFrameworkAdapter):
    """Route Gantry-selected tools into Haystack.

    Static slice (``haystack.tools.Tool`` objects) plus per-call live helpers
    (Haystack fixes a component's tools at construction). Every call routes
    through ``gantry.execute``. Works with haystack 2.x (``ToolInvoker``) and
    haystack >= 3.0 (``Agent`` owns tool execution — see
    :meth:`tool_invoker_builder`). :meth:`live` returns that builder; call
    ``await builder.build(query)`` before each new call.
    """

    live_tier = "per-call"
    _live_delegate = "tool_invoker_builder"

    @staticmethod
    def convert(spec: ToolSpec) -> Any:
        """Wrap a single :class:`ToolSpec` as a Haystack ``Tool``."""
        return _spec_to_haystack(spec)

    async def live_tools(
        self, query: str, *, limit: int | None = None, **select_kwargs: Any
    ) -> list[Any]:
        """Re-select Haystack ``Tool``s for THIS call's ``query`` (per-call selection)."""
        return await self.select(query, limit=limit, **select_kwargs)

    def tool_invoker_builder(
        self,
        *,
        limit: int | None = None,
        score_threshold: float = 0.0,
        namespaces: list[str] | None = None,
        required: list[str] | None = None,
        always_include: list[str] | None = None,
        **invoker_kwargs: Any,
    ) -> Any:
        """Return a builder that rebuilds a fresh tool-execution component per call.

        On haystack 2.x each ``await builder.build(query)`` returns a
        ``ToolInvoker``; on haystack >= 3.0 (which removed ``ToolInvoker``)
        it returns a ``haystack.components.agents.Agent`` and
        ``invoker_kwargs`` must include ``chat_generator=...``.
        ``invoker_kwargs`` are forwarded to the built component.
        """
        from agent_gantry.integrations.frameworks.live_wrappers import (
            GantryLiveHaystackToolInvoker,
        )

        return GantryLiveHaystackToolInvoker(
            self._gantry,
            **self._selection_kwargs(limit, score_threshold, namespaces, required, always_include),
            **invoker_kwargs,
        )
