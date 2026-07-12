"""Haystack 2.x native tool adapter for Agent-Gantry.

Selects a relevant slice of Gantry tools and wraps each as a Haystack
``haystack.tools.Tool`` — the native tool object Haystack components introspect
(name / description / JSON-schema parameters) and invoke via a plain callable.
The ``haystack`` import is lazy so ``import agent_gantry`` never requires
Haystack to be installed.

Public entry point: :class:`HaystackAdapter`.
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


async def _for_haystack(
    gantry: AgentGantry,
    query: str,
    *,
    limit: int = DEFAULT_TOOL_LIMIT,
    **select_kwargs: Any,
) -> list[Any]:
    """Select tools for ``query`` and return them as Haystack ``Tool``s."""
    specs = await GantryToolset(gantry).select(query, limit=limit, **select_kwargs)
    return [_spec_to_haystack(s) for s in specs]


class HaystackAdapter(BaseFrameworkAdapter):
    """Route Gantry-selected tools into Haystack.

    Static slice (``haystack.tools.Tool`` objects) plus per-call live helpers
    (Haystack fixes a ToolInvoker's tools at construction). Every call routes
    through ``gantry.execute``.
    """

    @staticmethod
    def convert(spec: ToolSpec) -> Any:
        """Wrap a single :class:`ToolSpec` as a Haystack ``Tool``."""
        return _spec_to_haystack(spec)

    async def live_tools(
        self, query: str, *, limit: int | None = None, **select_kwargs: Any
    ) -> list[Any]:
        """Re-select Haystack ``Tool``s for THIS call's ``query`` (per-call selection).

        Same selection surface as :meth:`select` (``score_threshold``,
        ``namespaces``, ``tools_already_used`` via ``**select_kwargs``).
        """
        return await _for_haystack(
            self._gantry, query, limit=self._default_limit if limit is None else limit, **select_kwargs
        )

    def tool_invoker_builder(
        self, *, limit: int | None = None, score_threshold: float = 0.0, **invoker_kwargs: Any
    ) -> Any:
        """Return a builder that rebuilds a fresh ``ToolInvoker`` per call with re-selected tools.

        ``invoker_kwargs`` are forwarded to ``ToolInvoker``. Call
        ``await builder.build(query)`` per call.
        """
        from agent_gantry.integrations.frameworks.live_wrappers import (
            GantryLiveHaystackToolInvoker,
        )

        return GantryLiveHaystackToolInvoker(
            self._gantry,
            limit=self._default_limit if limit is None else limit,
            score_threshold=score_threshold,
            **invoker_kwargs,
        )
