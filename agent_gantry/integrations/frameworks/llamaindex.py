"""LlamaIndex adapter: export selected Gantry tools as ``FunctionTool`` objects.

LlamaIndex agents consume :class:`llama_index.core.tools.FunctionTool` objects.
This module wraps Gantry-selected :class:`ToolSpec` handles into that native
type, routing every call back through ``gantry.execute`` so retries, timeouts,
circuit breakers, and the security policy still apply.

The ``llama_index`` import is lazy (performed inside the builder) so that
``import agent_gantry`` never requires LlamaIndex to be installed.

Public entry point: :class:`LlamaIndexAdapter`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from agent_gantry.integrations.frameworks.base import (
    DEFAULT_TOOL_LIMIT,
    BaseFrameworkAdapter,
    GantryToolset,
)

if TYPE_CHECKING:
    from agent_gantry.core.gantry import AgentGantry
    from agent_gantry.integrations.frameworks.base import ToolSpec


def _spec_to_llamaindex(spec: ToolSpec) -> Any:
    """Convert a :class:`ToolSpec` into a LlamaIndex ``FunctionTool``.

    Raises:
        ImportError: If ``llama-index-core`` is not installed.
    """
    try:
        from llama_index.core.tools import FunctionTool
    except ImportError as exc:  # pragma: no cover - exercised via fake module
        raise ImportError(
            "LlamaIndex is required for LlamaIndexAdapter; "
            "install it with `pip install llama-index-core`."
        ) from exc

    async_fn = spec.callable_for_signature()

    def _sync_fn(**kwargs: Any) -> Any:
        return spec.invoke(**kwargs)

    _sync_fn.__name__ = spec.name
    _sync_fn.__doc__ = spec.description
    # Copy the real signature so LlamaIndex introspection surfaces the actual
    # parameters instead of a bare **kwargs (no-argument) tool.
    _sync_fn.__signature__ = async_fn.__signature__  # type: ignore[attr-defined]
    _sync_fn.__annotations__ = dict(getattr(async_fn, "__annotations__", {}))

    return FunctionTool.from_defaults(
        fn=_sync_fn,
        async_fn=async_fn,
        name=spec.name,
        description=spec.description,
    )


async def _for_llamaindex(
    gantry: AgentGantry,
    query: str,
    *,
    limit: int = DEFAULT_TOOL_LIMIT,
    **select_kwargs: Any,
) -> list:
    """Select tools for ``query`` and return them as LlamaIndex ``FunctionTool``s."""
    specs = await GantryToolset(gantry).select(query, limit=limit, **select_kwargs)
    return [_spec_to_llamaindex(spec) for spec in specs]


class LlamaIndexAdapter(BaseFrameworkAdapter):
    """Route Gantry-selected tools into LlamaIndex.

    Static slice (``FunctionTool`` objects) plus deep per-turn live wiring
    (re-selects tools every reasoning step), both routed through ``gantry.execute``.
    """

    live_tier = "per-turn"

    @staticmethod
    def convert(spec: ToolSpec) -> Any:
        """Wrap a single :class:`ToolSpec` as a LlamaIndex ``FunctionTool``."""
        return _spec_to_llamaindex(spec)

    def live(
        self,
        *,
        limit: int | None = None,
        score_threshold: float = 0.0,
        namespaces: list[str] | None = None,
        **framework_kwargs: Any,
    ) -> Any:
        """Per-turn uniform entry point: delegates to :meth:`tool_retriever`.

        Returns a ``GantryToolRetriever`` (an ``llama_index.core.objects.
        ObjectRetriever`` subclass) — plug it into
        ``FunctionAgent(tool_retriever=<result>)``. No ``framework_kwargs``
        are required; any supplied are forwarded to the underlying retriever
        constructor.
        """
        return self.tool_retriever(
            limit=limit,
            score_threshold=score_threshold,
            namespaces=namespaces,
            **framework_kwargs,
        )

    def tool_retriever(
        self,
        *,
        limit: int | None = None,
        score_threshold: float = 0.0,
        namespaces: list[str] | None = None,
    ) -> Any:
        """Build a live per-turn ``ObjectRetriever`` for ``FunctionAgent(tool_retriever=...)``."""
        from agent_gantry.integrations.frameworks.llamaindex_live import (
            _gantry_tool_retriever,
        )

        return _gantry_tool_retriever(
            self._gantry,
            limit=self._default_limit if limit is None else limit,
            score_threshold=score_threshold,
            namespaces=namespaces,
        )

    def function_agent(
        self,
        llm: Any,
        *,
        name: str = "gantry_agent",
        limit: int | None = None,
        score_threshold: float = 0.0,
        namespaces: list[str] | None = None,
        **agent_kwargs: Any,
    ) -> Any:
        """Build a ``FunctionAgent`` wired to a live per-turn gantry retriever."""
        from agent_gantry.integrations.frameworks.llamaindex_live import (
            _gantry_function_agent,
        )

        return _gantry_function_agent(
            self._gantry,
            llm,
            name=name,
            limit=self._default_limit if limit is None else limit,
            score_threshold=score_threshold,
            namespaces=namespaces,
            **agent_kwargs,
        )
