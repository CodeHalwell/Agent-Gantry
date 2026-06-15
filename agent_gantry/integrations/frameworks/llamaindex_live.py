"""LlamaIndex *live* per-turn tool provider (deep integration).

This is the **deep**, per-turn counterpart to the static
:meth:`~agent_gantry.integrations.frameworks.llamaindex.LlamaIndexAdapter.select`
slice. Where the static slice selects a tool slice **once** and hands a
frozen list to the agent, this module re-selects the relevant tools **on every
reasoning step** by plugging into LlamaIndex's own native lifecycle hook.

The native hook
---------------
``llama_index.core.agent.workflow.FunctionAgent`` accepts a
``tool_retriever=<ObjectRetriever>``. On every step the agent calls
``await self.tool_retriever.aretrieve(query)`` (and the sync variant
``retrieve``) to fetch the tools to expose for *that* step — exactly the
re-selection point we want. We subclass
:class:`llama_index.core.objects.ObjectRetriever` and override
``retrieve`` / ``aretrieve`` so each call runs a fresh semantic selection
against the gantry and returns freshly converted ``FunctionTool`` objects.

Because we override both ``retrieve`` and ``aretrieve`` outright, we never use
the base class's ``retriever`` / ``object_node_mapping`` machinery (which is
designed for retrieving over a static, pre-indexed object set). We therefore
deliberately skip ``ObjectRetriever.__init__`` — there is no static index here,
the gantry *is* the index — while remaining a genuine ``ObjectRetriever``
subtype so ``FunctionAgent`` accepts it.

The ``llama_index`` import is lazy (performed inside the constructor / factory)
so that ``import agent_gantry`` never requires LlamaIndex to be installed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from agent_gantry.integrations.frameworks.base import GantryToolset
from agent_gantry.integrations.frameworks.llamaindex import _spec_to_llamaindex
from agent_gantry.query import latest_activity

if TYPE_CHECKING:
    from agent_gantry.core.gantry import AgentGantry


def _import_object_retriever() -> type:
    """Lazily import :class:`llama_index.core.objects.ObjectRetriever`.

    Raises:
        ImportError: If ``llama-index-core`` is not installed.
    """
    try:
        from llama_index.core.objects import ObjectRetriever
    except ImportError as exc:  # pragma: no cover - exercised when absent
        raise ImportError(
            "LlamaIndex is required for the live per-turn tool provider; "
            "install it with `pip install llama-index-core`."
        ) from exc
    return ObjectRetriever


def _query_from(str_or_query_bundle: Any) -> str:
    """Derive a query string from LlamaIndex's ``str | QueryBundle`` input.

    ``FunctionAgent`` passes the latest user/step text as a plain ``str``, but
    the ``ObjectRetriever`` contract also allows a ``QueryBundle``. Handle both
    (and a list of messages, via :func:`latest_activity`) so the retriever is
    robust regardless of how it's invoked.
    """
    if str_or_query_bundle is None:
        return ""
    if isinstance(str_or_query_bundle, str):
        return str_or_query_bundle
    # QueryBundle exposes ``query_str``.
    query_str = getattr(str_or_query_bundle, "query_str", None)
    if isinstance(query_str, str):
        return query_str
    # A list/iterable of chat messages — reuse the recency-aware generator.
    if isinstance(str_or_query_bundle, (list, tuple)):
        return latest_activity(str_or_query_bundle)
    return str(str_or_query_bundle)


_RETRIEVER_CLS: type | None = None


def _build_retriever_class() -> type:
    """Build (and cache) the ``GantryToolRetriever`` subclass of ``ObjectRetriever``.

    Deferred so importing this module never requires ``llama-index`` — the
    subclass (which needs ``ObjectRetriever`` as a real base) is only
    constructed when a retriever/agent is actually built. Mirrors the lazy
    class-build used by the AutoGen / Pydantic AI live providers.
    """
    global _RETRIEVER_CLS
    if _RETRIEVER_CLS is not None:
        return _RETRIEVER_CLS

    object_retriever = _import_object_retriever()

    class GantryToolRetriever(object_retriever):  # type: ignore[valid-type,misc]
        """Per-turn ``ObjectRetriever`` that re-selects gantry tools every step.

        Drop this into ``FunctionAgent(tool_retriever=...)``. On every reasoning
        step the agent calls :meth:`aretrieve` (or :meth:`retrieve`), which runs
        a fresh semantic selection against the gantry for the current query and
        returns the matching tools converted to LlamaIndex ``FunctionTool``
        objects. Each routes its invocation back through ``gantry.execute`` so
        retries, timeouts, circuit breakers, and the security policy apply.
        """

        def __init__(
            self,
            gantry: AgentGantry,
            *,
            limit: int = 5,
            score_threshold: float = 0.0,
        ) -> None:
            # Intentionally do NOT call super().__init__(): the base class wires
            # a static BaseRetriever + BaseObjectNodeMapping over a pre-indexed
            # object set. We override retrieve/aretrieve entirely and select live
            # against the gantry, so that machinery is unused (see module docstring).
            self._gantry = gantry
            self._toolset = GantryToolset(gantry)
            self._limit = limit
            self._score_threshold = score_threshold

        @property
        def gantry(self) -> AgentGantry:
            return self._gantry

        @property
        def limit(self) -> int:
            return self._limit

        @property
        def score_threshold(self) -> float:
            return self._score_threshold

        async def aretrieve(self, str_or_query_bundle: Any) -> list:
            """Re-select tools for the current step and return ``FunctionTool``s."""
            query = _query_from(str_or_query_bundle)
            if not (query or "").strip():
                # No retrieval signal: expose no tools rather than selecting on
                # an empty embedding (consistent with the other live providers).
                return []
            specs = await self._toolset.select(
                query, limit=self._limit, score_threshold=self._score_threshold
            )
            return [_spec_to_llamaindex(spec) for spec in specs]

        def retrieve(self, str_or_query_bundle: Any) -> list:
            """Synchronous counterpart to :meth:`aretrieve` (loop-safe bridge)."""
            from agent_gantry.integrations.frameworks.base import _run_coroutine_sync

            return _run_coroutine_sync(self.aretrieve(str_or_query_bundle))

    _RETRIEVER_CLS = GantryToolRetriever
    return _RETRIEVER_CLS


def _gantry_tool_retriever(
    gantry: AgentGantry,
    *,
    limit: int = 5,
    score_threshold: float = 0.0,
) -> Any:
    """Build a ``GantryToolRetriever`` for ``gantry``.

    Raises:
        ImportError: If ``llama-index-core`` is not installed.
    """
    return _build_retriever_class()(
        gantry, limit=limit, score_threshold=score_threshold
    )


def _gantry_function_agent(
    gantry: AgentGantry,
    llm: Any,
    *,
    name: str = "gantry_agent",
    limit: int = 5,
    score_threshold: float = 0.0,
    **agent_kwargs: Any,
) -> Any:
    """Build a ``FunctionAgent`` wired to a live per-turn gantry retriever.

    Equivalent to::

        FunctionAgent(
            name=name,
            llm=llm,
            tool_retriever=_gantry_tool_retriever(gantry, limit=limit, ...),
            **agent_kwargs,
        )

    Raises:
        ImportError: If ``llama-index-core`` is not installed.
    """
    try:
        from llama_index.core.agent.workflow import FunctionAgent
    except ImportError as exc:  # pragma: no cover - exercised when absent
        raise ImportError(
            "LlamaIndex is required for gantry_function_agent; "
            "install it with `pip install llama-index-core`."
        ) from exc

    return FunctionAgent(
        name=name,
        llm=llm,
        tool_retriever=_gantry_tool_retriever(
            gantry, limit=limit, score_threshold=score_threshold
        ),
        **agent_kwargs,
    )


__all__ = ["_gantry_tool_retriever", "_gantry_function_agent"]


def __getattr__(name: str) -> Any:
    """Expose the dynamically-built ``GantryToolRetriever`` subclass lazily.

    It subclasses ``llama_index.core.objects.ObjectRetriever``, so it's built on
    first access to keep ``import`` dependency-free. Internal/advanced use (e.g.
    ``isinstance`` checks); the public entry point is
    ``LlamaIndexAdapter.tool_retriever(...)``.
    """
    if name == "GantryToolRetriever":
        return _build_retriever_class()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
