"""LlamaIndex *live* per-turn tool provider (deep integration).

This is the **deep**, per-turn counterpart to the static
:func:`~agent_gantry.integrations.frameworks.llamaindex.for_llamaindex`
helper. Where ``for_llamaindex`` selects a tool slice **once** and hands a
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
from agent_gantry.integrations.frameworks.llamaindex import spec_to_llamaindex
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


# ``ObjectRetriever`` must be available as a real base class at class-definition
# time. We resolve it lazily here so module import stays dependency-free; the
# first attribute access that touches the class triggers the import.
_ObjectRetriever = _import_object_retriever()


class GantryToolRetriever(_ObjectRetriever):  # type: ignore[valid-type,misc]
    """Per-turn ``ObjectRetriever`` that re-selects gantry tools every step.

    Drop this into ``FunctionAgent(tool_retriever=...)``. On every reasoning
    step the agent calls :meth:`aretrieve` (or :meth:`retrieve`), which runs a
    fresh semantic selection against the gantry for the current query and
    returns the matching tools converted to LlamaIndex ``FunctionTool`` objects.
    Each returned tool routes its invocation back through ``gantry.execute`` so
    retries, timeouts, circuit breakers, and the security policy still apply.

    Args:
        gantry: The :class:`~agent_gantry.core.gantry.AgentGantry` providing
            semantic retrieval and execution.
        limit: Maximum number of tools to expose per step. Defaults to ``5``.
        score_threshold: Minimum semantic relevance score. Defaults to ``0.0``
            (no filtering), matching the rest of the framework adapters.
    """

    def __init__(
        self,
        gantry: AgentGantry,
        *,
        limit: int = 5,
        score_threshold: float = 0.0,
    ) -> None:
        # Intentionally do NOT call super().__init__(): the base class wires a
        # static BaseRetriever + BaseObjectNodeMapping to retrieve over a
        # pre-indexed object set. We override retrieve/aretrieve entirely and
        # select live against the gantry, so that machinery is unused. See the
        # module docstring for the full rationale.
        self._gantry = gantry
        self._toolset = GantryToolset(gantry)
        self._limit = limit
        self._score_threshold = score_threshold

    # -- read-only accessors ------------------------------------------------ #
    @property
    def gantry(self) -> AgentGantry:
        return self._gantry

    @property
    def limit(self) -> int:
        return self._limit

    @property
    def score_threshold(self) -> float:
        return self._score_threshold

    # -- native ObjectRetriever hook ---------------------------------------- #
    async def aretrieve(self, str_or_query_bundle: Any) -> list:
        """Re-select tools for the current step and return ``FunctionTool``s.

        This is the per-turn hook ``FunctionAgent`` awaits each step.
        """
        query = _query_from(str_or_query_bundle)
        specs = await self._toolset.select(
            query, limit=self._limit, score_threshold=self._score_threshold
        )
        return [spec_to_llamaindex(spec) for spec in specs]

    def retrieve(self, str_or_query_bundle: Any) -> list:
        """Synchronous counterpart to :meth:`aretrieve`.

        Runs the same live selection from synchronous LlamaIndex call sites,
        bridging to the async path safely whether or not an event loop is
        already running on the current thread.
        """
        from agent_gantry.integrations.frameworks.base import _run_coroutine_sync

        return _run_coroutine_sync(self.aretrieve(str_or_query_bundle))


def gantry_tool_retriever(
    gantry: AgentGantry,
    *,
    limit: int = 5,
    score_threshold: float = 0.0,
) -> GantryToolRetriever:
    """Build a :class:`GantryToolRetriever` for ``gantry``.

    Raises:
        ImportError: If ``llama-index-core`` is not installed.
    """
    return GantryToolRetriever(
        gantry, limit=limit, score_threshold=score_threshold
    )


def gantry_function_agent(
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
            tool_retriever=gantry_tool_retriever(gantry, limit=limit, ...),
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
        tool_retriever=gantry_tool_retriever(
            gantry, limit=limit, score_threshold=score_threshold
        ),
        **agent_kwargs,
    )


__all__ = [
    "GantryToolRetriever",
    "gantry_tool_retriever",
    "gantry_function_agent",
]
