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

import functools
import logging
from typing import TYPE_CHECKING, Any

from agent_gantry.adapters.selectors.base import SelectorFallbackError
from agent_gantry.integrations.frameworks.base import (
    DEFAULT_TOOL_LIMIT,
    GantryToolset,
    QueryBoundsError,
    _resolve_pins,
    _with_native_names,
    check_query_bounds,
)
from agent_gantry.integrations.frameworks.errors import MissingRequiredToolError
from agent_gantry.integrations.frameworks.llamaindex import _spec_to_llamaindex
from agent_gantry.integrations.frameworks.native import (
    ErrorFormatter,
    ErrorMode,
    check_error_mode,
)
from agent_gantry.query import latest_activity

if TYPE_CHECKING:
    from agent_gantry.core.gantry import AgentGantry

logger = logging.getLogger(__name__)


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
            "install it with `pip install 'agent-gantry[llamaindex]'` (or "
            "`uv add 'agent-gantry[llamaindex]'`)."
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


def _build_retriever_class() -> type:
    """The ``GantryToolRetriever`` subclass of the installed ``ObjectRetriever``.

    Deferred so importing this module never requires ``llama-index``; built
    once per base class.
    """
    return _retriever_class(_import_object_retriever())


@functools.cache
def _retriever_class(object_retriever: type) -> type:
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
            limit: int = DEFAULT_TOOL_LIMIT,
            score_threshold: float = 0.0,
            namespaces: list[str] | None = None,
            required: list[str] | None = None,
            always_include: list[str] | None = None,
            error_mode: ErrorMode = "model",
            error_formatter: ErrorFormatter | None = None,
        ) -> None:
            # Intentionally do NOT call super().__init__(): the base class wires
            # a static BaseRetriever + BaseObjectNodeMapping over a pre-indexed
            # object set. We override retrieve/aretrieve entirely and select live
            # against the gantry, so that machinery is unused (see module docstring).
            check_query_bounds(
                limit=limit, score_threshold=score_threshold, owner="GantryToolRetriever"
            )
            self._gantry = gantry
            self._toolset = GantryToolset(gantry)
            self._limit = limit
            self._score_threshold = score_threshold
            self._namespaces = namespaces
            self._required = required
            self._always_include = always_include
            self._error_mode = check_error_mode(error_mode)
            self._error_formatter = error_formatter

        def _tools(self, specs: list[Any]) -> list:
            return [
                _spec_to_llamaindex(
                    spec, error_mode=self._error_mode, error_formatter=self._error_formatter
                )
                for spec in specs
            ]

        def _pins(self) -> list[Any]:
            """The pinned tools alone: they never depend on retrieval."""
            try:
                return _with_native_names(
                    self._gantry,
                    _resolve_pins(
                        self._gantry,
                        [],
                        required=self._required,
                        always_include=self._always_include,
                        namespaces=self._namespaces,
                    ),
                )
            except MissingRequiredToolError:
                raise
            except Exception:  # pragma: no cover - registry lookup failing too
                return []

        @property
        def gantry(self) -> AgentGantry:
            return self._gantry

        @property
        def limit(self) -> int:
            return self._limit

        @property
        def score_threshold(self) -> float:
            return self._score_threshold

        @property
        def namespaces(self) -> list[str] | None:
            return self._namespaces

        async def aretrieve(self, str_or_query_bundle: Any) -> list:
            """Re-select tools for the current step and return ``FunctionTool``s.

            Never raises on selection failure — a broken retrieval must not
            break the agent's step. ``FunctionAgent`` calls this with only the
            query (no run identity to keep earlier tools against), so on
            failure this logs a WARNING and offers the pinned tools alone,
            which never depend on retrieval — see "Per-turn selection-failure
            policy" in ``integrations/frameworks/README.md``.
            """
            query = _query_from(str_or_query_bundle)
            try:
                specs = await self._toolset.select_or_empty(
                    query,
                    limit=self._limit,
                    score_threshold=self._score_threshold,
                    namespaces=self._namespaces,
                    required=self._required,
                    always_include=self._always_include,
                )
            except (MissingRequiredToolError, QueryBoundsError, SelectorFallbackError):
                raise
            except Exception:
                logger.warning(
                    "GantryToolRetriever.aretrieve: semantic retrieval failed; "
                    "offering only the pinned tools this step.",
                    exc_info=True,
                )
                return self._tools(self._pins())
            return self._tools(specs)

        def retrieve(self, str_or_query_bundle: Any) -> list:
            """Synchronous counterpart to :meth:`aretrieve` (loop-safe bridge)."""
            from agent_gantry.integrations.frameworks.base import _run_coroutine_sync

            return _run_coroutine_sync(self.aretrieve(str_or_query_bundle))

    return GantryToolRetriever


def _gantry_tool_retriever(
    gantry: AgentGantry,
    *,
    limit: int = DEFAULT_TOOL_LIMIT,
    score_threshold: float = 0.0,
    namespaces: list[str] | None = None,
    required: list[str] | None = None,
    always_include: list[str] | None = None,
    error_mode: ErrorMode = "model",
    error_formatter: ErrorFormatter | None = None,
) -> Any:
    """Build a ``GantryToolRetriever`` for ``gantry``.

    Raises:
        ImportError: If ``llama-index-core`` is not installed.
    """
    return _build_retriever_class()(
        gantry,
        limit=limit,
        score_threshold=score_threshold,
        namespaces=namespaces,
        required=required,
        always_include=always_include,
        error_mode=error_mode,
        error_formatter=error_formatter,
    )


def _gantry_function_agent(
    gantry: AgentGantry,
    llm: Any,
    *,
    name: str = "gantry_agent",
    limit: int = DEFAULT_TOOL_LIMIT,
    score_threshold: float = 0.0,
    namespaces: list[str] | None = None,
    required: list[str] | None = None,
    always_include: list[str] | None = None,
    error_mode: ErrorMode = "model",
    error_formatter: ErrorFormatter | None = None,
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
        from agent_gantry.integrations.frameworks.llamaindex import gantry_function_agent_class

        FunctionAgent = gantry_function_agent_class()  # noqa: N806
    except ImportError as exc:  # pragma: no cover - exercised when absent
        raise ImportError(
            "LlamaIndex is required for LlamaIndexAdapter.function_agent; "
            "install it with `pip install 'agent-gantry[llamaindex]'` (or "
            "`uv add 'agent-gantry[llamaindex]'`)."
        ) from exc

    return FunctionAgent(
        name=name,
        llm=llm,
        tool_retriever=_gantry_tool_retriever(
            gantry,
            limit=limit,
            score_threshold=score_threshold,
            namespaces=namespaces,
            required=required,
            always_include=always_include,
            error_mode=error_mode,
            error_formatter=error_formatter,
        ),
        **agent_kwargs,
    )


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
