"""Live Pydantic AI toolsets backed by dotfetch.

Two concrete :class:`pydantic_ai.toolsets.AbstractToolset` implementations, both
built lazily on first use so ``import dotfetch`` never needs Pydantic AI:

``DotFetchToolset`` (per-turn push)
    ``get_tools`` runs on every model step. The query is derived from the run
    context -- the newest user prompt, or after a tool call the run's prompt
    plus the newest tool result -- so a chained task ("look up the customer,
    then email them") surfaces the next tool it needs mid-run. Selections are
    cached per run (one retrieval per distinct query, not per step) and, with
    ``sticky=True``, tools offered earlier in a run stay offered. Per-run state
    lives on the copy :meth:`for_run` returns, so concurrent runs sharing one
    toolset never see each other's selections.

``DotFetchDeferredToolset`` (search pull)
    ``get_tools`` returns the catalogue with ``defer_loading=True`` so Pydantic
    AI's ``ToolSearch`` capability hides it until the model searches. Paired
    with the dotfetch search strategy
    (:mod:`~dotfetch.integrations.frameworks.pydantic_ai_capability`), the
    model discovers tools by meaning while the tool list sent each turn stays
    stable -- which keeps provider prompt caches warm.

Both execute through :func:`~dotfetch.integrations.frameworks.pydantic_ai.execute_for_pydantic_ai`
(``fetcher.execute`` plus the ``ModelRetry``/``ToolFailed``/``ApprovalRequired``
mapping), and tag each ``ToolDefinition.metadata`` with dotfetch's identity under
``"dotfetch"`` so wrappers (``prefixed``, ``renamed``, ``filtered``) and the
search strategy can map a tool back to the registry.
"""

from __future__ import annotations

import copy
import functools
import json
import logging
from typing import TYPE_CHECKING, Any

from dotfetch.adapters.selectors.base import SelectorFallbackError
from dotfetch.integrations.frameworks.base import (
    DEFAULT_TOOL_LIMIT,
    QueryBoundsError,
    ToolExecutionError,
    ToolSpec,
    _resolve_pins,
    _with_native_names,
    check_query_bounds,
    spec_from_tool,
)
from dotfetch.integrations.frameworks.base import DotFetchToolset as _BaseToolset
from dotfetch.integrations.frameworks.errors import MissingRequiredToolError
from dotfetch.integrations.frameworks.pydantic_ai import (
    _INSTALL_HINT,
    METADATA_KEY,
    ErrorFormatter,
    ErrorMode,
    check_error_mode,
    execute_for_pydantic_ai,
    json_schema_for,
    registry_token,
)
from dotfetch.query import latest_activity

if TYPE_CHECKING:
    from dotfetch.core.fetcher import DotFetch

logger = logging.getLogger(__name__)

DEFAULT_TOOLSET_ID = "dotfetch"

#: Tool results can be large (JSON payloads, documents). Only the head is
#: useful as a retrieval signal, and embedders have input limits.
_MAX_TOOL_RESULT_QUERY_CHARS = 2000

#: Per-run selection cache size. Queries that include tool results rarely
#: repeat, so only recent ones are worth keeping; oldest entries are evicted.
_MAX_CACHED_SELECTIONS = 32


def _require_pydantic_ai() -> tuple[Any, Any, Any, Any]:
    """Import the Pydantic AI toolset machinery, with a helpful error if missing.

    Returns ``(AbstractToolset, ToolsetTool, ToolDefinition, (SchemaValidator,
    core_schema))``.
    """
    try:
        from pydantic_ai.tools import ToolDefinition
        from pydantic_ai.toolsets import AbstractToolset
        from pydantic_ai.toolsets.abstract import ToolsetTool
        from pydantic_core import SchemaValidator, core_schema
    except ImportError as exc:  # pragma: no cover - exercised via importorskip
        raise ImportError(_INSTALL_HINT) from exc
    return AbstractToolset, ToolsetTool, ToolDefinition, (SchemaValidator, core_schema)


# --------------------------------------------------------------------------- #
# Query derivation
# --------------------------------------------------------------------------- #


def _content_text(content: Any) -> str:
    """Render a prompt/part ``content`` (str or content sequence) as plain text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, (list, tuple)):
        parts = [c for c in (_content_text(item) for item in content) if c]
        return " ".join(parts).strip()
    if isinstance(content, dict):
        # JSON rather than Python repr: quotes and keys read as the model
        # and the embedder would see them.
        try:
            return json.dumps(content, ensure_ascii=False, default=str).strip()
        except (TypeError, ValueError):
            return str(content).strip()
    # Multimodal items (ImageUrl, BinaryContent, ...) carry no retrieval text.
    if type(content).__module__.startswith("pydantic_ai"):
        return ""
    return str(content).strip()


def _latest_request(messages: Any) -> tuple[str, str]:
    """``(user_text, tool_result_text)`` of the newest ``ModelRequest`` with either.

    Pydantic AI ``ModelMessage`` objects are not role-shaped (they hold
    ``parts``), so the generic :func:`~dotfetch.query.latest_activity`
    walker can't read them. Retry-prompt-only requests are skipped: a retry
    doesn't change what the run is about.
    """
    try:
        ordered = list(messages)
    except TypeError:
        return "", ""
    for message in reversed(ordered):
        if getattr(message, "kind", None) == "response":
            continue
        parts = getattr(message, "parts", None) or []
        user: list[str] = []
        results: list[str] = []
        for part in parts:
            kind = getattr(part, "part_kind", None)
            text = _content_text(getattr(part, "content", None))
            if not text:
                continue
            if kind == "user-prompt":
                user.append(text)
            elif kind == "tool-return":
                results.append(text[:_MAX_TOOL_RESULT_QUERY_CHARS])
        if user or results:
            return " ".join(user), " ".join(results)
    return "", ""


def _query_from_ctx(ctx: Any, *, include_results: bool = True) -> str:
    """Derive the per-step selection query from a Pydantic AI ``RunContext``.

    1. The newest request carries a user prompt -> that prompt (a new turn).
    2. It carries tool results -> the run's prompt plus those results, so the
       next selection reflects both the goal and what just happened. With
       ``include_results=False`` tool output never enters the query (and so
       never reaches the embedder); the run's prompt is used instead.
    3. Otherwise ``ctx.prompt``, then :func:`latest_activity` for role-shaped
       or dict-shaped histories some setups pass through.
    """
    messages = getattr(ctx, "messages", None) or []
    prompt = _content_text(getattr(ctx, "prompt", None))
    user_text, result_text = _latest_request(messages)
    if user_text:
        return user_text
    if result_text and include_results:
        return f"{prompt}\n{result_text}".strip()
    if prompt:
        return prompt
    return latest_activity(messages)


def _tools_used(ctx: Any) -> list[str]:
    """Native names of tools the model has called so far in this conversation."""
    used: dict[str, None] = {}
    for message in getattr(ctx, "messages", None) or []:
        if getattr(message, "kind", None) != "response":
            continue
        for part in getattr(message, "parts", None) or []:
            if getattr(part, "part_kind", None) == "tool-call":
                name = getattr(part, "tool_name", None)
                if name:
                    used[name] = None
    return list(used)


# --------------------------------------------------------------------------- #
# Toolset classes (built against the installed AbstractToolset)
# --------------------------------------------------------------------------- #


def _tool_definition(tool_definition_cls: Any, spec: ToolSpec, *, defer_loading: bool) -> Any:
    kwargs: dict[str, Any] = {
        "name": spec.native_name,
        "description": spec.description,
        "parameters_json_schema": json_schema_for(spec),
        "metadata": {
            METADATA_KEY: {
                "qualified_name": spec.qualified_name,
                "registry": registry_token(spec._fetcher),
                "namespace": spec._namespace,
                "score": spec.score,
                "requires_confirmation": spec.requires_confirmation,
            }
        },
    }
    if defer_loading:
        kwargs["defer_loading"] = True
    return tool_definition_cls(**kwargs)


def _build_toolset_class() -> type:
    """The ``DotFetchToolset`` subclass of the installed ``AbstractToolset``."""
    return _toolset_classes(_require_pydantic_ai()[0])[0]


def _build_deferred_toolset_class() -> type:
    """The ``DotFetchDeferredToolset`` subclass of the installed ``AbstractToolset``."""
    return _toolset_classes(_require_pydantic_ai()[0])[1]


@functools.cache
def _toolset_classes(abstract_toolset: type) -> tuple[type, type]:
    AbstractToolset, ToolsetTool, ToolDefinition, (SchemaValidator, core_schema) = (  # noqa: N806
        _require_pydantic_ai()
    )

    # Like ``Tool.from_schema``, argument validation is delegated to dotfetch: the
    # JSON schema is advertised verbatim and the model's arguments pass through
    # to ``fetcher.execute``, whose validation errors come back as ``ModelRetry``.
    passthrough_validator = SchemaValidator(schema=core_schema.any_schema())

    class _DotFetchToolsetBase(AbstractToolset):  # type: ignore[misc, valid-type]
        """Shared identity, tool wrapping and execution."""

        _toolset_id: str
        _error_mode: ErrorMode
        _error_formatter: ErrorFormatter | None
        _max_retries: int
        _label: str

        @property
        def id(self) -> str | None:
            """Stable identifier for this toolset within an agent."""
            return self._toolset_id

        @property
        def label(self) -> str:
            return f"{self._label} ({self._toolset_id})"

        @property
        def tool_name_conflict_hint(self) -> str:
            return (
                "Give each dotfetch toolset a distinct `namespaces=` filter, or wrap "
                "one with `.prefixed(...)`, and a unique `id=`."
            )

        def _wrap(self, spec: ToolSpec, *, defer_loading: bool = False) -> Any:
            tool = ToolsetTool(
                toolset=self,
                tool_def=_tool_definition(ToolDefinition, spec, defer_loading=defer_loading),
                max_retries=self._max_retries,
                args_validator=passthrough_validator,
            )
            tool.fetcher_spec = spec
            return tool

        async def _lookup(self, name: str, ctx: Any, tool: Any) -> ToolSpec | None:
            raise NotImplementedError

        async def call_tool(
            self,
            name: str,
            tool_args: dict[str, Any],
            ctx: Any,
            tool: Any,
        ) -> Any:
            """Execute the selected tool through ``fetcher.execute``.

            Uses the spec attached to ``tool`` by ``get_tools``; when a wrapper
            rebuilt the ``ToolsetTool`` without it, the spec is recovered from
            this run's state or a fresh lookup.
            """
            spec = getattr(tool, "fetcher_spec", None) or await self._lookup(name, ctx, tool)
            if spec is None:
                message = (
                    f"Tool {name!r} is not available in this step's dotfetch selection; "
                    "call one of the tools currently offered instead."
                )
                if self._error_mode == "model":
                    from pydantic_ai.exceptions import ModelRetry

                    raise ModelRetry(message)
                # The raise contract: every failure is a ToolExecutionError.
                raise ToolExecutionError(name, "failure", message, "ToolNotFound")
            return await execute_for_pydantic_ai(
                spec,
                tool_args,
                ctx,
                error_mode=self._error_mode,
                error_formatter=self._error_formatter,
            )

    class DotFetchToolset(_DotFetchToolsetBase):
        """A dotfetch-backed toolset that re-selects tools on every model step."""

        _label = "dotfetch per-turn selection"

        def __init__(
            self,
            fetcher: DotFetch,
            *,
            limit: int = DEFAULT_TOOL_LIMIT,
            score_threshold: float = 0.0,
            namespaces: list[str] | None = None,
            required: list[str] | None = None,
            always_include: list[str] | None = None,
            error_mode: ErrorMode = "model",
            id: str | None = None,  # noqa: A002
            sticky: bool = True,
            max_retries: int = 1,
            query_from_results: bool = True,
            max_offered: int | None = None,
            error_formatter: ErrorFormatter | None = None,
        ) -> None:
            check_query_bounds(limit=limit, score_threshold=score_threshold, owner="DotFetchToolset")
            if max_retries < 0:
                raise ValueError("max_retries must be >= 0")
            if max_offered is not None and max_offered < 1:
                raise ValueError("max_offered must be >= 1")
            self._toolset = _BaseToolset(fetcher)
            self._limit = limit
            self._score_threshold = score_threshold
            self._namespaces = namespaces
            self._required = required
            self._always_include = always_include
            self._error_mode = check_error_mode(error_mode)
            self._toolset_id = id or DEFAULT_TOOLSET_ID
            self._sticky = sticky
            self._max_retries = max_retries
            self._query_from_results = query_from_results
            self._error_formatter = error_formatter
            self._max_offered = 3 * limit if max_offered is None else max_offered
            # An explicit query override, used when driving selection without a
            # full RunContext (tests, manual selection). Copied into each run.
            self._query: str | None = None
            # Per-run state; ``None`` on the shared instance (stateless), set on
            # the copy ``for_run`` returns.
            self._cache: dict[tuple[str, tuple[str, ...]], list[ToolSpec]] | None = None
            self._offered: dict[str, ToolSpec] | None = None
            # native name -> step counter at which it was last selected; drives
            # eviction while ``_offered`` keeps a stable (insertion) order.
            self._last_selected: dict[str, int] = {}
            self._step = 0

        # -- per-run isolation ---------------------------------------------- #
        async def for_run(self, ctx: Any) -> DotFetchToolset:
            """A fresh copy holding this run's selection cache and sticky tools."""
            run = copy.copy(self)
            run._cache = {}
            run._offered = {}
            run._last_selected = {}
            run._step = 0
            return run

        # -- per-turn query override ---------------------------------------- #
        @property
        def query(self) -> str | None:
            """An explicit query override for subsequent selections."""
            return self._query

        @query.setter
        def query(self, value: str | None) -> None:
            self._query = value

        def set_query(self, query: str | None) -> DotFetchToolset:
            """Set an explicit selection query; returns ``self`` for chaining.

            When ``None`` (the default) the query is derived from the run
            context each step. Set it before starting runs: runs already in
            progress keep the value they started with.
            """
            self._query = query
            return self

        # -- dynamic tool discovery ----------------------------------------- #
        async def _select(self, ctx: Any) -> list[ToolSpec]:
            query = (
                self._query
                if self._query is not None
                else _query_from_ctx(ctx, include_results=self._query_from_results)
            )
            used = _tools_used(ctx)
            key = (query, tuple(used))
            if self._cache is not None and key in self._cache:
                return self._cache[key]
            specs = await self._toolset.select_or_empty(
                query,
                limit=self._limit,
                score_threshold=self._score_threshold,
                namespaces=self._namespaces,
                tools_already_used=used,
                required=self._required,
                always_include=self._always_include,
            )
            if self._cache is not None:
                if len(self._cache) >= _MAX_CACHED_SELECTIONS:
                    self._cache.pop(next(iter(self._cache)))
                self._cache[key] = specs
            return specs

        async def get_tools(self, ctx: Any) -> dict[str, Any]:
            """Select tools for this step and return their definitions.

            Never raises on retrieval failure -- a broken vector store must not
            break the agent's run; it logs a WARNING and offers this run's
            earlier (sticky) tools, or none. See "Per-turn selection-failure
            policy" in ``integrations/frameworks/README.md``.
            """
            try:
                specs = await self._select(ctx)
            except (MissingRequiredToolError, QueryBoundsError, SelectorFallbackError):
                raise
            except Exception:
                logger.warning(
                    "DotFetchToolset.get_tools: semantic retrieval failed; continuing "
                    "with this run's previously offered tools (if any) this step.",
                    exc_info=True,
                )
                specs = []
            if self._sticky and self._offered is not None:
                specs = self._carry_over(specs)
            return {spec.native_name: self._wrap(spec) for spec in specs}

        def _carry_over(self, selected: list[ToolSpec]) -> list[ToolSpec]:
            """This step's selection plus recent earlier ones, up to ``max_offered``.

            The current selection (pins included) is always offered in full;
            carried-over tools are evicted least recently selected first. The
            offered order stays stable across steps.
            """
            assert self._offered is not None
            self._step += 1
            current = set()
            for spec in selected:
                # Assignment refreshes a re-selected tool's spec (score, schema)
                # without moving it: an existing key keeps its position.
                self._offered[spec.native_name] = spec
                self._last_selected[spec.native_name] = self._step
                current.add(spec.native_name)
            # Only tools outside this step's selection can be evicted. Pins are in
            # every selection, so they are never dropped -- even if there are more
            # of them than ``max_offered``, which then caps only carried-over tools.
            evictable = sorted(
                (name for name in self._offered if name not in current),
                key=lambda name: self._last_selected.get(name, 0),
            )
            while len(self._offered) > self._max_offered and evictable:
                name = evictable.pop(0)
                del self._offered[name]
                self._last_selected.pop(name, None)
            return list(self._offered.values())

        async def _lookup(self, name: str, ctx: Any, tool: Any) -> ToolSpec | None:
            if self._offered and name in self._offered:
                return self._offered[name]
            try:
                specs = await self._select(ctx)
            except (MissingRequiredToolError, QueryBoundsError, SelectorFallbackError):
                raise
            except Exception:
                # Same policy as get_tools: a retrieval outage must not surface
                # as a raw exception from a tool call; the tool is reported as
                # unavailable (ModelRetry, or ToolNotFound with error_mode="raise").
                logger.warning(
                    "DotFetchToolset.call_tool: semantic retrieval failed while "
                    "recovering %r; reporting it as unavailable.",
                    name,
                    exc_info=True,
                )
                return None
            for spec in specs:
                if spec.native_name == name:
                    return spec
            return None

    class DotFetchDeferredToolset(_DotFetchToolsetBase):
        """The dotfetch catalogue as ``defer_loading`` tools for ``ToolSearch``."""

        _label = "dotfetch catalogue (deferred)"

        def __init__(
            self,
            fetcher: DotFetch,
            *,
            namespaces: list[str] | None = None,
            required: list[str] | None = None,
            always_include: list[str] | None = None,
            error_mode: ErrorMode = "model",
            id: str | None = None,  # noqa: A002
            max_retries: int = 1,
            error_formatter: ErrorFormatter | None = None,
        ) -> None:
            if max_retries < 0:
                raise ValueError("max_retries must be >= 0")
            self._error_formatter = error_formatter
            self._fetcher = fetcher
            self._namespaces = namespaces
            self._required = required
            self._always_include = always_include
            self._error_mode = check_error_mode(error_mode)
            self._toolset_id = id or DEFAULT_TOOLSET_ID
            self._max_retries = max_retries
            self._catalogue: dict[str, tuple[ToolSpec, bool]] | None = None
            # ``True`` on the copy ``for_run`` returns: build the catalogue once
            # per run. The shared instance rebuilds on every call.
            self._per_run = False

        async def for_run(self, ctx: Any) -> DotFetchDeferredToolset:
            """A copy that builds the catalogue once for this run."""
            run = copy.copy(self)
            run._catalogue = None
            run._per_run = True
            return run

        def _build_catalogue(self) -> dict[str, tuple[ToolSpec, bool]]:
            """``native_name -> (spec, deferred)`` for every catalogue tool."""
            tools = [
                tool
                for tool in self._fetcher.list_tools_sync()
                if not self._namespaces or tool.namespace in self._namespaces
            ]
            specs = _with_native_names(
                self._fetcher, [spec_from_tool(self._fetcher, t) for t in tools]
            )
            pinned = _with_native_names(
                self._fetcher,
                _resolve_pins(
                    self._fetcher,
                    [],
                    required=self._required,
                    always_include=self._always_include,
                    namespaces=self._namespaces,
                ),
            )
            visible = {spec.qualified_name for spec in pinned}
            catalogue = {s.native_name: (s, s.qualified_name not in visible) for s in specs}
            for spec in pinned:
                catalogue.setdefault(spec.native_name, (spec, False))
            return catalogue

        async def _current(self) -> dict[str, tuple[ToolSpec, bool]]:
            if self._per_run and self._catalogue is not None:
                return self._catalogue
            # Lazy ``modules=`` and late registrations only reach the registry
            # on sync; the search strategy needs them embedded anyway. A sync
            # failure degrades to the tools already registered.
            try:
                await self._fetcher.ensure_synced()
            except Exception:
                logger.warning(
                    "DotFetchDeferredToolset: sync failed; building the catalogue from "
                    "the tools already registered.",
                    exc_info=True,
                )
            catalogue = self._build_catalogue()
            if self._per_run:
                self._catalogue = catalogue
            return catalogue

        async def get_tools(self, ctx: Any) -> dict[str, Any]:
            """Every catalogue tool, deferred unless pinned visible."""
            return {
                name: self._wrap(spec, defer_loading=deferred)
                for name, (spec, deferred) in (await self._current()).items()
            }

        async def _lookup(self, name: str, ctx: Any, tool: Any) -> ToolSpec | None:
            catalogue = await self._current()
            entry = catalogue.get(name)
            if entry is not None:
                return entry[0]
            # A wrapper renamed the tool: recover it from the definition's metadata.
            meta = (getattr(getattr(tool, "tool_def", None), "metadata", None) or {}).get(
                METADATA_KEY
            ) or {}
            qualified = meta.get("qualified_name")
            for spec, _ in catalogue.values():
                if spec.qualified_name == qualified:
                    return spec
            return None

    return DotFetchToolset, DotFetchDeferredToolset


def _fetcher_toolset(
    fetcher: DotFetch,
    *,
    limit: int = DEFAULT_TOOL_LIMIT,
    score_threshold: float = 0.0,
    namespaces: list[str] | None = None,
    required: list[str] | None = None,
    always_include: list[str] | None = None,
    error_mode: ErrorMode = "model",
    id: str | None = None,  # noqa: A002
    sticky: bool = True,
    max_retries: int = 1,
    query_from_results: bool = True,
    max_offered: int | None = None,
    error_formatter: ErrorFormatter | None = None,
) -> Any:
    """Build a per-turn :class:`DotFetchToolset` (``Agent(model, toolsets=[ts])``).

    Raises:
        ImportError: If ``pydantic-ai`` is not installed.
    """
    return _build_toolset_class()(
        fetcher,
        limit=limit,
        score_threshold=score_threshold,
        namespaces=namespaces,
        required=required,
        always_include=always_include,
        error_mode=error_mode,
        id=id,
        sticky=sticky,
        max_retries=max_retries,
        query_from_results=query_from_results,
        max_offered=max_offered,
        error_formatter=error_formatter,
    )


def _fetcher_deferred_toolset(
    fetcher: DotFetch,
    *,
    namespaces: list[str] | None = None,
    required: list[str] | None = None,
    always_include: list[str] | None = None,
    error_mode: ErrorMode = "model",
    id: str | None = None,  # noqa: A002
    max_retries: int = 1,
    error_formatter: ErrorFormatter | None = None,
) -> Any:
    """Build a :class:`DotFetchDeferredToolset` for use with ``ToolSearch``.

    Raises:
        ImportError: If ``pydantic-ai`` is not installed.
    """
    return _build_deferred_toolset_class()(
        fetcher,
        namespaces=namespaces,
        required=required,
        always_include=always_include,
        error_mode=error_mode,
        id=id,
        max_retries=max_retries,
        error_formatter=error_formatter,
    )


def __getattr__(name: str) -> Any:
    """Expose the dynamically-built toolset classes lazily (for ``isinstance``)."""
    if name == "DotFetchToolset":
        return _build_toolset_class()
    if name == "DotFetchDeferredToolset":
        return _build_deferred_toolset_class()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
