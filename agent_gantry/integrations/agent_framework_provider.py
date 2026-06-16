"""
Microsoft Agent Framework context provider for Agent-Gantry.

``GantryContextProvider`` is the idiomatic, AF-native way to attach
Agent-Gantry to an :class:`agent_framework.Agent`. It plugs into the AF 1.2
context-engineering pipeline by subclassing
:class:`agent_framework.ContextProvider`, which means it co-operates by
construction with every other provider — most importantly with
:class:`agent_framework.SkillsProvider`. Each provider contributes its own
tools and instructions via ``SessionContext.extend_tools`` /
``extend_instructions`` keyed by a unique ``source_id``; AF merges them
without conflict.

Two retrieval modes are supported:

- ``query_strategy="per_run"`` (default, back-compatible): retrieval happens
  once at the start of each ``agent.run(...)``. Tool selection is fixed for
  the whole reasoning loop.
- ``query_strategy="per_call"``: retrieval re-runs on every chat-completion
  round. Use this for multi-step agents whose tool needs evolve as they
  reason. The provider exposes a chat middleware via
  :meth:`GantryContextProvider.as_chat_middleware` that the integrator
  attaches to the same agent — AF context providers cannot themselves hook
  per-round events, so the middleware is the canonical extension point.

Why a context provider rather than (only) the existing
:class:`~agent_gantry.integrations.agent_framework_bridge.GantryToolBridge`?
The bridge is excellent for *static* tool sets — pre-bake once, pass to
``Agent(tools=[...])``. The provider is for *dynamic* per-turn tool
selection: every ``agent.run(...)`` call, the latest user message is
matched against Gantry's semantic router and only the top-k tools are
injected. This minimises the tool-definition payload sent to the LLM,
which is the original design goal of Agent-Gantry.

Both APIs coexist intentionally:

.. code-block:: python

    # Static — known tool set, baked at construction:
    bridge = GantryToolBridge(gantry)
    tools = bridge.wrap_tools(my_defs)
    agent = Agent(client, "...", tools=tools)

    # Dynamic per-run — top-k driven by the initial user message:
    provider = GantryContextProvider(gantry, top_k=5)
    agent = Agent(client, "...", context_providers=[provider])

    # Dynamic per-call — top-k re-selected every chat round:
    from agent_gantry.query import last_tool_result, fallback_chain, last_user_text
    provider = GantryContextProvider(
        gantry,
        top_k=3,
        query_strategy="per_call",
        query_generator=fallback_chain(last_tool_result, last_user_text),
    )
    agent = Agent(
        client,
        "...",
        context_providers=[provider],
        middleware=[provider.as_chat_middleware()],
    )

The provider also flows transparently through every AF orchestration
primitive (``WorkflowBuilder``, ``SequentialBuilder``, ``HandoffBuilder``,
``AgentExecutor``, ``WorkflowAgent``) because they all dispatch through
``agent.run()``, which fires the provider pipeline.
"""

from __future__ import annotations

import inspect
import logging
from collections.abc import Callable
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any

from agent_gantry.integrations.agent_framework_bridge import (
    GantryToolBridge,
    RetrievalDecision,
)
from agent_gantry.query import (
    fallback_chain,
    last_tool_result,
    last_user_text,
)
from agent_gantry.utils.render import render_result

# Rolling cap on retained per-round retrieval decisions (see `selections`).
_MAX_SELECTION_HISTORY = 200

# Max kept tools shown in a single trace line before eliding the remainder.
_TRACE_SURFACED_CAP = 5

_DEFAULT_PER_RUN_QUERY = last_user_text


def _default_per_call_query() -> Any:
    """Recommended ``per_call`` default — tool result, then user text.

    Built lazily so the composed callable is fresh each time
    (``fallback_chain`` is cheap; the indirection just avoids a global).
    """
    return fallback_chain(last_tool_result, last_user_text)


if TYPE_CHECKING:
    from agent_gantry.core.gantry import AgentGantry
    from agent_gantry.integrations.anthropic_skills import SkillRegistry
    from agent_gantry.schema.tool import ToolDefinition

logger = logging.getLogger(__name__)


class MissingRequiredToolError(LookupError):
    """Raised when a tool listed in ``required=[...]`` is not present in the gantry."""


def _import_context_provider() -> type:
    """Lazily import :class:`agent_framework.ContextProvider`.

    Import is deferred to instantiation so the module remains importable
    in environments without ``agent-framework`` installed (e.g. unit
    tests for the rest of Gantry).
    """
    try:
        from agent_framework import ContextProvider
    except ImportError as exc:  # pragma: no cover - depends on install
        raise ImportError(
            "GantryContextProvider requires the 'agent-framework' package. "
            "Install with: pip install 'agent-gantry[agent-frameworks]'"
        ) from exc
    return ContextProvider


def _import_chat_middleware() -> Any:
    """Lazily import :func:`agent_framework.chat_middleware`."""
    try:
        from agent_framework import chat_middleware
    except ImportError as exc:  # pragma: no cover - depends on install
        raise ImportError(
            "GantryContextProvider.as_chat_middleware() requires the "
            "'agent-framework' package. Install with: "
            "pip install 'agent-gantry[agent-frameworks]'"
        ) from exc
    return chat_middleware


def _import_function_middleware() -> Any:
    """Lazily import :func:`agent_framework.function_middleware`."""
    try:
        from agent_framework import function_middleware
    except ImportError as exc:  # pragma: no cover - depends on install
        raise ImportError(
            "GantryContextProvider.trace() requires the 'agent-framework' "
            "package. Install with: pip install 'agent-gantry[agent-frameworks]'"
        ) from exc
    return function_middleware


def _tool_name(tool: Any) -> str:
    return (
        getattr(tool, "name", None)
        or getattr(tool, "__name__", None)
        or ""
    )


async def _maybe_await(value: Any) -> Any:
    """Await ``value`` if it's awaitable, else return it as-is."""
    if inspect.isawaitable(value):
        return await value
    return value


# Cache the synthesized impl class per AF base type. Keyed by the base class
# identity, so a test that swaps a stubbed ``ContextProvider`` for a different
# one gets its own entry rather than a stale subclass. (Hot-reloading the real
# AF base mid-process would return the cached subclass; not a concern in normal
# use, where the base is imported once.)
_IMPL_CLASS_CACHE: dict[type, type] = {}


def _build_impl_class(base: type) -> type:
    """Build (and cache) the concrete ``ContextProvider`` subclass.

    ``base`` is :class:`agent_framework.ContextProvider`, imported lazily, so
    the subclass cannot be declared at module top level. It holds no per-call
    state — every value is passed to ``__init__`` — so one class is built per
    base type and reused across all provider instances.
    """
    cached = _IMPL_CLASS_CACHE.get(base)
    if cached is not None:
        return cached

    class _GantryContextProviderImpl(base):  # type: ignore[misc,valid-type]
        """Concrete ContextProvider subclass; see :class:`GantryContextProvider`."""

        def __init__(
            self,
            *,
            gantry: AgentGantry,
            bridge: GantryToolBridge,
            top_k: int,
            score_threshold: float | str,
            query_strategy: str,
            query_generator: Callable[..., Any],
            skills: bool,
            skill_registry: SkillRegistry | None,
            always_include_effective: list[str],
            declared_always_include: list[str],
            required: list[str],
            static_tools: list[Any],
            query_kwargs: dict[str, Any],
            source_id: str,
            verbose: bool,
        ) -> None:
            super().__init__(source_id=source_id)
            self._gantry = gantry
            self._bridge = bridge
            self._top_k = top_k
            self._score_threshold = score_threshold
            self._query_strategy = query_strategy
            self._query_generator = query_generator
            self._skills_enabled = skills
            self._skill_registry = skill_registry
            self._always_include = always_include_effective
            self._declared_always_include = list(declared_always_include)
            self._required = list(required)
            self._static_tools = list(static_tools)
            self._query_kwargs = dict(query_kwargs)
            self._source_id = source_id
            self._verbose = verbose
            # Per-task run-state held in ContextVars (not plain attributes)
            # so a single provider shared across concurrent agent.run()
            # calls — e.g. a multi-user server — doesn't interleave its
            # selection history. The vars are per-instance (created here,
            # once per provider) so multiple providers stay independent.
            self._last_selection_var: ContextVar[RetrievalDecision | None] = (
                ContextVar("gantry_last_selection", default=None)
            )
            # Default None (not ()) so we can tell "never set in this
            # context" (-> fall back to the plain snapshot) apart from
            # "reset for a fresh run, nothing retrieved yet" (-> return ()
            # without leaking the previous run's history).
            self._selections_var: ContextVar[tuple[RetrievalDecision, ...] | None] = (
                ContextVar("gantry_selections", default=None)
            )
            # Plain-attribute fallbacks. A run driven inside a *child* task
            # (AF may wrap agent.run() in asyncio.create_task, which copies
            # the context) writes the ContextVar in that copy — invisible to
            # the caller's context, which would make post-run
            # `provider.selections` reads silently empty. So we also stash
            # the latest *coherent* snapshot here (last-writer-wins, never
            # interleaved — it mirrors one task's ContextVar value) and read
            # it only when the context-local value is unset.
            self._last_selection_plain: RetrievalDecision | None = None
            self._selections_plain: tuple[RetrievalDecision, ...] = ()
            # Trace round counter, allocated once per provider (not per
            # trace() call) — ContextVars aren't GC'd, so a fresh one per
            # call would leak in a server that builds the middleware
            # repeatedly. Context-scoped so concurrent runs count
            # independently.
            self._trace_round_var: ContextVar[int] = ContextVar(
                "gantry_trace_round", default=0
            )
            # One-shot warnings: we don't want to spam the log every
            # round even when the misconfiguration persists.
            self._warned_about_missing_chat_middleware = False
            self._logged_top_k_math = False
            if verbose:
                # Don't override a user-configured level; only nudge
                # when the logger has no explicit level set.
                if logger.level == logging.NOTSET:
                    logger.setLevel(logging.INFO)

        # ----- Public, read-only configuration accessors --------------
        @property
        def top_k(self) -> int:
            return self._top_k

        @property
        def score_threshold(self) -> float | str:
            return self._score_threshold

        @property
        def query_strategy(self) -> str:
            return self._query_strategy

        @property
        def always_include(self) -> tuple[str, ...]:
            return tuple(self._declared_always_include)

        @property
        def required(self) -> tuple[str, ...]:
            return tuple(self._required)

        @property
        def gantry(self) -> AgentGantry:
            return self._gantry

        @property
        def bridge(self) -> GantryToolBridge:
            return self._bridge

        @property
        def static_tools(self) -> tuple[Any, ...]:
            return tuple(self._static_tools)

        @property
        def last_selection(self) -> RetrievalDecision | None:
            """The most recent retrieval decision, or ``None``.

            Cleared/replaced on every ``before_run`` (per_run mode) or
            every chat-middleware tick (per_call mode). Read this from
            middleware, tests, or interactive sessions to diagnose
            "why did the LLM not see tool X this round?".

            Backed by a :class:`~contextvars.ContextVar`: the value is
            scoped to the current task/context, so concurrent
            ``agent.run()`` calls on a shared provider don't clobber each
            other's in-context reads. When the context-local value is unset
            — e.g. a post-run read from the parent task after AF ran the
            agent in a child task whose context copy we can't see — this
            falls back to the latest coherent snapshot so introspection
            never goes silently empty.
            """
            in_context = self._last_selection_var.get()
            return in_context if in_context is not None else self._last_selection_plain

        @property
        def selections(self) -> tuple[RetrievalDecision, ...]:
            """Per-round retrieval history, oldest first (bounded window).

            ``last_selection`` is a single most-recent slot, so reading it
            from *function* middleware is inherently laggy — it holds
            whatever the latest chat round selected, which is not guaranteed
            to be the round that produced the call you're handling.
            ``selections`` keeps the full per-round sequence (capped at
            ``_MAX_SELECTION_HISTORY``) so a trace or audit can correlate
            "what was surfaced" with "what the model then called" across the
            whole run instead of just the last round.

            Scoped to a single ``agent.run`` — :meth:`before_run` resets the
            history at the start of each run, so a shared provider doesn't
            bleed one run's (or one user's) selections into the next. (This
            differs from :attr:`last_selection`, which is the single
            latest decision and is not reset.) Task-scoped via a
            ``ContextVar`` — each concurrent run accumulates its own history
            rather than interleaving into one shared list — with a
            plain-attribute fallback so a post-run read from a different
            task still sees the latest run's (coherent, non-interleaved)
            history instead of an empty tuple.
            """
            in_context = self._selections_var.get()
            return in_context if in_context is not None else self._selections_plain

        # ----- AF lifecycle ------------------------------------------
        async def before_run(
            self,
            *,
            agent: Any,
            session: Any,
            context: Any,
            state: dict[str, Any],
        ) -> None:
            # Run boundary: scope the selection history to this run so a
            # shared provider doesn't carry the previous run's (or user's)
            # selections forward. Reset both the context-local var and the
            # cross-task plain snapshot; per_run appends below, per_call
            # appends via the chat middleware. last_selection is the single
            # latest decision and is deliberately not reset.
            self._selections_var.set(())
            self._selections_plain = ()

            # per_call mode requires the chat middleware to do the
            # actual per-round refresh. If the user forgot to attach
            # it, the agent silently degrades to per_run behaviour
            # with extra plumbing — warn once so the misconfiguration
            # is visible without strangers reading the source.
            if (
                self._query_strategy == "per_call"
                and not self._warned_about_missing_chat_middleware
                and not self._chat_middleware_attached(agent)
            ):
                self._warned_about_missing_chat_middleware = True
                logger.warning(
                    "GantryContextProvider: query_strategy='per_call' is "
                    "enabled but as_chat_middleware() does not appear to "
                    "be attached to the agent — the provider will fall "
                    "back to per_run behaviour. Pass "
                    "middleware=[provider.as_chat_middleware()] to your "
                    "Agent(...) or use provider.attach_to(agent)."
                )

            # In per_call mode, the chat_middleware does the dynamic
            # retrieval per LLM round; we still inject always_include /
            # skill tools at run-start so they are present even if the
            # middleware is not attached.
            tools, _ = await self._collect_tools(
                context.input_messages,
                include_dynamic=(self._query_strategy == "per_run"),
            )
            if tools:
                context.extend_tools(self._source_id, tools)
                logger.debug(
                    "GantryContextProvider: injected %d tools at run start "
                    "(strategy=%s)",
                    len(tools),
                    self._query_strategy,
                )

        # ----- Setup helpers -----------------------------------------
        def attach_to(self, agent: Any, *, trace: bool = False) -> Any:
            """Register this provider and its middleware on ``agent``.

            Single-call helper for the common ``per_call`` setup. Does
            the dance of appending to ``agent.context_providers`` and
            ``agent.middleware`` (whichever attribute the AF version
            exposes), creating the lists if missing. Returns
            ``agent`` for chaining.

            The chat middleware is only attached when
            ``query_strategy='per_call'`` — in ``per_run`` mode it's
            a no-op and would just add overhead. The context
            provider is always attached.

            Args:
                agent: The AF agent to attach to.
                trace: When ``True``, also attach :meth:`trace` (a function
                    middleware) so every tool call prints a readable
                    per-round line — the built-in replacement for
                    hand-rolled trace glue. Defaults to ``False``.
            """
            # context_providers
            providers_attr = self._first_present_attr(
                agent, ("context_providers", "_context_providers")
            )
            if providers_attr is not None:
                current = getattr(agent, providers_attr) or []
                if self not in current:
                    try:
                        setattr(agent, providers_attr, [*current, self])
                    except (AttributeError, TypeError):
                        try:
                            current.append(self)
                        except Exception:
                            logger.warning(
                                "GantryContextProvider.attach_to: could "
                                "not attach context provider to agent %r.",
                                type(agent).__name__,
                            )
            else:
                logger.warning(
                    "GantryContextProvider.attach_to: agent %r has no "
                    "context_providers attribute; the provider was not "
                    "attached.",
                    type(agent).__name__,
                )

            # middleware: per-call chat refresher (per_call only) plus the
            # console trace (when requested). Both live in AF's single
            # `middleware` list and attach together.
            to_add: list[Any] = []
            if self._query_strategy == "per_call":
                to_add.append(self.as_chat_middleware())
            if trace:
                to_add.append(self.trace())

            if to_add:
                middleware_attr = self._first_present_attr(
                    agent, ("middleware", "_middleware")
                )
                if middleware_attr is not None:
                    current_mw = getattr(agent, middleware_attr) or []
                    try:
                        setattr(agent, middleware_attr, [*current_mw, *to_add])
                    except (AttributeError, TypeError):
                        try:
                            for mw in to_add:
                                current_mw.append(mw)
                        except Exception:
                            logger.warning(
                                "GantryContextProvider.attach_to: could "
                                "not attach middleware to agent %r.",
                                type(agent).__name__,
                            )
                else:
                    logger.warning(
                        "GantryContextProvider.attach_to: agent %r has "
                        "no middleware attribute; middleware was not "
                        "attached.",
                        type(agent).__name__,
                    )
            return agent

        # ----- Console trace middleware ------------------------------
        def trace(
            self,
            *,
            render: bool = True,
            printer: Callable[[str], None] = print,
            limit: int = 200,
        ) -> Any:
            """Return an AF *function* middleware that prints a per-call trace.

            This is the library-owned replacement for the hand-rolled
            ``trace_tool_calls`` + ``_preview`` glue. For every tool the
            agent invokes it prints one line before the call —
            ``>>> round N: tool(args)  [surfaced: name:score, …]`` — using
            :attr:`last_selection` for the surfaced set, and (when
            ``render`` is ``True``) one line after with a short preview of
            the result via :func:`~agent_gantry.render_result`.

            Attach it alongside the provider, either via
            ``provider.attach_to(agent, trace=True)`` or by adding
            ``provider.trace()`` to ``Agent(middleware=[...])``.

            Cheap to call (the round counter lives on the provider, not the
            returned middleware), but attach just one trace middleware per
            agent — multiple share the provider's counter and would
            interleave round numbers.

            Args:
                render: Print the post-call result preview line. Default
                    ``True``.
                printer: Sink for the rendered lines. Defaults to
                    :func:`print`; pass ``logger.info`` or a custom callable
                    to redirect.
                limit: Max characters of the result preview before
                    truncation. Default ``200``.
            """
            function_middleware_decorator = _import_function_middleware()
            provider = self
            # Per-provider round counter (allocated in __init__, not here),
            # context-scoped so concurrent runs count independently without
            # a shared closure dict and without leaking a ContextVar per
            # trace() call.
            round_var = self._trace_round_var

            @function_middleware_decorator
            async def _gantry_trace(context: Any, call_next: Any) -> None:
                n = round_var.get() + 1
                round_var.set(n)
                surfaced = ""
                selection = provider.last_selection
                if selection is not None:
                    # Show only the tools that actually made the surface
                    # (kept), capped — `candidates` is the full ranked list
                    # incl. threshold-dropped entries, which would mislabel
                    # and bloat the line.
                    kept = selection.kept
                    shown = kept[:_TRACE_SURFACED_CAP]
                    rendered = ", ".join(f"{c.name}:{c.score:.2f}" for c in shown)
                    if len(kept) > len(shown):
                        rendered += f", +{len(kept) - len(shown)} more"
                    surfaced = f"  [surfaced: {rendered}]"
                name = getattr(getattr(context, "function", None), "name", "?")
                try:
                    args: Any = dict(context.arguments)
                except (TypeError, ValueError):
                    args = getattr(context, "arguments", None)
                printer(f">>> round {n}: {name}({args}){surfaced}")
                await call_next()
                if render:
                    preview = render_result(
                        getattr(context, "result", None),
                        limit=limit,
                        collapse_whitespace=True,
                    )
                    printer(f"<<< round {n}: {name} -> {preview}")

            return _gantry_trace

        @staticmethod
        def _first_present_attr(obj: Any, names: tuple[str, ...]) -> str | None:
            for n in names:
                if hasattr(obj, n):
                    return n
            return None

        def _chat_middleware_attached(self, agent: Any) -> bool:
            """Best-effort check for ``as_chat_middleware`` on the agent.

            Heuristic: look for any middleware whose qualified name
            contains the provider's sentinel ``_per_call_retrieval``.
            AF wraps the decorated coroutine into different objects
            across versions, so we just match on the name.
            """
            if agent is None:
                return True  # don't warn when there's nothing to inspect
            mw_attr = self._first_present_attr(
                agent, ("middleware", "_middleware")
            )
            if mw_attr is None:
                return True
            middlewares = getattr(agent, mw_attr) or []
            for m in middlewares:
                candidates = (
                    getattr(m, "__name__", None),
                    getattr(m, "__qualname__", None),
                    getattr(getattr(m, "func", None), "__name__", None),
                    type(m).__name__,
                )
                for c in candidates:
                    if isinstance(c, str) and "_per_call_retrieval" in c:
                        return True
            return False

        # ----- Per-call middleware factory ---------------------------
        def as_chat_middleware(self) -> Any:
            """Return an AF chat_middleware that refreshes tools each round.

            Use with ``Agent(middleware=[provider.as_chat_middleware()])``
            in conjunction with ``query_strategy="per_call"``. The
            returned middleware reads ``context.messages`` (or
            ``context.options`` as appropriate), runs the configured
            query generator, retrieves a fresh top-k from the gantry,
            and updates the tool list before the LLM call. If retrieval
            fails it logs the exception and leaves the existing tools
            unchanged — retrieval must never break the agent run.
            """
            chat_middleware_decorator = _import_chat_middleware()
            provider = self

            @chat_middleware_decorator
            async def _per_call_retrieval(context: Any, call_next: Any) -> None:
                try:
                    await provider._refresh_tools_on_chat_context(context)
                except Exception:
                    logger.exception(
                        "GantryContextProvider: per-call retrieval failed; "
                        "continuing with existing tools."
                    )
                await call_next()

            return _per_call_retrieval

        async def _refresh_tools_on_chat_context(self, context: Any) -> None:
            """Mutate ``context.options['tools']`` for the current round.

            Strategy: drop **all** tools whose names are known to the
            gantry registry (these are tools we — or another provider
            bridging the same gantry — could have injected). Anything
            else in ``existing`` is foreign (skills from another
            provider, AF-native tools added at construction time) and
            is preserved. Then append the freshly retrieved top-k plus
            always_include / skill pins. This guarantees the per-round
            top-k stays bounded and stale dynamic selections from
            earlier rounds do not accumulate.

            Mutates the existing options dict **in place** rather than
            replacing the reference. ``FunctionInvocationLayer`` in
            agent-framework keeps the same options dict across its
            inner loop and uses it both as the chat-call payload *and*
            as the tool-lookup table when executing function calls.
            Reassigning ``context.options = new_options`` would only
            update the chat-call payload — the function executor would
            still see the original (stale) tool list and fail to find
            the tools we just injected.
            """
            options = getattr(context, "options", None)
            existing: list[Any] = []
            if isinstance(options, dict):
                existing = list(options.get("tools") or [])
            elif options is not None:
                # Pydantic ChatOptions / plain objects: peer-provider
                # tools live on the `tools` attribute. Read them too,
                # otherwise the combined list would drop everything
                # the model already carried (skills, static tools,
                # tools from another ContextProvider).
                attr_tools = getattr(options, "tools", None)
                if attr_tools:
                    existing = list(attr_tools)

            messages = (
                getattr(context, "messages", None)
                or getattr(context, "input_messages", None)
                or []
            )
            fresh, _ = await self._collect_tools(
                messages, include_dynamic=True
            )
            # _collect_tools updates the last-selection / selections
            # ContextVars on every dynamic refresh, so the per-call
            # middleware path automatically exposes the latest decision via
            # the provider's `last_selection` / `selections` properties.
            gantry_names = self._all_known_tool_names()
            static_names = {
                _tool_name(t) for t in self._static_tools if _tool_name(t)
            }

            preserved = []
            dropped = 0
            for t in existing:
                name = _tool_name(t)
                # Drop stale gantry-known tools so the per-round
                # surface stays bounded. Static tools may appear in
                # both `existing` (carried across rounds) and `fresh`
                # (we re-add them ourselves); dedup happens below.
                if name and name in gantry_names and name not in static_names:
                    dropped += 1
                    continue
                preserved.append(t)

            combined: list[Any] = []
            seen_combined: set[str] = set()
            for t in preserved + fresh:
                name = _tool_name(t)
                if name and name in seen_combined:
                    continue
                if name:
                    seen_combined.add(name)
                combined.append(t)

            if isinstance(options, dict):
                # Mutate in place — see docstring above.
                options["tools"] = combined
            elif options is not None:
                # Non-dict options (e.g. a Pydantic ChatOptions model).
                # The in-place invariant from the docstring applies here
                # too — AF's FunctionInvocationLayer keeps the same
                # options reference across the inner loop, so we must
                # mutate the existing object rather than reassign
                # context.options. Try setattr first (works for plain
                # objects and mutable Pydantic models); only fall back
                # to a rebuilt-copy + reassignment if the attribute is
                # genuinely immutable (frozen model).
                mutated_in_place = False
                try:
                    setattr(options, "tools", combined)
                    mutated_in_place = True
                except (AttributeError, TypeError, ValueError):
                    pass
                if not mutated_in_place:
                    if hasattr(options, "model_copy"):
                        new_options = options.model_copy(update={"tools": combined})
                        try:
                            context.options = new_options
                        except AttributeError:
                            logger.warning(
                                "GantryContextProvider: cannot update tools on "
                                "immutable options object %r (no in-place setattr, "
                                "no settable context.options). %d Gantry tools were "
                                "selected but will NOT be visible to AF.",
                                type(options).__name__,
                                len(combined),
                            )
                    else:
                        logger.warning(
                            "GantryContextProvider: options object %r is neither "
                            "dict-like nor a Pydantic model and rejected attribute "
                            "assignment. %d Gantry tools were selected but will "
                            "NOT be visible to AF.",
                            type(options).__name__,
                            len(combined),
                        )

            logger.debug(
                "GantryContextProvider: refreshed tools per-call "
                "(%d preserved + %d gantry, %d stale dropped, %d total)",
                len(preserved),
                len(fresh),
                dropped,
                len(combined),
            )

        def _all_known_tool_names(self) -> set[str]:
            """Return every tool name registered in the gantry.

            Used to identify which entries in ``context.options['tools']``
            were produced by this gantry (or any other provider sharing
            it) so they can be dropped before re-injecting the fresh
            per-round selection.
            """
            registry = getattr(self._gantry, "_registry", None)
            if registry is None:
                return set()
            lister = getattr(registry, "list_tools", None)
            if not callable(lister):
                return set()
            try:
                return {t.name for t in lister() if getattr(t, "name", None)}
            except Exception:
                return set()

        # ----- Tool assembly -----------------------------------------
        async def _collect_tools(
            self,
            messages: Any,
            *,
            include_dynamic: bool,
        ) -> tuple[list[Any], set[str]]:
            tools: list[Any] = []
            seen: set[str] = set()
            decision: RetrievalDecision | None = None

            if include_dynamic:
                query = await self._build_query(messages)
                if query:
                    try:
                        retrieved, decision = (
                            await self._bridge.get_tools_with_decision(
                                query,
                                limit=self._top_k,
                                score_threshold=self._score_threshold,
                                **self._query_kwargs,
                            )
                        )
                    except Exception:
                        logger.exception(
                            "GantryContextProvider: semantic retrieval failed; "
                            "continuing without dynamic tools."
                        )
                        retrieved = []
                    for t in retrieved:
                        name = _tool_name(t)
                        if name and name in seen:
                            continue
                        if name:
                            seen.add(name)
                        tools.append(t)

            # Skills: always-on tools bound to registered Gantry skills.
            skill_count = 0
            if self._skills_enabled:
                skill_tool_names = self._collect_skill_tool_names()
                extra = self._wrap_named(skill_tool_names, seen, source="skill")
                tools.extend(extra)
                skill_count = len(extra)

            # Explicit always_include / required pins.
            always_count = 0
            if self._always_include:
                extra = self._wrap_named(
                    self._always_include, seen, source="always_include"
                )
                tools.extend(extra)
                always_count = len(extra)

            # Static AF-native tools that live outside the gantry registry.
            static_count = 0
            for t in self._static_tools:
                name = _tool_name(t)
                if name and name in seen:
                    continue
                if name:
                    seen.add(name)
                tools.append(t)
                static_count += 1

            if include_dynamic and decision is not None:
                self._last_selection_var.set(decision)
                # ContextVar holds an immutable tuple; bound it by slicing
                # so history stays capped without a shared mutable deque.
                # `or ()` covers a refresh in a context where before_run
                # never reset the var (default None).
                history = (
                    (self._selections_var.get() or ()) + (decision,)
                )[-_MAX_SELECTION_HISTORY:]
                self._selections_var.set(history)
                # Mirror to the plain fallbacks (coherent snapshot for
                # cross-task post-run introspection; see the properties).
                self._last_selection_plain = decision
                self._selections_plain = history
                if self._verbose:
                    logger.info("gantry: %s", decision.summary())

            # One-shot info log clarifying the top_k math whenever
            # non-dynamic contributions raise the surface above top_k.
            extra_total = skill_count + always_count + static_count
            if (
                include_dynamic
                and extra_total > 0
                and not self._logged_top_k_math
            ):
                self._logged_top_k_math = True
                logger.info(
                    "GantryContextProvider: dynamic top_k=%d + %d preserved "
                    "(skills=%d, always_include=%d, static=%d). The final "
                    "tool surface is the dynamic slice plus preserved tools.",
                    self._top_k,
                    extra_total,
                    skill_count,
                    always_count,
                    static_count,
                )

            return tools, seen

        async def dry_run_retrieve(
            self,
            query: str,
            *,
            limit: int | None = None,
            score_threshold: float | str | None = None,
        ) -> RetrievalDecision:
            """Run the live retrieval path against a synthetic query.

            Uses the *exact same* threshold, top_k, and ``query_kwargs``
            as the live middleware. Use this to diagnose why a tool
            is or is not surfacing without spinning up an agent.

            Args:
                query: The query string to retrieve against.
                limit: Override ``top_k`` for this call only.
                score_threshold: Override the threshold for this call
                    only (accepts the same forms as the constructor).

            Returns:
                The :class:`RetrievalDecision` produced by the bridge.
            """
            eff_limit = self._top_k if limit is None else limit
            eff_threshold = (
                self._score_threshold
                if score_threshold is None
                else score_threshold
            )
            _, decision = await self._bridge.get_tools_with_decision(
                query,
                limit=eff_limit,
                score_threshold=eff_threshold,
                **self._query_kwargs,
            )
            return decision

        async def _build_query(self, messages: Any) -> str:
            try:
                result = self._query_generator(messages)
                return await _maybe_await(result) or ""
            except Exception:
                logger.exception(
                    "GantryContextProvider: query_generator raised; falling "
                    "back to last_user_text."
                )
                return last_user_text(messages)

        def _resolve_skill_registry(self) -> SkillRegistry | None:
            if self._skill_registry is not None:
                return self._skill_registry
            # Best-effort: pick up a registry attached to the gantry
            # instance (e.g. via SkillsClient).
            return getattr(self._gantry, "_skill_registry", None) or getattr(
                self._gantry, "skill_registry", None
            )

        def _collect_skill_tool_names(self) -> list[str]:
            registry = self._resolve_skill_registry()
            if registry is None:
                logger.debug(
                    "GantryContextProvider: skills=True but no SkillRegistry "
                    "available on gantry — no skill tools injected."
                )
                return []
            names: list[str] = []
            seen_names: set[str] = set()
            for skill in registry.list_skills():
                for tool_name in skill.tools or []:
                    if tool_name not in seen_names:
                        seen_names.add(tool_name)
                        names.append(tool_name)
            return names

        def _wrap_named(
            self, names: list[str], seen: set[str], *, source: str
        ) -> list[Any]:
            wrapped: list[Any] = []
            missing: list[str] = []
            for name in names:
                if name in seen:
                    continue
                tool_def = self._lookup_tool_def(name)
                if tool_def is None:
                    missing.append(name)
                    continue
                wrapped.append(self._bridge.wrap_single(tool_def))
                seen.add(name)
            if missing:
                logger.warning(
                    "GantryContextProvider: %s tool(s) not found in registry "
                    "and will be skipped: %s",
                    source,
                    missing,
                )
            return wrapped

        def _lookup_tool_def(self, name: str) -> ToolDefinition | None:
            registry = getattr(self._gantry, "_registry", None)
            if registry is None:
                return None
            lookup = getattr(registry, "get_tool_by_name", None)
            if callable(lookup):
                found = lookup(name)
                if found is not None:
                    return found
            lookup = getattr(registry, "get_tool", None)
            if callable(lookup):
                return lookup(name)
            return None

    _IMPL_CLASS_CACHE[base] = _GantryContextProviderImpl
    return _GantryContextProviderImpl


class GantryContextProvider:
    """AF context provider that injects semantically-selected Gantry tools.

    Constructing the provider returns an instance of the dynamically
    generated ``agent_framework.ContextProvider`` subclass. The factory
    pattern keeps the import of ``agent_framework`` lazy so importing
    ``agent_gantry`` doesn't require AF to be installed.

    Args:
        gantry: The :class:`~agent_gantry.core.gantry.AgentGantry` instance
            providing semantic retrieval and execution.
        top_k: Maximum number of dynamically retrieved tools per chat-completion
            round (or per ``agent.run`` call in ``per_run`` mode). Defaults
            to ``5``. Note: ``top_k`` governs only the *dynamic* selection.
            Tools contributed by skills, by ``always_include`` / ``required``,
            and by ``static_tools`` are appended on top of the ``top_k``
            slice. A user setting ``top_k=6`` with one skill and one
            always-include tool will see ``6 + 2 = 8`` tools.
        score_threshold: Minimum semantic relevance score for retrieved
            tools. Defaults to ``0.0`` (no filtering). Long queries dilute
            absolute cosine similarities, so the previous ``0.3`` default
            silently filtered relevant tools on multi-step pipelines.
            Accepts either a ``float`` (absolute cosine cutoff) or the
            string ``"relative:<frac>"`` (e.g. ``"relative:0.8"`` retains
            anything within 80% of the top score) for length-robust
            filtering.
        query_strategy: ``"per_run"`` (default) refreshes the tool selection
            once at the start of ``agent.run()``. ``"per_call"`` re-runs
            retrieval on every chat-completion round; pair this mode with
            :meth:`as_chat_middleware` to install the per-round refresher.
        query_generator: Callable used to derive the retrieval query string
            from the conversation messages. When ``None`` the default
            depends on ``query_strategy``: ``last_user_text`` for
            ``per_run`` (back-compat), and ``fallback_chain(last_tool_result,
            last_user_text)`` for ``per_call`` — the latter is what makes
            ``per_call`` actually adapt across rounds. Sync or async are
            both accepted. See :mod:`agent_gantry.query` for the built-in
            alternatives.
        skills: When ``True``, every invocation also injects the union of
            tools bound to all registered Gantry skills (from the supplied
            ``skill_registry`` if any, else the registry attached to the
            gantry instance via :class:`SkillsClient`). These tools are
            *always* included regardless of the semantic match. Defaults
            to ``False``.
        skill_registry: Optional explicit
            :class:`~agent_gantry.integrations.anthropic_skills.SkillRegistry`.
            When ``None`` and ``skills=True``, the provider attempts to use
            ``gantry._skill_registry`` if present; otherwise it logs a
            warning and skips skill tools.
        always_include: Optional list of Gantry tool names to inject on
            every invocation (in addition to the dynamic top-k). Useful
            for pinning utility tools the LLM should always see. Missing
            names log a WARNING and are skipped.
        required: Optional list of Gantry tool names that **must** be
            available in the registry. The provider validates these at
            construction time and raises :class:`MissingRequiredToolError`
            if any are missing. They are also injected on every invocation
            (treated as ``always_include`` once present). Use this when a
            workflow is broken if the tool isn't surfaced (typo / dropped
            registration).
        as_function_tool: Forwarded to the underlying
            :class:`GantryToolBridge`. ``None`` (default) auto-detects;
            ``True`` forces ``agent_framework.FunctionTool`` wrapping;
            ``False`` returns bare callables (still accepted by AF).
        source_id: Provider identifier used by AF for tool/message
            attribution. Defaults to ``"agent_gantry"``. Override when
            running multiple Gantry instances side-by-side.
        bridge: Optional pre-built :class:`GantryToolBridge` to reuse its
            wrapper cache across multiple providers. When ``None`` a new
            bridge is built from the supplied ``gantry``.
        static_tools: AF-native tools (or any objects accepted by
            ``Agent(tools=[...])``) that should be appended to every
            round's surface. Unlike ``always_include`` — which pins
            *gantry-registered* tools by name — ``static_tools`` is for
            tools that live outside the gantry registry. They are never
            filtered by the per-call refresh.
        verbose: When ``True`` (default ``False``), the provider logs a
            one-line INFO summary of every retrieval round:
            ``gantry: query="…" → top5: [name:0.61, …]``. Sets the
            ``agent_gantry`` logger to INFO if it has no level set.
        **query_kwargs: Additional keyword arguments forwarded to
            :meth:`GantryToolBridge.get_tools` (e.g. ``namespaces``,
            ``required_capabilities``, ``enable_reranking``).

    Example:
        .. code-block:: python

            from agent_framework import Agent, SkillsProvider
            from agent_framework.openai import OpenAIChatClient
            from agent_gantry import AgentGantry, GantryContextProvider
            from agent_gantry.query import last_tool_result

            gantry = AgentGantry()
            # ... register tools, sync ...

            provider = GantryContextProvider(
                gantry,
                top_k=3,
                query_strategy="per_call",
                query_generator=last_tool_result,
                required=["validate_boundaries"],
            )

            agent = Agent(
                OpenAIChatClient(),
                "You are a helpful assistant.",
                context_providers=[provider, SkillsProvider(skill_paths="./skills")],
                middleware=[provider.as_chat_middleware()],
            )
    """

    def __new__(
        cls,
        gantry: AgentGantry,
        *,
        top_k: int = 5,
        score_threshold: float | str = 0.0,
        query_strategy: str = "per_run",
        query_generator: Callable[..., Any] | None = None,
        skills: bool = False,
        skill_registry: SkillRegistry | None = None,
        always_include: list[str] | None = None,
        required: list[str] | None = None,
        static_tools: list[Any] | None = None,
        as_function_tool: bool | None = None,
        source_id: str = "agent_gantry",
        bridge: GantryToolBridge | None = None,
        verbose: bool = False,
        **query_kwargs: Any,
    ) -> Any:
        if query_strategy not in ("per_run", "per_call"):
            raise ValueError(
                f"query_strategy must be 'per_run' or 'per_call', got {query_strategy!r}"
            )

        context_provider_cls = _import_context_provider()

        bridge = bridge or GantryToolBridge(
            gantry,
            score_threshold=score_threshold,
            as_function_tool=as_function_tool,
        )
        always_include = list(always_include or [])
        required = list(required or [])
        static_tools_list: list[Any] = list(static_tools or [])

        # Default query generator depends on strategy: per_call wants
        # round-to-round adaptation, so the historical last_user_text
        # default (which returns the same string every round) would
        # silently disable the very thing per_call enables.
        if query_generator is None:
            if query_strategy == "per_call":
                query_generator = _default_per_call_query()
            else:
                query_generator = _DEFAULT_PER_RUN_QUERY
        elif (
            query_strategy == "per_call"
            and query_generator is last_user_text
        ):
            logger.warning(
                "GantryContextProvider: query_strategy='per_call' was set "
                "but query_generator=last_user_text returns the same string "
                "every round, defeating per-call adaptation. Consider "
                "fallback_chain(last_tool_result, last_user_text) instead."
            )

        if required:
            known = gantry.list_tools_sync()
            available = {t.name for t in known} | {
                f"{t.namespace}.{t.name}" for t in known
            }
            missing = [name for name in required if name not in available]
            if missing:
                raise MissingRequiredToolError(
                    f"GantryContextProvider: required tool(s) not found in gantry: "
                    f"{missing}. Did you forget to register them, or is there a typo?"
                )

        # Combine for "always inject" set; required is a strict superset.
        always_include_effective: list[str] = []
        seen_ai: set[str] = set()
        for n in (*required, *always_include):
            if n not in seen_ai:
                seen_ai.add(n)
                always_include_effective.append(n)
        impl_cls = _build_impl_class(context_provider_cls)
        return impl_cls(
            gantry=gantry,
            bridge=bridge,
            top_k=top_k,
            score_threshold=score_threshold,
            query_strategy=query_strategy,
            query_generator=query_generator,
            skills=skills,
            skill_registry=skill_registry,
            always_include_effective=always_include_effective,
            declared_always_include=always_include,
            required=required,
            static_tools=static_tools_list,
            query_kwargs=query_kwargs,
            source_id=source_id,
            verbose=verbose,
        )


__all__ = ["GantryContextProvider", "MissingRequiredToolError"]
