"""Deep, per-turn dynamic-tool provider for Pydantic AI (pydantic-ai).

This is the *live* Pydantic AI integration — the deep counterpart to the
schema/wrapping helpers in
:mod:`agent_gantry.integrations.frameworks.pydantic_ai`
(``_for_pydantic_ai`` / ``_spec_to_pydantic_ai``). Where ``_for_pydantic_ai``
selects a tool slice **once** and hands Pydantic AI a static list of ``Tool``
objects, this module plugs Gantry directly into Pydantic AI's native
dynamic-tool hook so the tool set is re-selected from the registry on **every
run/step**.

The native hook is :class:`pydantic_ai.toolsets.AbstractToolset` — a toolset an
``Agent`` consumes. On each run/step the agent calls
:meth:`AbstractToolset.get_tools` to discover what is currently available, then
:meth:`AbstractToolset.call_tool` to execute one. :class:`GantryToolset`
overrides those: ``get_tools`` derives the current query from the
:class:`~pydantic_ai.tools.RunContext` (the latest user prompt / messages), runs
a fresh Gantry selection, and returns a dict of
:class:`~pydantic_ai.toolsets.abstract.ToolsetTool` keyed by tool name — each
tool's ``ToolDefinition.parameters_json_schema`` set from the Gantry tool's JSON
schema; ``call_tool`` resolves the matching
:class:`~agent_gantry.integrations.frameworks.base.ToolSpec` and routes
execution through ``gantry.execute`` (preserving retries, timeouts, circuit
breakers and the security policy).

The per-turn query is derived from the run context, so the tools an agent sees
change run-to-run as the conversation focus shifts — no manual ``set_query``
needed (though one is provided for driving selection without a full context)::

    toolset = _gantry_toolset(gantry, limit=5)
    agent = Agent(model, toolsets=[toolset])
    await agent.run("send an email to the team")   # email tools surface

The ``pydantic_ai`` import is lazy (only inside the class/factory), so
``import agent_gantry`` never requires Pydantic AI to be installed.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from agent_gantry.integrations.frameworks.base import DEFAULT_TOOL_LIMIT, ToolSpec
from agent_gantry.integrations.frameworks.base import GantryToolset as _BaseToolset
from agent_gantry.query import latest_activity

if TYPE_CHECKING:
    from agent_gantry.core.gantry import AgentGantry

logger = logging.getLogger(__name__)


def _require_pydantic_ai() -> tuple[Any, Any, Any, Any]:
    """Import the Pydantic AI toolset machinery, with a helpful error if missing.

    Returns ``(AbstractToolset, ToolsetTool, ToolDefinition, SchemaValidator)``
    — everything needed to build a concrete toolset against the installed
    version.
    """
    try:
        from pydantic_ai.tools import ToolDefinition
        from pydantic_ai.toolsets import AbstractToolset
        from pydantic_ai.toolsets.abstract import ToolsetTool
        from pydantic_core import SchemaValidator, core_schema
    except ImportError as exc:  # pragma: no cover - exercised via importorskip
        raise ImportError(
            "Pydantic AI live (per-turn) support requires `pydantic-ai`. "
            "Install it with `pip install pydantic-ai`."
        ) from exc
    return AbstractToolset, ToolsetTool, ToolDefinition, (SchemaValidator, core_schema)


def _query_from_ctx(ctx: Any) -> str:
    """Derive the per-turn selection query from a Pydantic AI ``RunContext``.

    Pydantic AI's ``ModelMessage`` objects are *not* role-shaped (they hold a
    ``parts`` list, not a ``.role``/``.content``), so the generic
    :func:`~agent_gantry.query.latest_activity` walker can't read them directly.
    We derive the query in order of decreasing directness:

    1. ``ctx.prompt`` — the current run's user prompt (string or content
       sequence). This is the freshest signal each run.
    2. The latest text-bearing part in ``ctx.messages`` — newest user-prompt or
       tool-return content, so a chaining/autonomous agent re-selects from what
       just happened.
    3. :func:`latest_activity` as a final fallback (covers role-shaped or
       dict-shaped histories some setups pass through).
    """
    prompt = getattr(ctx, "prompt", None)
    text = _content_text(prompt)
    if text:
        return text

    messages = getattr(ctx, "messages", None) or []
    text = _latest_part_text(messages)
    if text:
        return text

    return latest_activity(messages)


def _latest_part_text(messages: Any) -> str:
    """Pull text from the newest user-prompt / tool-return part in ``messages``.

    Walks Pydantic AI ``ModelMessage`` objects from the end, returning the text
    of the first ``user-prompt`` or ``tool-return`` part it finds — the content
    that should drive the next selection. Empty string if none surface.
    """
    try:
        ordered = list(messages)
    except TypeError:
        return ""
    for message in reversed(ordered):
        parts = getattr(message, "parts", None)
        if not parts:
            continue
        for part in reversed(parts):
            kind = getattr(part, "part_kind", None)
            if kind in ("user-prompt", "tool-return"):
                text = _content_text(getattr(part, "content", None))
                if text:
                    return text
    return ""


def _content_text(content: Any) -> str:
    """Render a prompt/part ``content`` (str or content sequence) as plain text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, (list, tuple)):
        parts = [c for c in (_content_text(item) for item in content) if c]
        return " ".join(parts).strip()
    return str(content).strip()


def _build_toolset_class() -> type:
    """Build the ``GantryToolset`` subclass against the installed toolset base.

    The base class only exists once ``pydantic_ai`` is importable, so the
    concrete subclass is constructed lazily on first use and cached. This keeps
    ``import agent_gantry`` dependency-free while still yielding a real
    ``AbstractToolset`` subclass that an ``Agent`` accepts.
    """
    AbstractToolset, ToolsetTool, ToolDefinition, (SchemaValidator, core_schema) = (  # noqa: N806
        _require_pydantic_ai()
    )

    # Like ``Tool.from_schema``, schema validation of arguments is delegated to
    # the JSON schema we expose on the ``ToolDefinition``; the args validator is
    # permissive so the model's arguments pass through to ``gantry.execute``
    # verbatim (Gantry performs its own validation/coercion on execute).
    _passthrough_validator = SchemaValidator(schema=core_schema.any_schema())

    class GantryToolset(AbstractToolset):  # type: ignore[misc, valid-type]
        """A Gantry-backed Pydantic AI :class:`AbstractToolset`.

        Provides a *dynamic* tool set: :meth:`get_tools` re-runs Gantry
        selection for the query derived from the current run context each call,
        so the tools an agent sees change run-to-run as the query
        (conversation focus) changes.
        """

        def __init__(
            self,
            gantry: AgentGantry,
            *,
            limit: int = DEFAULT_TOOL_LIMIT,
            score_threshold: float = 0.0,
            namespaces: list[str] | None = None,
        ) -> None:
            self._toolset = _BaseToolset(gantry)
            self._limit = limit
            self._score_threshold = score_threshold
            self._namespaces = namespaces
            # An explicit query override, used when driving selection without a
            # full RunContext (tests, manual selection). When set it takes
            # precedence over the context-derived query.
            self._query: str | None = None
            # Specs from the most recent ``get_tools`` selection, keyed by the
            # tool name the model calls. ``call_tool`` resolves against this.
            self._selected: dict[str, ToolSpec] = {}

        # -- identity (required abstract member) ----------------------------- #
        @property
        def id(self) -> str | None:
            """Stable identifier for this toolset within an agent."""
            return "agent-gantry"

        @property
        def label(self) -> str:
            return "Agent-Gantry (per-turn dynamic selection)"

        # -- per-turn query override ----------------------------------------- #
        @property
        def query(self) -> str | None:
            """An explicit query override for the next :meth:`get_tools`."""
            return self._query

        @query.setter
        def query(self, value: str | None) -> None:
            self._query = value

        def set_query(self, query: str | None) -> GantryToolset:
            """Set an explicit selection query; returns ``self`` for chaining.

            Useful for driving ``get_tools`` without constructing a full
            ``RunContext``. When ``None`` (the default), the query is derived
            from the run context each call.
            """
            self._query = query
            return self

        # -- dynamic tool discovery ------------------------------------------ #
        async def get_tools(self, ctx: Any) -> dict[str, Any]:
            """Re-select tools for the current run and return their definitions.

            Called by the agent on every run/step. The query is derived from
            ``ctx`` (latest user prompt / messages) unless an explicit override
            was set via :meth:`set_query`. The freshly selected specs are cached
            so :meth:`call_tool` can resolve and invoke them.

            Never raises on selection failure — a broken retrieval must not
            break the agent's run. ``self._selected`` persists across runs
            (``call_tool`` resolves against it), so on failure this logs a
            WARNING and leaves the previous run's selection in place rather
            than wiping it — see "Per-turn selection-failure policy" in
            ``integrations/frameworks/README.md``.
            """
            query = self._query if self._query is not None else _query_from_ctx(ctx)
            try:
                specs = await self._toolset.select_or_empty(
                    query,
                    limit=self._limit,
                    score_threshold=self._score_threshold,
                    namespaces=self._namespaces,
                )
            except Exception:
                logger.warning(
                    "GantryToolset.get_tools: semantic retrieval failed; "
                    "continuing with the previous run's tools.",
                    exc_info=True,
                )
                return {
                    name: self._spec_to_tool(ctx, spec) for name, spec in self._selected.items()
                }
            self._selected = {spec.name: spec for spec in specs}
            return {spec.name: self._spec_to_tool(ctx, spec) for spec in specs}

        def _spec_to_tool(self, ctx: Any, spec: ToolSpec) -> Any:
            """Wrap a :class:`ToolSpec` as a native ``ToolsetTool``.

            The Gantry JSON schema is used verbatim as the tool definition's
            ``parameters_json_schema`` so the model sees the real parameters.
            """
            tool_def = ToolDefinition(
                name=spec.name,
                description=spec.description,
                parameters_json_schema=spec.parameters or {"type": "object", "properties": {}},
            )
            return ToolsetTool(
                toolset=self,
                tool_def=tool_def,
                max_retries=getattr(ctx, "max_retries", 1) or 1,
                args_validator=_passthrough_validator,
            )

        # -- execution ------------------------------------------------------- #
        async def call_tool(
            self,
            name: str,
            tool_args: dict[str, Any],
            ctx: Any,
            tool: Any,
        ) -> Any:
            """Execute a selected tool through Gantry and return its result.

            Resolves ``name`` against the specs cached by the last
            :meth:`get_tools` call. If the tool was never selected (e.g. a
            direct ``call_tool`` with no prior ``get_tools``), a fresh selection
            for the current context is run first.
            """
            spec = self._selected.get(name)
            if spec is None:
                await self.get_tools(ctx)
                spec = self._selected.get(name)
            if spec is None:
                raise KeyError(f"Tool {name!r} is not available in this toolset.")
            return await spec.ainvoke(**(dict(tool_args) if tool_args else {}))

    return GantryToolset


_GANTRY_TOOLSET_CLASS: type | None = None


def _get_class() -> type:
    global _GANTRY_TOOLSET_CLASS
    if _GANTRY_TOOLSET_CLASS is None:
        _GANTRY_TOOLSET_CLASS = _build_toolset_class()
    return _GANTRY_TOOLSET_CLASS


def _gantry_toolset(
    gantry: AgentGantry,
    *,
    limit: int = DEFAULT_TOOL_LIMIT,
    score_threshold: float = 0.0,
    namespaces: list[str] | None = None,
) -> Any:
    """Build a :class:`GantryToolset` for per-turn dynamic tool provision.

    The returned object is a real :class:`pydantic_ai.toolsets.AbstractToolset`
    that an ``Agent`` can consume directly (``Agent(model, toolsets=[ts])``). On
    every run/step the agent calls ``get_tools`` and the tool set is re-selected
    from Gantry for the run's query (derived from the run context).

    Raises:
        ImportError: If ``pydantic-ai`` is not installed.
    """
    cls = _get_class()
    return cls(gantry, limit=limit, score_threshold=score_threshold, namespaces=namespaces)


def __getattr__(name: str) -> Any:
    """Expose the dynamically-built ``GantryToolset`` subclass lazily.

    It subclasses ``pydantic_ai.toolsets.AbstractToolset``, so it's built on first
    access to keep ``import`` dependency-free. Internal/advanced use (e.g.
    ``isinstance`` checks); the public entry point is ``PydanticAIAdapter.toolset(...)``.
    """
    if name == "GantryToolset":
        return _get_class()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
