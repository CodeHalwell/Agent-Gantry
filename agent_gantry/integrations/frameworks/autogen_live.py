"""Deep, per-turn dynamic-tool provider for AutoGen (autogen-core).

This is the *live* AutoGen integration — the deep counterpart to the
schema/registration helpers in :mod:`agent_gantry.integrations.frameworks.autogen`
(``for_autogen`` / ``register_with_autogen``). Where ``for_autogen`` selects a
tool slice **once** and hands AutoGen a static set of callables, this module
plugs Gantry directly into AutoGen's native dynamic-tool hook so the tool set is
re-selected from the registry on **every turn**.

The native hook is :class:`autogen_core.tools.Workbench` — a tool provider an
``AssistantAgent`` consumes. On each turn the agent calls
:meth:`Workbench.list_tools` to discover what is currently available, then
:meth:`Workbench.call_tool` to execute one. :class:`GantryWorkbench` overrides
those two methods: ``list_tools`` runs a fresh Gantry selection for the current
query and returns the matching :class:`~autogen_core.tools.ToolSchema` list;
``call_tool`` looks up the selected :class:`~agent_gantry.integrations.frameworks.base.ToolSpec`
and routes execution through ``gantry.execute`` (preserving retries, timeouts,
circuit breakers and the security policy), wrapping the output in a
:class:`~autogen_core.tools.ToolResult`.

Drive the per-turn behaviour by updating the query between turns::

    wb = gantry_workbench(gantry, limit=5)
    wb.set_query("send an email to the team")
    tools = await wb.list_tools()          # email tools surface
    result = await wb.call_tool("send_email", {"to": "team@x.com"})

The ``autogen_core`` import is lazy (only inside the class/factory), so
``import agent_gantry`` never requires AutoGen to be installed.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from agent_gantry.integrations.frameworks.base import GantryToolset, ToolSpec

if TYPE_CHECKING:
    from agent_gantry.core.gantry import AgentGantry


def _require_autogen() -> Any:
    """Import ``autogen_core.tools`` lazily, with a helpful error if missing."""
    try:
        import autogen_core.tools as _tools
    except ImportError as exc:  # pragma: no cover - exercised via importorskip
        raise ImportError(
            "AutoGen live (per-turn) support requires `autogen-core`. "
            "Install it with `pip install autogen-core`."
        ) from exc
    return _tools


def _build_workbench_class() -> type:
    """Build the ``GantryWorkbench`` subclass against the installed Workbench.

    The base class only exists once ``autogen_core`` is importable, so the
    concrete subclass is constructed lazily on first use and cached. This keeps
    ``import agent_gantry`` dependency-free while still yielding a real
    ``Workbench`` subclass that an ``AssistantAgent`` accepts.
    """
    tools = _require_autogen()
    Workbench = tools.Workbench  # noqa: N806
    ToolResult = tools.ToolResult  # noqa: N806
    TextResultContent = tools.TextResultContent  # noqa: N806

    class GantryWorkbench(Workbench):  # type: ignore[misc, valid-type]
        """A Gantry-backed AutoGen :class:`~autogen_core.tools.Workbench`.

        Provides a *dynamic* tool set: :meth:`list_tools` re-runs Gantry
        selection for the current query each call, so the tools an agent sees
        change turn-to-turn as the query (conversation focus) changes.
        """

        def __init__(
            self,
            gantry: AgentGantry,
            *,
            query: str = "",
            limit: int = 5,
            score_threshold: float = 0.0,
        ) -> None:
            self._toolset = GantryToolset(gantry)
            self._query = query
            self._limit = limit
            self._score_threshold = score_threshold
            # Specs from the most recent ``list_tools`` selection, keyed by the
            # tool name the model calls. ``call_tool`` resolves against this.
            self._selected: dict[str, ToolSpec] = {}

        # -- per-turn query -------------------------------------------------- #
        @property
        def query(self) -> str:
            """The query that drives the next :meth:`list_tools` selection."""
            return self._query

        @query.setter
        def query(self, value: str) -> None:
            self._query = value

        def set_query(self, query: str) -> GantryWorkbench:
            """Set the query that drives selection; returns ``self`` for chaining."""
            self._query = query
            return self

        # -- dynamic tool discovery ------------------------------------------ #
        async def list_tools(self) -> list[Any]:
            """Re-select tools for the current query and return their schemas.

            Called by the agent on every turn. The freshly selected specs are
            cached so :meth:`call_tool` can resolve and invoke them.
            """
            specs = await self._toolset.select(
                self._query,
                limit=self._limit,
                score_threshold=self._score_threshold,
            )
            self._selected = {spec.name: spec for spec in specs}
            return [self._spec_to_schema(spec) for spec in specs]

        @staticmethod
        def _spec_to_schema(spec: ToolSpec) -> Any:
            """Convert a :class:`ToolSpec` into an AutoGen ``ToolSchema``.

            ``ToolSchema``/``ParametersSchema`` are ``TypedDict``s, so a plain
            dict with the right keys is the native representation.
            """
            params = spec.parameters or {"type": "object", "properties": {}}
            parameters: dict[str, Any] = {
                "type": params.get("type", "object"),
                "properties": dict(params.get("properties") or {}),
            }
            required = params.get("required")
            if required:
                parameters["required"] = list(required)
            return {
                "name": spec.name,
                "description": spec.description,
                "parameters": parameters,
            }

        # -- execution ------------------------------------------------------- #
        async def call_tool(
            self,
            name: str,
            arguments: Mapping[str, Any] | None = None,
            cancellation_token: Any | None = None,
            call_id: str | None = None,
        ) -> Any:
            """Execute a selected tool through Gantry and wrap the result.

            Resolves ``name`` against the specs cached by the last
            :meth:`list_tools` call. If the tool was never selected (or
            execution fails) an error :class:`~autogen_core.tools.ToolResult` is
            returned rather than raising, matching the base contract.
            """
            # Snapshot the mapping by reference: a concurrent ``list_tools`` on
            # the same instance replaces ``self._selected`` wholesale, so binding
            # a local reference avoids a torn read between resolve and execute.
            selected = self._selected
            spec = selected.get(name)
            if spec is None:
                # Fall back to a fresh selection so a direct ``call_tool`` (no
                # prior ``list_tools``) still resolves the current query's tools.
                await self.list_tools()
                spec = self._selected.get(name)
            if spec is None:
                return ToolResult(
                    name=name,
                    result=[TextResultContent(content=f"Tool {name} not found.")],
                    is_error=True,
                )
            try:
                output = await spec.ainvoke(**(dict(arguments) if arguments else {}))
                return ToolResult(
                    name=name,
                    result=[TextResultContent(content=_as_text(output))],
                    is_error=False,
                )
            except Exception as exc:  # noqa: BLE001 - surfaced as tool error
                return ToolResult(
                    name=name,
                    result=[TextResultContent(content=f"{type(exc).__name__}: {exc}")],
                    is_error=True,
                )

        # -- lifecycle (no-ops: selection is stateless) ---------------------- #
        async def start(self) -> None:
            return None

        async def stop(self) -> None:
            return None

        async def reset(self) -> None:
            self._selected = {}

        async def save_state(self) -> Mapping[str, Any]:
            return {
                "query": self._query,
                "limit": self._limit,
                "score_threshold": self._score_threshold,
            }

        async def load_state(self, state: Mapping[str, Any]) -> None:
            self._query = state.get("query", self._query)
            self._limit = state.get("limit", self._limit)
            self._score_threshold = state.get("score_threshold", self._score_threshold)

    return GantryWorkbench


def _as_text(value: Any) -> str:
    """Render a tool's raw return value as text for a ``TextResultContent``."""
    if isinstance(value, str):
        return value
    return str(value)


_GANTRY_WORKBENCH_CLASS: type | None = None


def _get_class() -> type:
    global _GANTRY_WORKBENCH_CLASS
    if _GANTRY_WORKBENCH_CLASS is None:
        _GANTRY_WORKBENCH_CLASS = _build_workbench_class()
    return _GANTRY_WORKBENCH_CLASS


def __getattr__(name: str) -> Any:
    """Expose ``GantryWorkbench`` lazily so the base class is built on access.

    ``from ...autogen_live import GantryWorkbench`` triggers this and builds the
    real ``Workbench`` subclass on demand, keeping import-time dependency-free.
    """
    if name == "GantryWorkbench":
        return _get_class()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def gantry_workbench(
    gantry: AgentGantry,
    *,
    query: str = "",
    limit: int = 5,
    score_threshold: float = 0.0,
) -> Any:
    """Build a :class:`GantryWorkbench` for per-turn dynamic tool provision.

    The returned object is a real :class:`autogen_core.tools.Workbench` that an
    ``AssistantAgent`` can consume directly. Update its query between turns
    (via :meth:`GantryWorkbench.set_query` or the ``query`` property) to change
    which tools the next turn discovers.

    Raises:
        ImportError: If ``autogen-core`` is not installed.
    """
    cls = _get_class()
    return cls(
        gantry,
        query=query,
        limit=limit,
        score_threshold=score_threshold,
    )
