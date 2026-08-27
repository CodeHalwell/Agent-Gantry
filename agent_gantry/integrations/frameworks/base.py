"""Shared base for native per-framework tool adapters.

Agent-Gantry's job is to *select* a small, relevant slice of tools from a large
registry. To hand those tools to a concrete agent framework (LangChain,
LlamaIndex, CrewAI, Pydantic AI, OpenAI Agents SDK, Smolagents, Haystack, Agno,
…) they must be wrapped as that framework's *native* tool object — a callable
that the framework can introspect (name, description, JSON-schema parameters)
and invoke.

Every adapter in this subpackage follows the same shape:

    toolset = GantryToolset(gantry)
    specs   = await toolset.select(query, limit=3)      # ranked ToolSpec list
    native  = [spec.to_<framework>() ...]               # framework-specific

``ToolSpec`` is the framework-neutral handle. It exposes the metadata a
framework needs plus two invocation entry points that both route through
``gantry.execute`` (so retries, timeouts, circuit breakers, and the security
policy all still apply):

- :meth:`ToolSpec.ainvoke` — async, ``**kwargs`` or a single dict.
- :meth:`ToolSpec.invoke`  — sync wrapper, safe to call even from inside a
  running event loop (it offloads to a shared worker thread and blocks).

The adapters are intentionally dependency-free at import time: the third-party
framework is imported lazily inside the ``to_*`` builder, so ``import
agent_gantry`` never requires LangChain et al. to be installed.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import os
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Literal

from agent_gantry.integrations.frameworks.errors import MissingRequiredToolError
from agent_gantry.schema.execution import ExecutionStatus, ToolCall
from agent_gantry.schema.query import ConversationContext, ToolQuery

if TYPE_CHECKING:
    from agent_gantry.core.gantry import AgentGantry
    from agent_gantry.schema.tool import ToolDefinition

logger = logging.getLogger(__name__)


#: Default number of tools surfaced per selection call, shared by every static
#: adapter (:class:`GantryToolset`, :class:`BaseFrameworkAdapter`) and every
#: live/deep per-turn provider (``live_wrappers.py``, ``frameworks/*_live.py``,
#: :class:`~agent_gantry.integrations.agent_framework_adapter.AgentFrameworkAdapter`).
#: A single named constant keeps the two families from drifting apart again —
#: they previously disagreed (static adapters defaulted to 3, live paths to 5).
DEFAULT_TOOL_LIMIT = 5

#: The deepest dynamic tool re-selection tier a framework adapter supports,
#: surfaced uniformly as ``adapter.live_tier``. See
#: :attr:`BaseFrameworkAdapter.live_tier` and
#: ``integrations/frameworks/README.md`` for the full per-framework table.
#:
#: - ``"per-turn"`` — the framework calls back into Gantry (directly, or via a
#:   Gantry-built hook/toolset/provider) on every model turn / reasoning step,
#:   so the tool surface can change *mid-run* (LangGraph, LlamaIndex,
#:   Pydantic AI, OpenAI Agents SDK, Semantic Kernel, Google ADK, AutoGen,
#:   Strands, Microsoft Agent Framework).
#: - ``"per-call"`` — the framework fixes its tool list at agent-construction
#:   time with no native mid-run hook, so the deepest Gantry can do is rebuild
#:   a fresh agent/tool list before each new top-level call (LangChain,
#:   CrewAI, Agno, Haystack, Smolagents — see ``live_wrappers.py``).
LiveTier = Literal["per-turn", "per-call"]


class ToolExecutionError(RuntimeError):
    """Raised when a Gantry-backed tool invocation does not succeed.

    Two subclasses distinguish the "never executed by design" outcomes so a
    framework caller can branch without string-matching the status:
    :class:`ToolConfirmationRequiredError` (the tool is confirmation-gated and was
    not run — approve by re-issuing with ``ToolCall(require_confirmation=
    False)``) and :class:`ToolPermissionDeniedError` (the security policy refused
    it). Catching :class:`ToolExecutionError` still catches every outcome.
    """

    def __init__(self, tool_name: str, status: str, error: str | None) -> None:
        self.tool_name = tool_name
        self.status = status
        self.error = error
        super().__init__(f"Tool {tool_name!r} failed (status={status}): {error or 'no detail'}")


class ToolConfirmationRequiredError(ToolExecutionError):
    """The tool requires human confirmation and was deliberately not executed."""


class ToolPermissionDeniedError(ToolExecutionError):
    """The security policy denied the tool call; it was not executed."""


@dataclass(frozen=True)
class ToolSpec:
    """Framework-neutral handle to one selected Gantry tool.

    Attributes:
        name: The tool's bare name (what the model calls).
        qualified_name: ``namespace.name`` — unique within a registry.
        description: Human/LLM-facing description.
        parameters: JSON-Schema object describing the arguments.
        requires_confirmation: Whether the underlying tool is high-risk.
        score: The semantic score from selection (0.0 if unknown).
    """

    name: str
    qualified_name: str
    description: str
    parameters: dict[str, Any]
    requires_confirmation: bool
    score: float
    _gantry: AgentGantry
    _namespace: str

    # -- invocation -------------------------------------------------------- #
    async def ainvoke(self, *args: Any, **kwargs: Any) -> Any:
        """Execute the tool through Gantry and return its raw result.

        Accepts either a single positional mapping (``ainvoke({"a": 1})``) or
        keyword arguments (``ainvoke(a=1)``). Raises :class:`ToolExecutionError`
        on any non-success status so framework error handling kicks in.

        ``None``-valued arguments for *optional* parameters are dropped: several
        frameworks (CrewAI, Pydantic-schema validators, …) materialize every
        declared optional field as ``None`` when the model didn't supply it,
        but the tool's JSON schema types those params (e.g. ``string``) and the
        executor rejects ``None``. Dropping them lets the tool's own default
        apply — ``None`` for a required param is kept so the error stays clear.
        """
        arguments = _coerce_arguments(args, kwargs)
        required = set(self.parameters.get("required") or [])
        arguments = {k: v for k, v in arguments.items() if v is not None or k in required}
        # Pass the namespace: selection resolved this spec to one specific
        # tool, and a bare-name execute would prefer ``default.<name>`` if
        # another namespace registers the same name.
        result = await self._gantry.execute(
            ToolCall(tool_name=self.name, namespace=self._namespace, arguments=arguments)
        )
        if result.status != ExecutionStatus.SUCCESS:
            exc_cls = ToolExecutionError
            if result.status == ExecutionStatus.PENDING_CONFIRMATION:
                exc_cls = ToolConfirmationRequiredError
            elif result.status == ExecutionStatus.PERMISSION_DENIED:
                exc_cls = ToolPermissionDeniedError
            raise exc_cls(
                self.name, getattr(result.status, "value", str(result.status)), result.error
            )
        return result.result

    def invoke(self, *args: Any, **kwargs: Any) -> Any:
        """Synchronous wrapper around :meth:`ainvoke`.

        Safe to call from synchronous framework code (CrewAI ``_run``,
        Smolagents ``forward``, Haystack/Agno function entrypoints, …) whether
        or not an event loop is already running on the current thread. When a
        loop is running, the coroutine is executed on a dedicated worker thread
        and this call blocks for the result — mirroring how those frameworks
        invoke a synchronous tool. Prefer :meth:`ainvoke` directly in async code.
        """
        return _run_coroutine_sync(self.ainvoke(*args, **kwargs))

    def callable_for_signature(
        self,
        *,
        union_optional: bool = False,
        type_matched_defaults: bool = False,
        annotated_descriptions: bool = False,
    ) -> Callable[..., Any]:
        """Return a plain async function that calls this tool by keyword.

        Frameworks that build their own tool object from a function (Smolagents,
        Agno, Pydantic AI, OpenAI Agents SDK, AutoGen) can wrap this. The
        returned function carries ``__name__`` / ``__doc__`` **and a real
        ``__signature__``** derived from :attr:`parameters`, so frameworks that
        introspect the signature to build the LLM tool schema see the actual
        parameters instead of a bare ``**kwargs`` (which would surface as a
        no-argument tool).

        ``union_optional`` (Semantic Kernel), ``type_matched_defaults``
        (Google ADK) and ``annotated_descriptions`` (frameworks that read
        parameter descriptions from ``Annotated`` metadata — Semantic Kernel,
        AG2) are opt-in signature tweaks — see :meth:`python_signature`.
        """

        async def _fn(**kwargs: Any) -> Any:
            return await self.ainvoke(**kwargs)

        _fn.__name__ = self.name
        _fn.__doc__ = self.description
        _fn.__signature__ = self.python_signature(  # type: ignore[attr-defined]
            union_optional=union_optional,
            type_matched_defaults=type_matched_defaults,
            annotated_descriptions=annotated_descriptions,
        )
        _fn.__annotations__ = {
            p.name: p.annotation
            for p in _fn.__signature__.parameters.values()
            if p.annotation is not inspect.Parameter.empty
        }
        return _fn

    def python_signature(
        self,
        *,
        union_optional: bool = False,
        type_matched_defaults: bool = False,
        annotated_descriptions: bool = False,
    ) -> inspect.Signature:
        """Build an :class:`inspect.Signature` from the JSON-Schema parameters.

        Each property becomes a keyword-only parameter; required properties have
        no default. Optional properties default to the schema's own ``default``
        when it declares one, else ``None``. Array properties with a typed
        ``items`` schema annotate as ``list[T]`` so signature-introspecting
        frameworks surface the item type. Three opt-in modes adapt the
        signature for stricter frameworks:

        - ``union_optional``: annotate optional params ``T | None`` (Semantic
          Kernel infers required-ness from the annotation, not the default).
        - ``type_matched_defaults``: for optional params with no schema
          ``default``, default to a type-matched empty value (``""`` / ``0`` /
          ``False`` / ``[]`` / ``{}``) instead of ``None`` (Google ADK's
          automatic function calling rejects both union types and a ``None``
          default whose type mismatches the annotation).
        - ``annotated_descriptions``: wrap each annotation in
          ``Annotated[T, "<description>"]`` when the property carries a
          ``description`` — the convention Semantic Kernel and AG2 read
          parameter descriptions from.
        """
        from typing import Annotated

        properties = self.parameters.get("properties") or {}
        required = set(self.parameters.get("required") or [])
        params: list[inspect.Parameter] = []
        for name, prop in properties.items():
            prop = prop if isinstance(prop, dict) else {}
            json_type = prop.get("type")
            annotation = _annotation_for_prop(prop)
            if name in required:
                default = inspect.Parameter.empty
            else:
                if union_optional:
                    # `T | None` — valid at runtime on the project's floor
                    # (3.10+, enforced by ruff UP) and the form SK uses to
                    # infer optionality.
                    annotation = annotation | None
                schema_default = prop.get("default")
                if schema_default is not None and _matches_json_type(
                    schema_default, json_type
                ):
                    # The schema's own default is the most faithful signal —
                    # surface it instead of a synthetic placeholder.
                    default = schema_default
                elif type_matched_defaults:
                    default = _typed_default(json_type)
                else:
                    default = None
            description = prop.get("description")
            if annotated_descriptions and isinstance(description, str) and description:
                annotation = Annotated[annotation, description]
            params.append(
                inspect.Parameter(
                    name,
                    inspect.Parameter.KEYWORD_ONLY,
                    default=default,
                    annotation=annotation,
                )
            )
        return inspect.Signature(params)


_JSON_TO_PYTHON: dict[str, Any] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "array": list,
    "object": dict,
}

# Scalar "empty" default per JSON-Schema type, used to mark an optional
# parameter without a None default (which strict validators like Google ADK
# reject as incompatible with a non-Optional annotation).
_SCALAR_DEFAULTS: dict[str, Any] = {
    "string": "",
    "integer": 0,
    "number": 0.0,
    "boolean": False,
}


def _typed_default(json_type: Any) -> Any:
    """Return a fresh, type-matched empty default for an optional parameter."""
    if isinstance(json_type, list):  # e.g. ["string", "null"]
        json_type = next((t for t in json_type if t != "null"), None)
    if json_type == "array":
        return []
    if json_type == "object":
        return {}
    return _SCALAR_DEFAULTS.get(json_type, "")


def _json_type_to_python(json_type: Any) -> Any:
    """Map a JSON-Schema ``type`` to a Python annotation (default ``str``)."""
    if isinstance(json_type, list):  # e.g. ["string", "null"]
        json_type = next((t for t in json_type if t != "null"), None)
    return _JSON_TO_PYTHON.get(json_type, str)


def _literal_members(prop: dict[str, Any]) -> tuple[Any, ...] | None:
    """The ``Literal`` members a schema's ``enum``/``const`` pins it to.

    ``None`` when the schema constrains nothing, or names a value no
    ``Literal`` can hold — there the caller keeps the plain type annotation
    rather than inventing one.
    """
    if "const" in prop:
        values: list[Any] = [prop["const"]]
    else:
        enum_values = prop.get("enum")
        if not isinstance(enum_values, list) or not enum_values:
            return None
        values = enum_values
    # Floats are admitted for the same reason the schema bridge admits them:
    # PEP 586 disallows a float member statically, but ``typing`` and every
    # framework validator downstream enforce it correctly at runtime, and the
    # alternative is advertising no constraint at all.
    if not all(isinstance(v, (str, int, bool, float)) or v is None for v in values):
        return None
    return tuple(values)


def _annotation_for_prop(prop: dict[str, Any]) -> Any:
    """Python annotation for one property schema, recursively.

    Frameworks that rebuild their LLM schema from the signature rather than
    from ``parameters_schema`` — Semantic Kernel, AG2, Google ADK's fallback
    path — see only what this returns, so anything it drops is invisible to
    the model. ``{"type": "array", "items": {"type": "string"}}`` annotates
    as ``list[str]`` rather than a bare ``list``; an ``enum``/``const``
    becomes a ``Literal`` rather than a bare ``str``, which previously let
    those frameworks advertise ``Literal["fast", "slow"]`` as unconstrained
    text; and a typed mapping keeps its value type.

    An object with declared ``properties`` still degrades to a bare ``dict``.
    Rebuilding it as a nested model (what CrewAI and LlamaIndex get from
    ``pydantic_model_from_schema``) would change what these frameworks
    introspect in a way this codebase can't exercise against their real
    schema derivation, so it stays a documented gap rather than an untested
    behaviour change.
    """
    literal = _literal_members(prop)
    if literal is not None:
        return Literal[literal]

    json_type = prop.get("type")
    annotation = _json_type_to_python(json_type)

    if annotation is list:
        items = prop.get("items")
        if isinstance(items, dict) and items:
            item_annotation = _annotation_for_prop(items)
            # A bare ``str`` here is the fallback for an untyped item schema,
            # not a real annotation — keep the bare container instead of
            # asserting an item type the schema never declared.
            if item_annotation is not str or items.get("type") == "string":
                return list[item_annotation]
        return annotation

    if annotation is dict:
        additional = prop.get("additionalProperties")
        properties = prop.get("properties")
        if isinstance(additional, dict) and additional and not properties:
            return dict[str, _annotation_for_prop(additional)]
        return annotation

    return annotation


def _matches_json_type(value: Any, json_type: Any) -> bool:
    """Whether a schema ``default`` is usable for its declared JSON type.

    Guards against surfacing a mistyped default (e.g. ``default: "5"`` on an
    ``integer`` property) as a Python signature default, where a strict
    framework validator would then reject the tool outright.
    """
    if isinstance(json_type, list):
        json_type = next((t for t in json_type if t != "null"), None)
    if json_type is None:
        return True
    if json_type == "boolean":
        return isinstance(value, bool)
    if json_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if json_type == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if json_type == "string":
        return isinstance(value, str)
    if json_type == "array":
        return isinstance(value, list)
    if json_type == "object":
        return isinstance(value, dict)
    return True


# Shared worker threads for running coroutines from sync framework callbacks
# while an event loop is active on the calling thread. Reused across
# invocations so we don't pay the spawn/teardown cost of a fresh pool each call.
_SYNC_BRIDGE_POOL: Any = None

#: Guards lazy pool construction: without it two threads racing through the
#: ``is None`` check each build a pool and one is silently discarded.
_SYNC_BRIDGE_POOL_LOCK = threading.Lock()

#: Set on threads owned by the bridge pool, so a nested sync invocation can
#: tell it is about to wait on the very pool it is occupying.
_BRIDGE_THREAD = threading.local()

#: The bridge fans out concurrent sync tool calls, so it must not be a single
#: worker: every sync tool call in the process shares this pool, and one slow
#: tool (default timeout: 30s) would otherwise block all the others — a
#: multi-agent CrewAI run serializes completely. Threads here are almost always
#: parked waiting on a coroutine, so they are cheap; the cap mirrors
#: ThreadPoolExecutor's own default.
_SYNC_BRIDGE_MAX_WORKERS = min(32, (os.cpu_count() or 1) + 4)


def _bridge_pool() -> Any:
    """Return the shared bridge pool, constructing it once."""
    global _SYNC_BRIDGE_POOL
    if _SYNC_BRIDGE_POOL is None:
        with _SYNC_BRIDGE_POOL_LOCK:
            if _SYNC_BRIDGE_POOL is None:
                import concurrent.futures

                _SYNC_BRIDGE_POOL = concurrent.futures.ThreadPoolExecutor(
                    max_workers=_SYNC_BRIDGE_MAX_WORKERS,
                    thread_name_prefix="gantry-sync-bridge",
                )
    return _SYNC_BRIDGE_POOL


def _run_on_bridge_thread(coro: Any) -> Any:
    """Run ``coro`` in a fresh event loop, marking the thread as the bridge's."""
    _BRIDGE_THREAD.active = True
    try:
        return asyncio.run(coro)
    finally:
        _BRIDGE_THREAD.active = False


def _run_coroutine_sync(coro: Any) -> Any:
    """Run an awaitable to completion from synchronous code, loop-or-not.

    If no event loop runs on the current thread, use :func:`asyncio.run`.
    Otherwise (we're inside a running loop — e.g. a framework invoked our sync
    tool from within its async agent loop), run the coroutine on a bridge
    worker thread with its own loop and block for the result. This avoids the
    "coroutine attached to a different loop" / "loop already running" errors
    that a naive ``asyncio.run`` would raise.

    A *nested* call — a tool handler that itself calls ``ToolSpec.invoke`` —
    gets its own throwaway thread rather than a pool slot. Submitting it to the
    pool would make the occupied worker wait on the pool it is occupying, which
    deadlocks outright at one worker and can still exhaust a larger pool at
    depth. Nesting is rare, so paying for a thread there is the right trade.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    if getattr(_BRIDGE_THREAD, "active", False):
        result: dict[str, Any] = {}

        def _runner() -> None:
            try:
                result["value"] = _run_on_bridge_thread(coro)
            except BaseException as exc:  # re-raised on the calling thread
                result["error"] = exc

        thread = threading.Thread(
            target=_runner, name="gantry-sync-bridge-nested", daemon=True
        )
        thread.start()
        thread.join()
        if "error" in result:
            raise result["error"]
        return result.get("value")

    return _bridge_pool().submit(_run_on_bridge_thread, coro).result()


def _coerce_arguments(args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
    """Normalize ``(*args, **kwargs)`` into a single argument dict."""
    if args:
        if len(args) == 1 and isinstance(args[0], dict) and not kwargs:
            return dict(args[0])
        raise TypeError(
            "Tool invocation accepts a single mapping positional argument or "
            "keyword arguments, not both / multiple positionals."
        )
    return dict(kwargs)


def spec_from_tool(gantry: AgentGantry, tool: ToolDefinition, score: float = 0.0) -> ToolSpec:
    """Build a :class:`ToolSpec` from a Gantry :class:`ToolDefinition`."""
    qualified = getattr(tool, "qualified_name", None) or f"{tool.namespace}.{tool.name}"
    params = tool.parameters_schema or {"type": "object", "properties": {}}
    return ToolSpec(
        name=tool.name,
        qualified_name=qualified,
        description=tool.description or tool.name,
        parameters=params,
        requires_confirmation=bool(getattr(tool, "requires_confirmation", False)),
        score=float(score),
        _gantry=gantry,
        _namespace=tool.namespace,
    )


def _bare_qualified_name(tool: ToolDefinition) -> str:
    """Return a tool's ``namespace.name`` string — no version suffix.

    This is the format a ``required=[...]``/``always_include=[...]`` caller
    uses to disambiguate same-named tools across namespaces, and the same
    format
    :class:`~agent_gantry.integrations.agent_framework_provider.GantryContextProvider`
    uses for its own ``required`` validation (``f"{t.namespace}.{t.name}"``).
    Deliberately distinct from :attr:`ToolDefinition.qualified_name` /
    :attr:`ToolSpec.qualified_name`, which both additionally suffix
    ``:<version>`` — a version pin isn't something a ``required=[...]``
    caller would reasonably type, so matching against it would make
    ``namespace.name`` lookups silently fail.
    """
    return f"{tool.namespace}.{tool.name}"


def _resolve_tool_names(
    gantry: AgentGantry, names: Sequence[str]
) -> tuple[list[ToolDefinition], list[str]]:
    """Resolve bare or qualified tool names against the gantry's registry.

    Reads :meth:`AgentGantry.list_tools_sync` — the in-memory registry, no
    vector-store round trip — the same source
    :class:`~agent_gantry.integrations.agent_framework_provider.GantryContextProvider`
    uses to validate its own ``required=[...]``. A name matches either the
    tool's bare ``name`` or its ``namespace.name`` qualified name (see
    :func:`_bare_qualified_name`); bare-name lookups take the first match
    across namespaces (mirroring
    :meth:`~agent_gantry.core.registry.ToolRegistry.get_tool_by_name`).

    Returns ``(found, missing)`` — ``(requested name, resolved ToolDefinition)``
    pairs in the order ``names`` was given, and the subset of ``names`` that
    matched nothing. The requested name is kept alongside the resolution so
    callers can tell a bare pin (``"foo"``) from a qualified one
    (``"other.foo"``) when deduplicating against the semantic slice.
    """
    known = gantry.list_tools_sync()
    by_qualified: dict[str, ToolDefinition] = {}
    by_bare: dict[str, ToolDefinition] = {}
    for tool in known:
        by_qualified.setdefault(_bare_qualified_name(tool), tool)
        by_bare.setdefault(tool.name, tool)

    found: list[tuple[str, ToolDefinition]] = []
    missing: list[str] = []
    for name in names:
        tool = by_qualified.get(name) or by_bare.get(name)
        if tool is None:
            missing.append(name)
        else:
            found.append((name, tool))
    return found, missing


def _pin_specs(
    gantry: AgentGantry,
    names: Sequence[str] | None,
    seen_qualified: set[str],
    seen_bare: set[str],
    *,
    kind: Literal["required", "always_include"],
) -> list[ToolSpec]:
    """Resolve ``names`` and wrap any not already present as pinned :class:`ToolSpec`\\ s.

    Shared by :meth:`GantryToolset.select` and :meth:`GantryToolset.select_or_empty`
    for both the ``required`` and ``always_include`` keyword arguments — the
    only difference between the two is what happens when a name doesn't
    resolve against the registry:

    - ``kind="required"``: raises :class:`MissingRequiredToolError` listing
      every unresolved name. Mirrors
      :class:`~agent_gantry.integrations.agent_framework_provider.GantryContextProvider`,
      which validates ``required=[...]`` and raises the same error type.
    - ``kind="always_include"``: logs a ``WARNING`` naming the unresolved
      tools and silently skips them — mirrors
      ``GantryContextProvider``'s ``always_include`` semantics (a missing
      always-include tool is a soft-fail, not a hard error).

    Deduplication respects the pin's own name form:

    - A **qualified** pin (``"other.foo"``) is skipped only when *that exact
      tool* (same ``namespace.name``) is already present — a same-named tool
      from a different namespace in the semantic slice does NOT satisfy it.
    - A **bare** pin (``"foo"``) is satisfied by *any* already-present tool
      with that bare name (semantic slice or an earlier pin), since the caller
      expressed no namespace preference.

    ``seen_qualified``/``seen_bare`` are mutated in place so an earlier call
    (``required`` is pinned before ``always_include``) deduplicates against
    later ones.
    """
    if not names:
        return []
    found, missing = _resolve_tool_names(gantry, names)
    if missing:
        if kind == "required":
            raise MissingRequiredToolError(
                f"GantryToolset.select: required tool(s) not found in gantry: "
                f"{missing}. Did you forget to register them, or is there a typo?"
            )
        logger.warning(
            "GantryToolset.select: always_include tool(s) not found in gantry "
            "and will be skipped: %s",
            missing,
        )
    pinned: list[ToolSpec] = []
    for requested, tool_def in found:
        bare_qualified = _bare_qualified_name(tool_def)
        if bare_qualified in seen_qualified:
            continue
        is_bare_pin = requested == tool_def.name
        if is_bare_pin and tool_def.name in seen_bare:
            continue
        spec = spec_from_tool(gantry, tool_def)
        seen_qualified.add(bare_qualified)
        seen_bare.add(tool_def.name)
        pinned.append(spec)
    return pinned


def _resolve_pins(
    gantry: AgentGantry,
    specs: list[ToolSpec],
    *,
    required: Sequence[str] | None,
    always_include: Sequence[str] | None,
) -> list[ToolSpec]:
    """Build the pinned-tool tail appended after ``specs`` (the semantic slice).

    Ordering matches
    :class:`~agent_gantry.integrations.agent_framework_provider.GantryContextProvider`:
    dynamic/semantic tools first (``specs``, already ranked), then
    ``required`` (in the order given), then ``always_include`` (in the order
    given, skipping anything ``required`` already pinned). Deduplicated
    against ``specs`` and against each other. Pinned tools are never counted
    against ``limit`` — the same choice
    ``GantryContextProvider.top_k`` makes for its own ``required`` /
    ``always_include`` — so a caller's tool budget for the *semantic* slice
    is never silently reduced by pins, and a required tool is never dropped
    because the semantic slice happened to fill ``limit`` first.
    """
    if not required and not always_include:
        return []
    seen_qualified: set[str] = set()
    seen_bare: set[str] = set()
    for s in specs:
        seen_qualified.add(f"{s._namespace}.{s.name}")
        seen_bare.add(s.name)
    pinned = _pin_specs(gantry, required, seen_qualified, seen_bare, kind="required")
    pinned += _pin_specs(
        gantry, always_include, seen_qualified, seen_bare, kind="always_include"
    )
    return pinned


class GantryToolset:
    """Framework-neutral entry point: select tools, then export to a framework.

    Example::

        toolset = GantryToolset(gantry)
        specs = await toolset.select("send an email", limit=3)
        # turn the specs into a framework's native objects via that framework's
        # adapter (each exposes a ``convert`` staticmethod):
        from agent_gantry.langchain import LangChainAdapter
        lc_tools = [LangChainAdapter.convert(s) for s in specs]
    """

    def __init__(self, gantry: AgentGantry, *, default_limit: int = DEFAULT_TOOL_LIMIT) -> None:
        self._gantry = gantry
        self._default_limit = default_limit

    async def select(
        self,
        query: str,
        *,
        limit: int | None = None,
        score_threshold: float = 0.0,
        namespaces: list[str] | None = None,
        tools_already_used: list[str] | None = None,
        required: list[str] | None = None,
        always_include: list[str] | None = None,
    ) -> list[ToolSpec]:
        """Run semantic selection and return ranked :class:`ToolSpec` handles.

        ``score_threshold`` defaults to ``0.0`` (no filtering) — matching the
        high-level convenience API and avoiding the silent-drop trap of the raw
        ``ToolQuery`` 0.5 default.

        ``required`` and ``always_include`` (both bare or ``namespace.name``
        qualified tool names) are pinned onto the semantic slice — ported
        from
        :class:`~agent_gantry.integrations.agent_framework_provider.GantryContextProvider`,
        the Microsoft Agent Framework provider that originated this feature,
        so every framework adapter gets the same guarantee:

        - ``required``: every listed tool **must** end up in the result. A
          tool already present in the semantic slice counts; anything
          missing is fetched from the registry and appended. If a name
          doesn't resolve against the registry at all, raises
          :class:`~agent_gantry.integrations.frameworks.errors.MissingRequiredToolError`
          rather than silently returning an incomplete selection.
        - ``always_include``: same resolution and append behaviour, but a
          name that isn't in the registry is logged as a ``WARNING`` and
          skipped rather than raising.

        Both are appended *after* the semantic slice, in the order given
        (``required`` before ``always_include``), deduplicated against it and
        against each other — see :func:`_resolve_pins` for the full ordering
        and dedup contract. Neither counts against ``limit``: ``limit`` bounds
        only the semantic retrieval, so a required tool is never dropped
        because the semantic slice already filled the budget, and pins never
        silently shrink the semantic slice a caller asked for.
        """
        context = ConversationContext(
            query=query,
            tools_already_used=list(tools_already_used or []),
        )
        result = await self._gantry.retrieve(
            ToolQuery(
                context=context,
                limit=limit or self._default_limit,
                score_threshold=score_threshold,
                namespaces=namespaces,
            )
        )
        specs = [spec_from_tool(self._gantry, st.tool, st.semantic_score) for st in result.tools]
        if not required and not always_include:
            return specs
        pinned = _resolve_pins(
            self._gantry, specs, required=required, always_include=always_include
        )
        return specs + pinned

    async def select_or_empty(
        self,
        query: str,
        *,
        limit: int | None = None,
        score_threshold: float = 0.0,
        namespaces: list[str] | None = None,
        tools_already_used: list[str] | None = None,
        required: list[str] | None = None,
        always_include: list[str] | None = None,
    ) -> list[ToolSpec]:
        """Like :meth:`select`, but skips the *semantic* leg for a blank query.

        Every per-turn live provider (``integrations/frameworks/*_live.py``,
        :class:`~agent_gantry.integrations.refresh.ToolRefresher`) needs this
        exact guard: selecting on an empty embedding yields an arbitrary
        top-k for some embedders, so a turn with no retrieval signal (no new
        user text, no tool result yet) should surface no *dynamic* tools
        rather than a nonsensical selection. This was previously
        re-implemented verbatim in nearly every ``*_live.py`` module (each
        with an identical "consistent with the other live providers" comment)
        — centralised here so each live provider keeps its own
        framework-specific query derivation but shares this one selection
        primitive.

        ``required`` / ``always_include`` are resolved and returned even on a
        blank query — they don't depend on the query's retrieval signal, and
        ``GantryContextProvider`` injects its own ``always_include``/``required``
        pins unconditionally for the same reason (a workflow that needs a
        pinned tool needs it whether or not this turn produced new query
        text). A blank query with a missing ``required`` tool still raises
        :class:`~agent_gantry.integrations.frameworks.errors.MissingRequiredToolError`.
        """
        if not (query or "").strip():
            return _resolve_pins(
                self._gantry, [], required=required, always_include=always_include
            )
        return await self.select(
            query,
            limit=limit,
            score_threshold=score_threshold,
            namespaces=namespaces,
            tools_already_used=tools_already_used,
            required=required,
            always_include=always_include,
        )


class BaseFrameworkAdapter:
    """Shared base for native per-framework tool adapters.

    Subclasses implement the :meth:`convert` staticmethod (one ``ToolSpec`` →
    one framework-native tool object). This base supplies the parts every
    adapter shares — construction and the ``select`` → ``convert`` pipeline —
    so each concrete adapter only declares ``convert`` plus any
    framework-specific helpers (agent builders, live retrievers, …).

    Uniform live entry point
    -------------------------
    Every adapter's *dynamic* re-selection surface is named differently
    per framework (``react_agent``, ``toolset``, ``tool_hook``,
    ``function_provider``, ``agent_builder``, …) because each framework
    exposes a different native hook. :attr:`live_tier` and :meth:`live`
    give callers a single, framework-agnostic way to ask "how deep does
    this adapter's dynamic tier go, and how do I get the live object for
    it?" without knowing which bespoke method to call. The bespoke methods
    themselves are never removed or renamed — they remain the documented,
    framework-idiomatic path; ``live()`` is a thin uniform layer that
    delegates to one of them. See ``integrations/frameworks/README.md``
    for the full per-framework table (tier, delegate, return type, where
    to plug it in).
    """

    def __init__(self, gantry: AgentGantry, *, default_limit: int = DEFAULT_TOOL_LIMIT) -> None:
        self._gantry = gantry
        self._default_limit = default_limit

    #: The deepest dynamic re-selection tier this adapter's framework
    #: supports — ``"per-turn"`` or ``"per-call"`` (see :data:`LiveTier`).
    #: Every concrete subclass MUST set this.
    live_tier: ClassVar[LiveTier]

    @staticmethod
    def convert(spec: ToolSpec) -> Any:
        """Wrap a single :class:`ToolSpec` as the framework's native tool object.

        Subclasses must re-declare this as a ``@staticmethod`` (it is part of the
        public API — callers use ``SomeAdapter.convert(spec)`` without an
        instance — and :meth:`select` dispatches through ``self.convert``).
        """
        raise NotImplementedError

    def live(
        self,
        *,
        limit: int | None = None,
        score_threshold: float = 0.0,
        namespaces: list[str] | None = None,
        required: list[str] | None = None,
        always_include: list[str] | None = None,
        **framework_kwargs: Any,
    ) -> Any:
        """Return this framework's live/dynamic tool object (uniform entry point).

        Every adapter's ``live()`` accepts the same five explicit keywords —
        ``limit``, ``score_threshold``, ``namespaces``, ``required``,
        ``always_include`` — plus ``**framework_kwargs`` forwarded verbatim to
        whichever framework-idiomatic bespoke method it delegates to
        (``react_agent``, ``toolset``, ``tool_hook``, ``agent_builder``, …).
        Some frameworks' native hooks are inherently tied to an external
        object the caller must supply (a chat model, an already-built agent,
        a kernel); those adapters require it as a named ``framework_kwargs``
        entry and raise a ``TypeError`` if it's missing — see the concrete
        override's docstring for exactly what is returned, which
        ``framework_kwargs`` are required, and where to plug the result in.
        :attr:`live_tier` tells you how deep the re-selection goes before you
        call this. ``required``/``always_include`` follow
        :meth:`GantryToolset.select`'s semantics (pinned tools, not counted
        against ``limit``; see that method for the full contract) and are
        re-applied on every dynamic re-selection round, not just the first.

        Subclasses MUST override this. The bespoke method(s) it wraps remain
        the documented, framework-native path and are never removed —
        ``live()`` is only a uniform layer on top of them.
        """
        raise NotImplementedError

    async def select(
        self,
        query: str,
        *,
        limit: int | None = None,
        score_threshold: float = 0.0,
        namespaces: list[str] | None = None,
        tools_already_used: list[str] | None = None,
        required: list[str] | None = None,
        always_include: list[str] | None = None,
    ) -> list[Any]:
        """Select tools for ``query`` as the framework's native tool objects.

        ``limit`` defaults to the adapter's ``default_limit``. ``score_threshold``,
        ``namespaces``, ``tools_already_used``, ``required``, and
        ``always_include`` are explicit, first-class keyword arguments — not
        buried in ``**kwargs`` — and are forwarded verbatim to
        :meth:`GantryToolset.select` (see its docstring for the
        ``required``/``always_include`` pinning contract). Each call still
        routes through ``gantry.execute`` so retries, timeouts, circuit
        breakers, and the security policy apply.
        """
        specs = await GantryToolset(self._gantry).select(
            query,
            limit=self._default_limit if limit is None else limit,
            score_threshold=score_threshold,
            namespaces=namespaces,
            tools_already_used=tools_already_used,
            required=required,
            always_include=always_include,
        )
        return [self.convert(s) for s in specs]
