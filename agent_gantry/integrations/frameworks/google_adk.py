"""Google ADK native tool adapter for Agent-Gantry.

Selects a relevant slice of Gantry tools and wraps each as a Google Agent
Development Kit ``FunctionTool``, routed through ``gantry.execute`` and using
ADK's own mechanisms:

- **Errors** -- a failed call returns ``{"error": "<type>: <message>"}`` as the
  function response, ADK's convention for telling the model a tool failed (an
  exception would end the run). ``error_formatter`` redacts the text;
  ``error_mode="raise"`` lets
  :class:`~agent_gantry.integrations.frameworks.base.ToolExecutionError` propagate.
- **Approval** -- a confirmation-gated tool (``requires_confirmation=True`` or a
  ``SecurityPolicy(require_confirmation=[...])`` pattern) sets ADK's
  ``require_confirmation``, so the run asks for an ``adk_request_confirmation``
  response; a confirmed call runs exactly once with only Gantry's confirmation
  gate cleared, a rejected one never runs.
- **Run context** -- a Gantry tool reads the ADK ``ToolContext`` (session
  ``state``, ``user_id``, ``invocation_id``, artifacts, memory) with
  :func:`current_tool_context`, or through
  :func:`agent_gantry.current_tool_invocation`.
- **Tracing** -- ADK traces with OpenTelemetry; its active span becomes the
  Gantry ``ToolCall``'s trace and parent span.

Per-turn selection lives in :meth:`GoogleADKAdapter.before_model_callback`.
The ``google.adk`` import is lazy so ``import agent_gantry`` never requires ADK.

Public entry point: :class:`GoogleADKAdapter`.
"""

from __future__ import annotations

import contextvars
import copy
from typing import TYPE_CHECKING, Any

from agent_gantry.core.invocation import current_tool_invocation
from agent_gantry.integrations.frameworks.base import (
    DEFAULT_TOOL_LIMIT,
    BaseFrameworkAdapter,
    ToolConfirmationRequiredError,
    ToolExecutionError,
    ToolSpec,
)
from agent_gantry.integrations.frameworks.native import (
    ErrorFormatter,
    ErrorMode,
    check_error_mode,
    execute_spec,
    failure_text,
)

if TYPE_CHECKING:
    from agent_gantry.core.gantry import AgentGantry

FRAMEWORK_NAME = "google_adk"

_INSTALL_HINT = (
    "Google ADK support requires `google-adk`. Install it with "
    "`pip install 'agent-gantry[google-adk]'` (or `uv add 'agent-gantry[google-adk]'`)."
)

_PENDING_APPROVAL_TEXT = (
    "This tool call requires human approval and was not run. Ask the user to "
    "approve it, or continue without it."
)

#: The ADK ``ToolContext`` of the call being run (set by ``run_async``).
_TOOL_CONTEXT: contextvars.ContextVar[Any] = contextvars.ContextVar(
    "agent_gantry_adk_tool_context", default=None
)


def current_tool_context() -> Any:
    """The ADK ``ToolContext`` of the Gantry tool currently executing.

    Carries the session ``state``, ``user_id``, ``invocation_id``,
    ``function_call_id``, artifacts and memory. ``None`` outside a call driven
    by this adapter.
    """
    invocation = current_tool_invocation()
    if invocation is None or invocation.framework != FRAMEWORK_NAME:
        return None
    return invocation.native_context


def _confirmed(tool_context: Any) -> bool:
    confirmation = getattr(tool_context, "tool_confirmation", None)
    return getattr(confirmation, "confirmed", None) is True


def _metadata(tool_context: Any) -> dict[str, Any]:
    extras: dict[str, Any] = {}
    for key in ("agent_name", "user_id"):
        value = getattr(tool_context, key, None)
        if isinstance(value, str):
            extras[key] = value
    session = getattr(tool_context, "session", None)
    session_id = getattr(session, "id", None)
    if isinstance(session_id, str):
        extras["session_id"] = session_id
    return extras


def _execute_callable(spec: ToolSpec, signature_fn: Any) -> Any:
    """An async callable with ``signature_fn``'s signature that runs through Gantry.

    It reads the ADK ``ToolContext`` that ``run_async`` put in :data:`_TOOL_CONTEXT`.
    """
    aliases = spec.parameter_aliases()

    async def _call(**kwargs: Any) -> Any:
        if aliases:
            kwargs = {aliases.get(key, key): value for key, value in kwargs.items()}
        tool_context = _TOOL_CONTEXT.get()
        if tool_context is None:
            return await spec.ainvoke(**kwargs)
        call_id = getattr(tool_context, "function_call_id", None)
        run_id = getattr(tool_context, "invocation_id", None)
        return await execute_spec(
            spec,
            kwargs,
            framework=FRAMEWORK_NAME,
            native_context=tool_context,
            deps=getattr(tool_context, "state", None),
            run_id=run_id if isinstance(run_id, str) else None,
            tool_call_id=call_id if isinstance(call_id, str) else None,
            metadata=_metadata(tool_context),
            approved=_confirmed(tool_context),
        )

    _call.__name__ = signature_fn.__name__
    _call.__doc__ = signature_fn.__doc__
    _call.__signature__ = signature_fn.__signature__  # type: ignore[attr-defined]
    _call.__annotations__ = dict(signature_fn.__annotations__)
    return _call


def _accepts_require_confirmation(function_tool_cls: Any) -> bool:
    import inspect

    try:
        return "require_confirmation" in inspect.signature(function_tool_cls).parameters
    except (TypeError, ValueError):
        return False


def _spec_to_google_adk(
    spec: ToolSpec,
    *,
    error_mode: ErrorMode = "model",
    error_formatter: ErrorFormatter | None = None,
) -> Any:
    """Wrap a :class:`ToolSpec` as a Google ADK ``FunctionTool``.

    The declaration the model sees comes from the Gantry tool's own JSON
    schema whenever the installed ``google-genai`` supports
    ``FunctionDeclaration.parameters_json_schema`` (a ``FunctionTool``
    subclass overrides ``_get_declaration``), so per-parameter descriptions,
    enums, nested objects and the required/optional split survive; otherwise
    ADK introspects the callable's schema-derived signature. ``run_async`` is
    overridden to expose the ``ToolContext``, map failures to an ``{"error": ...}``
    response (``error_mode="model"``) and pass ADK's confirmation through.

    Raises:
        ImportError: If ``google-adk`` is not installed.
    """
    try:
        from google.adk.tools import FunctionTool
    except ImportError as exc:  # pragma: no cover - exercised via stub
        raise ImportError(_INSTALL_HINT) from exc

    error_mode = check_error_mode(error_mode)
    model_mode = error_mode == "model"
    # ADK's automatic function calling rejects `T | None` and `None`-typed
    # defaults, so opt into type-matched empty defaults for optional params.
    fn = _execute_callable(spec, spec.callable_for_signature(type_matched_defaults=True))

    kwargs: dict[str, Any] = {}
    gantry = spec._gantry
    if model_mode and gantry is not None and _accepts_require_confirmation(FunctionTool):
        tool_ref = spec.tool or spec.name
        namespace = spec._namespace

        def _require_confirmation(*_args: Any, **_kwargs: Any) -> bool:
            return gantry.requires_confirmation(tool_ref, namespace)

        kwargs["require_confirmation"] = _require_confirmation

    try:

        class _GantryADKFunctionTool(FunctionTool):  # type: ignore[misc, valid-type]
            def _get_declaration(self) -> Any:
                declaration = _declaration_from_schema(spec)
                if declaration is not None:
                    return declaration
                return super()._get_declaration()

            async def run_async(self, *, args: Any, tool_context: Any) -> Any:
                token = _TOOL_CONTEXT.set(tool_context)
                try:
                    return await super().run_async(args=args, tool_context=tool_context)
                except ToolExecutionError as exc:
                    if not model_mode:
                        raise
                    if isinstance(exc, ToolConfirmationRequiredError):
                        return {"error": _PENDING_APPROVAL_TEXT}
                    return {"error": failure_text(exc, error_formatter)}
                finally:
                    _TOOL_CONTEXT.reset(token)

    except TypeError:
        # ``FunctionTool`` isn't subclassable (test doubles, exotic shims) —
        # fall back to the plain wrapper; ADK then derives the declaration
        # from the callable's signature as before.
        return FunctionTool(func=fn)
    return _GantryADKFunctionTool(func=fn, **kwargs)


def _declaration_from_schema(spec: ToolSpec) -> Any:
    """Build a genai ``FunctionDeclaration`` carrying the schema verbatim.

    Returns ``None`` (caller falls back to ADK's signature introspection)
    when the installed ``google-genai`` predates ``parameters_json_schema``
    or the declaration can't be constructed.
    """
    try:
        from google.genai import types as genai_types
    except ImportError:  # pragma: no cover - genai always ships with adk
        return None
    declaration_cls = getattr(genai_types, "FunctionDeclaration", None)
    if declaration_cls is None or "parameters_json_schema" not in getattr(
        declaration_cls, "model_fields", {}
    ):
        return None
    # Deep, not shallow: a ``dict(...)`` leaves every nested ``properties`` /
    # ``items`` subschema aliased to the registry's canonical
    # ``ToolDefinition.parameters_schema``, so anything that adjusts the
    # declaration corrupts the tool for every other consumer. Property names
    # are the callable's aliases: ADK filters the model's arguments to the
    # signature's parameter names, so the declaration must use the same ones.
    schema = copy.deepcopy(spec.aliased_parameters()) if spec.parameters else {}
    schema.setdefault("type", "object")
    schema.setdefault("properties", {})
    try:
        return declaration_cls(
            name=spec.native_name,
            description=spec.description,
            parameters_json_schema=schema,
        )
    except Exception:  # noqa: BLE001 - fall back to signature introspection
        return None


class GoogleADKAdapter(BaseFrameworkAdapter):
    """Route Gantry tools into Google ADK through its native mechanisms.

    Static slice (``google.adk.tools.FunctionTool`` objects from :meth:`select`)
    plus per-turn selection (:meth:`before_model_callback` re-selects tools
    before every model request; :meth:`agent` builds an ``Agent`` wired with
    it). Every call routes through ``gantry.execute``; see the module docs for
    errors, approval, run context and tracing.

    Args:
        gantry: The registry to select from and execute against.
        default_limit: Default selection budget.
        error_mode: Default ``ErrorMode`` for the tools this adapter builds.
        error_formatter: Rewrites failure text before the model sees it.
    """

    live_tier = "per-turn"
    _live_delegate = "before_model_callback"

    def __init__(
        self,
        gantry: AgentGantry,
        *,
        default_limit: int = DEFAULT_TOOL_LIMIT,
        error_mode: ErrorMode = "model",
        error_formatter: ErrorFormatter | None = None,
    ) -> None:
        super().__init__(gantry, default_limit=default_limit)
        self._error_mode = check_error_mode(error_mode)
        self._error_formatter = error_formatter

    def _mode(self, error_mode: ErrorMode | None) -> ErrorMode:
        return self._error_mode if error_mode is None else check_error_mode(error_mode)

    @staticmethod
    def convert(
        spec: ToolSpec,
        *,
        error_mode: ErrorMode = "model",
        error_formatter: ErrorFormatter | None = None,
    ) -> Any:
        """Wrap a single :class:`ToolSpec` as a Google ADK ``FunctionTool``."""
        return _spec_to_google_adk(spec, error_mode=error_mode, error_formatter=error_formatter)

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
        error_mode: ErrorMode | None = None,
    ) -> list[Any]:
        """Select tools for ``query`` once, as ADK ``FunctionTool`` objects."""
        from agent_gantry.integrations.frameworks.base import GantryToolset

        mode = self._mode(error_mode)
        specs = await GantryToolset(self._gantry).select(
            query,
            tools_already_used=tools_already_used,
            **self._selection_kwargs(limit, score_threshold, namespaces, required, always_include),
        )
        return [
            _spec_to_google_adk(s, error_mode=mode, error_formatter=self._error_formatter)
            for s in specs
        ]

    def _live_kwargs(self, error_mode: ErrorMode | None, **extra: Any) -> dict[str, Any]:
        return {
            "error_mode": self._mode(error_mode),
            "error_formatter": self._error_formatter,
            **extra,
        }

    def before_model_callback(
        self,
        *,
        limit: int | None = None,
        score_threshold: float = 0.0,
        namespaces: list[str] | None = None,
        required: list[str] | None = None,
        always_include: list[str] | None = None,
        error_mode: ErrorMode | None = None,
        sticky: bool = True,
        max_offered: int | None = None,
        query_from_results: bool = True,
    ) -> Any:
        """Build an ADK ``before_model_callback`` that injects Gantry tools per turn.

        See :mod:`~agent_gantry.integrations.frameworks.google_adk_live` for the
        query, per-run state, stickiness and failure behaviour.
        """
        from agent_gantry.integrations.frameworks.google_adk_live import (
            _gantry_before_model_callback,
        )

        return _gantry_before_model_callback(
            self._gantry,
            **self._selection_kwargs(limit, score_threshold, namespaces, required, always_include),
            **self._live_kwargs(
                error_mode,
                sticky=sticky,
                max_offered=max_offered,
                query_from_results=query_from_results,
            ),
        )

    def agent(
        self,
        *,
        model: Any,
        name: str,
        instruction: str = "",
        limit: int | None = None,
        score_threshold: float = 0.0,
        namespaces: list[str] | None = None,
        required: list[str] | None = None,
        always_include: list[str] | None = None,
        error_mode: ErrorMode | None = None,
        sticky: bool = True,
        max_offered: int | None = None,
        query_from_results: bool = True,
        **agent_kwargs: Any,
    ) -> Any:
        """Build an ADK ``Agent`` wired for per-turn dynamic tool selection (tools=[] + callback)."""
        from agent_gantry.integrations.frameworks.google_adk_live import (
            _gantry_adk_agent,
        )

        return _gantry_adk_agent(
            self._gantry,
            model=model,
            name=name,
            instruction=instruction,
            **self._selection_kwargs(limit, score_threshold, namespaces, required, always_include),
            **self._live_kwargs(
                error_mode,
                sticky=sticky,
                max_offered=max_offered,
                query_from_results=query_from_results,
            ),
            **agent_kwargs,
        )


__all__ = ["GoogleADKAdapter", "current_tool_context"]
