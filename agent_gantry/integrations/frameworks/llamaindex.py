"""LlamaIndex adapter: export selected Gantry tools as ``FunctionTool`` objects.

LlamaIndex agents consume :class:`llama_index.core.tools.FunctionTool` objects.
This module wraps Gantry-selected :class:`ToolSpec` handles into that native
type, routing every call through ``gantry.execute`` and using LlamaIndex's own
mechanisms:

- **Errors** -- a failed call raises :class:`LlamaIndexToolError`, whose text is
  ``<type>: <message>`` (capped, optionally redacted by ``error_formatter``);
  LlamaIndex agents turn a raised tool error into ``ToolOutput(is_error=True)``
  for the model, so the run carries on. ``error_mode="raise"`` raises the plain
  :class:`~agent_gantry.integrations.frameworks.base.ToolExecutionError` instead.
- **Approval** -- a confirmation-gated call (``requires_confirmation=True`` or a
  ``SecurityPolicy(require_confirmation=[...])`` pattern) running in a
  workflow agent waits with ``ctx.wait_for_event(HumanResponseEvent,
  waiter_event=InputRequiredEvent(...))``. Answer ``HumanResponseEvent(response="yes")``
  to run it exactly once; anything else rejects it. Outside a workflow the
  model is told the call needs approval.
- **Run context** -- the workflow ``Context`` is injected (an extra ``ctx``
  parameter, hidden from the schema) and readable with
  :func:`current_workflow_context` or :func:`agent_gantry.current_tool_invocation`.
- **Tracing** -- an active OpenTelemetry span becomes the Gantry ``ToolCall``'s
  trace and parent span.

The ``llama_index`` import is lazy (performed inside the builder) so that
``import agent_gantry`` never requires LlamaIndex to be installed.

Public entry point: :class:`LlamaIndexAdapter`.
"""

from __future__ import annotations

import inspect
import json
from typing import TYPE_CHECKING, Any

from agent_gantry.core.invocation import current_tool_invocation
from agent_gantry.integrations.frameworks.base import (
    DEFAULT_TOOL_LIMIT,
    BaseFrameworkAdapter,
    ToolConfirmationRequiredError,
    ToolExecutionError,
    _run_coroutine_sync,
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
    from agent_gantry.integrations.frameworks.base import ToolSpec

FRAMEWORK_NAME = "llamaindex"

_INSTALL_HINT = (
    "LlamaIndex is required for LlamaIndexAdapter; install it with "
    "`pip install 'agent-gantry[llamaindex]'` (or `uv add 'agent-gantry[llamaindex]'`)."
)

_PENDING_APPROVAL_TEXT = (
    "This tool call requires human approval and was not run. Ask the user to "
    "approve it, or continue without it."
)
_REJECTED_TEXT = "The user rejected this tool call, so it was not run."

#: ``HumanResponseEvent.response`` values that approve a gated call.
_APPROVALS = frozenset({"y", "yes", "approve", "approved"})


class LlamaIndexToolError(ToolExecutionError):
    """A failed Gantry call as LlamaIndex shows it to the model.

    ``str()`` is the model-facing failure text, which LlamaIndex agents put in
    ``ToolOutput(is_error=True)``; the attributes are those of the original
    :class:`~agent_gantry.integrations.frameworks.base.ToolExecutionError`.
    """

    def __init__(self, source: ToolExecutionError | None, text: str, *, tool_name: str) -> None:
        super().__init__(
            source.tool_name if source is not None else tool_name,
            source.status if source is not None else "rejected",
            source.error if source is not None else text,
            source.error_type if source is not None else None,
        )
        self.text = text

    def __str__(self) -> str:
        return self.text


def current_workflow_context() -> Any:
    """The LlamaIndex workflow ``Context`` of the Gantry tool currently executing.

    ``None`` outside a workflow agent run, or for a tool called directly.
    """
    invocation = current_tool_invocation()
    if invocation is None or invocation.framework != FRAMEWORK_NAME:
        return None
    return invocation.native_context


def _is_approval(event: Any) -> bool:
    if getattr(event, "approved", None) is True:
        return True
    response = getattr(event, "response", None)
    return isinstance(response, str) and response.strip().lower() in _APPROVALS


def _context_types() -> tuple[Any, Any, Any] | None:
    try:
        from llama_index.core.workflow import Context, HumanResponseEvent, InputRequiredEvent
    except ImportError:
        return None
    return Context, HumanResponseEvent, InputRequiredEvent


def _spec_to_llamaindex(
    spec: ToolSpec,
    *,
    error_mode: ErrorMode = "model",
    error_formatter: ErrorFormatter | None = None,
) -> Any:
    """Convert a :class:`ToolSpec` into a LlamaIndex ``FunctionTool``.

    Args:
        spec: The tool to wrap.
        error_mode: ``"model"`` (failures raise :class:`LlamaIndexToolError`,
            gated calls wait for approval in a workflow) or ``"raise"``.
        error_formatter: Rewrites failure text before the model sees it.

    Raises:
        ImportError: If ``llama-index-core`` is not installed.
    """
    try:
        from llama_index.core.tools import FunctionTool
    except ImportError as exc:  # pragma: no cover - exercised via fake module
        raise ImportError(_INSTALL_HINT) from exc

    error_mode = check_error_mode(error_mode)
    model_mode = error_mode == "model"
    async_fn = spec.callable_for_signature()

    # Prefer an explicit args model built from the Gantry JSON schema:
    # LlamaIndex otherwise re-derives the schema from the wrapper's signature,
    # which flattens per-parameter descriptions, enums, and typed array items.
    from agent_gantry.integrations.frameworks.schema_bridge import (
        pydantic_model_from_schema,
    )

    fn_schema = None
    if spec.parameters.get("properties"):
        fn_schema = pydantic_model_from_schema(f"{spec.native_name}_Args", spec.parameters)

    def _coerced(kwargs: dict[str, Any]) -> dict[str, Any]:
        """Re-run the args model over ``kwargs`` before dispatch.

        LlamaIndex validates the call against ``fn_schema`` but forwards the
        caller's *original* values, so a model answering ``"1"`` for an
        ``integer`` parameter passed the tool's own validation and was then
        rejected by the executor, which holds the caller to the advertised
        schema. CrewAI forwards the validated values and never had this gap.
        Running the same model here closes it: what the engine sees is what
        the framework approved.

        Only the keys the caller actually supplied are rewritten — dumping the
        whole model would inject defaults for omitted parameters, and the
        handler's own defaults are what should apply there. Failures fall back
        to the original arguments rather than turning a call the framework
        accepted into an error; the executor validates either way.
        """
        if fn_schema is None or not kwargs:
            return kwargs
        try:
            # ``exclude_unset`` so the dump carries only what the caller
            # actually supplied. A plain dump recursively materializes every
            # omitted optional field — including inside a typed map's values,
            # where an optional child with no schema default becomes ``None``.
            # Executor normalization walks named ``properties`` but not
            # schema-valued map entries, so that injected null survived to be
            # rejected, turning a valid call into an error. Coercions on the
            # fields that *were* set are unaffected, which is what this
            # function exists for.
            dumped = fn_schema(**kwargs).model_dump(exclude_unset=True)
        except Exception:  # noqa: BLE001 - advisory; the executor still validates
            return kwargs
        return {key: dumped.get(key, value) for key, value in kwargs.items()}

    def _needs_approval() -> bool:
        gantry = spec._gantry
        return (
            model_mode
            and gantry is not None
            and gantry.requires_confirmation(spec.tool or spec.name, spec._namespace)
        )

    def _error(exc: ToolExecutionError) -> Exception:
        if not model_mode:
            return exc
        if isinstance(exc, ToolConfirmationRequiredError):
            return LlamaIndexToolError(exc, _PENDING_APPROVAL_TEXT, tool_name=spec.native_name)
        return LlamaIndexToolError(
            exc, failure_text(exc, error_formatter), tool_name=spec.native_name
        )

    def _run(ctx: Any, kwargs: dict[str, Any], approved: bool) -> Any:
        return execute_spec(
            spec,
            _coerced(kwargs),
            framework=FRAMEWORK_NAME,
            native_context=ctx,
            deps=getattr(ctx, "store", None) if ctx is not None else None,
            approved=approved,
        )

    signature = async_fn.__signature__  # type: ignore[attr-defined]
    # A tool with its own ``ctx`` argument keeps it; no workflow context then.
    inject_ctx = _context_types() is not None and "ctx" not in signature.parameters

    def _split(kwargs: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
        if not inject_ctx:
            return None, kwargs
        return kwargs.pop("ctx", None), kwargs

    async def _async_fn(**kwargs: Any) -> Any:
        ctx, kwargs = _split(kwargs)
        approved = False
        types = _context_types()
        if ctx is not None and types is not None and _needs_approval():
            _, human_response, input_required = types
            arguments = dict(kwargs)
            # The step re-runs from the top once the response arrives, so
            # nothing before this wait may have side effects.
            response = await ctx.wait_for_event(
                human_response,
                waiter_event=input_required(
                    prefix=f"Approve the call to {spec.native_name!r}?",
                    gantry_tool=spec.native_name,
                    gantry_arguments=arguments,
                ),
                waiter_id=f"gantry-approval:{spec.native_name}:"
                f"{json.dumps(arguments, sort_keys=True, default=str)}",
            )
            if not _is_approval(response):
                raise LlamaIndexToolError(None, _REJECTED_TEXT, tool_name=spec.native_name)
            approved = True
        try:
            return await _run(ctx, kwargs, approved)
        except ToolExecutionError as exc:
            raise _error(exc) from exc

    def _sync_fn(**kwargs: Any) -> Any:
        ctx, kwargs = _split(kwargs)
        try:
            return _run_coroutine_sync(_run(ctx, kwargs, False))
        except ToolExecutionError as exc:
            raise _error(exc) from exc

    annotations = dict(getattr(async_fn, "__annotations__", {}))
    partial_params: dict[str, Any] = {}
    types = _context_types()
    if inject_ctx and types is not None:
        context_cls = types[0]
        ctx_param = inspect.Parameter(
            "ctx", inspect.Parameter.KEYWORD_ONLY, default=None, annotation=context_cls
        )
        params = list(signature.parameters.values())
        positional = [p for p in params if p.kind is not inspect.Parameter.KEYWORD_ONLY]
        keyword = [p for p in params if p.kind is inspect.Parameter.KEYWORD_ONLY]
        try:
            signature = signature.replace(parameters=[*positional, ctx_param, *keyword])
            annotations["ctx"] = context_cls
            # The agent passes the real Context; a direct call gets None.
            partial_params["ctx"] = None
        except ValueError:  # pragma: no cover - exotic signature; no context
            pass

    for fn in (_async_fn, _sync_fn):
        fn.__name__ = spec.native_name
        fn.__doc__ = spec.description
        # Copy the real signature so LlamaIndex introspection surfaces the
        # actual parameters instead of a bare **kwargs (no-argument) tool.
        fn.__signature__ = signature  # type: ignore[attr-defined]
        fn.__annotations__ = annotations

    return FunctionTool.from_defaults(
        fn=_sync_fn,
        async_fn=_async_fn,
        name=spec.native_name,
        description=spec.description,
        fn_schema=fn_schema,
        partial_params=partial_params or None,
    )


class LlamaIndexAdapter(BaseFrameworkAdapter):
    """Route Gantry tools into LlamaIndex through its native mechanisms.

    Static slice (``FunctionTool`` objects from :meth:`select`) plus per-step
    live wiring (:meth:`tool_retriever` re-selects tools every reasoning step;
    :meth:`function_agent` builds a ``FunctionAgent`` with it). Every call
    routes through ``gantry.execute``; see the module docs for errors,
    approval, run context and tracing. :meth:`live` returns the
    :meth:`tool_retriever` — plug it into ``FunctionAgent(tool_retriever=<result>)``.

    Args:
        gantry: The registry to select from and execute against.
        default_limit: Default selection budget.
        error_mode: Default ``ErrorMode`` for the tools this adapter builds.
        error_formatter: Rewrites failure text before the model sees it.
    """

    live_tier = "per-turn"
    _live_delegate = "tool_retriever"

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
        """Wrap a single :class:`ToolSpec` as a LlamaIndex ``FunctionTool``."""
        return _spec_to_llamaindex(spec, error_mode=error_mode, error_formatter=error_formatter)

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
        """Select tools for ``query`` once, as LlamaIndex ``FunctionTool`` objects."""
        from agent_gantry.integrations.frameworks.base import GantryToolset

        mode = self._mode(error_mode)
        specs = await GantryToolset(self._gantry).select(
            query,
            tools_already_used=tools_already_used,
            **self._selection_kwargs(limit, score_threshold, namespaces, required, always_include),
        )
        return [
            _spec_to_llamaindex(s, error_mode=mode, error_formatter=self._error_formatter)
            for s in specs
        ]

    def tool_retriever(
        self,
        *,
        limit: int | None = None,
        score_threshold: float = 0.0,
        namespaces: list[str] | None = None,
        required: list[str] | None = None,
        always_include: list[str] | None = None,
        error_mode: ErrorMode | None = None,
    ) -> Any:
        """Build a live per-turn ``ObjectRetriever`` for ``FunctionAgent(tool_retriever=...)``."""
        from agent_gantry.integrations.frameworks.llamaindex_live import (
            _gantry_tool_retriever,
        )

        return _gantry_tool_retriever(
            self._gantry,
            **self._selection_kwargs(limit, score_threshold, namespaces, required, always_include),
            error_mode=self._mode(error_mode),
            error_formatter=self._error_formatter,
        )

    def function_agent(
        self,
        llm: Any,
        *,
        name: str = "gantry_agent",
        limit: int | None = None,
        score_threshold: float = 0.0,
        namespaces: list[str] | None = None,
        required: list[str] | None = None,
        always_include: list[str] | None = None,
        error_mode: ErrorMode | None = None,
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
            **self._selection_kwargs(limit, score_threshold, namespaces, required, always_include),
            error_mode=self._mode(error_mode),
            error_formatter=self._error_formatter,
            **agent_kwargs,
        )


__all__ = ["LlamaIndexAdapter", "LlamaIndexToolError", "current_workflow_context"]
