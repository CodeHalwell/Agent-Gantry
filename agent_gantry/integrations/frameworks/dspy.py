"""DSPy native tool adapter for Agent-Gantry (``dspy>=3.5``).

Selects a relevant slice of Gantry tools and wraps each as a ``dspy.Tool``,
the tool object ``dspy.ReAct`` introspects and calls. The ``dspy`` import is
lazy, so ``import agent_gantry`` never requires DSPy.

Public entry point: :class:`DSPyAdapter`.

Errors
------
``dspy.ReAct`` (3.5) records a tool that raises as an error tool result
(``is_error=True``, ``"Execution error in <tool>: ..."``) and gives the model
another turn. In the default ``error_mode="model"`` a failed Gantry call raises
:class:`DSPyToolError`, whose text is ``"<type>: <message>"`` (rewritten by
``error_formatter``, capped). DSPy itself adds a short traceback (up to five
frames) to every tool error it shows the model; it ends with that text, and the
original Gantry error is not chained onto it, so a redacted message stays
redacted. ``error_mode="raise"`` raises the plain
:class:`~agent_gantry.integrations.frameworks.base.ToolExecutionError`.

Approval
--------
DSPy has no human-in-the-loop mechanism of its own. ``DSPyAdapter(gantry,
approve=fn)`` decides confirmation-gated calls (``requires_confirmation=True``
or a ``SecurityPolicy`` pattern): ``fn`` gets an :class:`ApprovalRequest` and
returns ``True`` to run the call, once, with Gantry's confirmation gate
cleared (every other policy check still applies). A refused call, or a gated
call with no approver, is reported to the model as not run.

Run context
-----------
:func:`current_dspy_module` gives the innermost DSPy module running the call
(normally the ``dspy.ReAct``), also available as
``current_tool_invocation().native_context``.

Synchronous tool bodies
-----------------------
Each tool body is synchronous, bridged onto Gantry's async ``execute``, so it
works under both ``react(...)`` (``dspy.Tool.__call__``) and ``await
react.acall(...)`` (``dspy.Tool.acall`` returns a non-coroutine result as is)
with no ``allow_tool_async_sync_conversion`` setting.

Per-run selection
------------------
``dspy.ReAct`` fixes its tools, and bakes their names into its instructions,
at construction. :meth:`DSPyAdapter.agent_builder` therefore builds a fresh
``dspy.ReAct`` per call with tools selected for that call's query; pins are
still offered if retrieval fails.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from agent_gantry.adapters.selectors.base import SelectorFallbackError
from agent_gantry.core.invocation import current_tool_invocation
from agent_gantry.integrations.frameworks.base import (
    DEFAULT_TOOL_LIMIT,
    PER_RUN_TIER,
    BaseFrameworkAdapter,
    GantryToolset,
    ToolConfirmationRequiredError,
    ToolExecutionError,
    ToolSpec,
    _resolve_pins,
    _run_coroutine_sync,
    _with_native_names,
)
from agent_gantry.integrations.frameworks.errors import MissingRequiredToolError
from agent_gantry.integrations.frameworks.live_wrappers import _PerCallBuilder
from agent_gantry.integrations.frameworks.native import (
    ErrorFormatter,
    ErrorMode,
    check_error_mode,
    execute_spec,
    failure_text,
    otel_ids,
)

if TYPE_CHECKING:
    from agent_gantry.core.gantry import AgentGantry

logger = logging.getLogger(__name__)

FRAMEWORK_NAME = "dspy"

_INSTALL_HINT = (
    "DSPy support requires `dspy>=3.5`. Install it with `pip install 'agent-gantry[dspy]'` "
    "(or `uv add 'agent-gantry[dspy]'`)."
)

_PENDING_APPROVAL_TEXT = (
    "This tool call requires human approval and was not run. Ask the user to "
    "approve it, or continue without it."
)
_DECLINED_TEXT = "The user declined this tool call, so it was not run. Continue without it."


class DSPyToolError(ToolExecutionError):
    """A failed Gantry call as DSPy shows it to the model.

    ``str()`` is the model-facing failure text; the attributes are those of
    the original :class:`~agent_gantry.integrations.frameworks.base.ToolExecutionError`.
    """

    def __init__(self, source: ToolExecutionError, text: str) -> None:
        super().__init__(source.tool_name, source.status, source.error, source.error_type)
        self.text = text

    def __str__(self) -> str:
        return self.text


# Its public import path, which is also what ``dspy.ReAct`` prints before the text.
DSPyToolError.__module__ = "agent_gantry.dspy"


@dataclass(frozen=True)
class ApprovalRequest:
    """A confirmation-gated Gantry tool call waiting for a decision.

    Attributes:
        tool_name: The tool's name as the model sees it.
        arguments: The arguments the model called it with.
        description: The tool's description.
        module: The innermost DSPy module running the call (usually the
            ``dspy.ReAct``), when known.
    """

    tool_name: str
    arguments: dict[str, Any]
    description: str = ""
    module: Any = field(default=None, repr=False)


#: Decides an :class:`ApprovalRequest`: ``True`` runs the call, anything else blocks it.
Approver = Callable[[ApprovalRequest], bool]


def current_dspy_module() -> Any:
    """The innermost DSPy module running the Gantry tool currently executing.

    Normally the ``dspy.ReAct`` that called the tool. ``None`` outside a DSPy
    tool call, or when the tool is called directly.
    """
    invocation = current_tool_invocation()
    if invocation is None or invocation.framework != FRAMEWORK_NAME:
        return None
    return invocation.native_context


def _calling_module() -> Any:
    """The innermost module on DSPy's call stack (read on the caller's thread)."""
    try:
        from dspy.dsp.utils.settings import settings
    except ImportError:  # pragma: no cover - dspy layout changed
        return None
    modules = getattr(settings, "caller_modules", None) or []
    return modules[-1] if modules else None


def _dspy_args(spec: ToolSpec) -> tuple[dict[str, Any], dict[str, Any], dict[str, str]]:
    """``dspy.Tool`` ``args`` / ``arg_types`` / ``arg_desc`` from the spec's JSON Schema."""
    from dspy.adapters.types.tool import convert_input_schema_to_tool_args

    # DSPy's converter looks up a property's ``type`` in a dict. JSON Schema
    # also permits a list (e.g. ["string", "null"]), which raises TypeError in
    # DSPy. Omit only that type during conversion so DSPy uses Any for runtime
    # parsing, then restore the union in the advertised schema. DSPy's
    # JSON-Schema validator and Gantry still enforce the full union.
    properties = spec.parameters.get("properties") or {}
    unions = {
        name: prop["type"]
        for name, prop in properties.items()
        if isinstance(prop, dict) and isinstance(prop.get("type"), list)
    }
    parameters = spec.parameters
    if unions:
        parameters = {
            **parameters,
            "properties": {
                name: {key: value for key, value in prop.items() if key != "type"}
                if name in unions
                else prop
                for name, prop in properties.items()
            },
        }
    args, arg_types, arg_desc = convert_input_schema_to_tool_args(parameters)
    for name, types in unions.items():
        args[name] = {**args[name], "type": types}
    return args, arg_types, arg_desc


def _spec_to_dspy(
    spec: ToolSpec,
    *,
    error_mode: ErrorMode = "model",
    error_formatter: ErrorFormatter | None = None,
    approve: Approver | None = None,
) -> Any:
    """Wrap a :class:`ToolSpec` as a ``dspy.Tool``.

    Args:
        spec: The tool to wrap.
        error_mode: ``"model"`` (failures raise :class:`DSPyToolError`, gated
            calls go through ``approve``) or ``"raise"``.
        error_formatter: Rewrites failure text before the model sees it.
        approve: Decides confirmation-gated calls; ``None`` reports them as
            needing approval.

    Raises:
        ImportError: If ``dspy`` is not installed.
    """
    try:
        from dspy.adapters.types.tool import Tool as DSPyTool
    except ImportError as exc:  # pragma: no cover - exercised via stub
        raise ImportError(_INSTALL_HINT) from exc

    error_mode = check_error_mode(error_mode)
    model_mode = error_mode == "model"
    args, arg_types, arg_desc = _dspy_args(spec)
    gantry = spec._gantry
    tool_ref = spec.tool or spec.name
    namespace = spec._namespace

    def _decide(module: Any, kwargs: dict[str, Any]) -> bool:
        """Ask the approver about a gated call; ``True`` means run it approved."""
        if not model_mode or approve is None or gantry is None:
            return False
        if not gantry.requires_confirmation(tool_ref, namespace):
            return False
        request = ApprovalRequest(
            tool_name=spec.native_name,
            arguments=dict(kwargs),
            description=spec.description,
            module=module,
        )
        try:
            decision = approve(request)
        except Exception:
            logger.warning(
                "DSPy approver for %r raised; blocking the call.", spec.native_name, exc_info=True
            )
            decision = False
        if decision is not True:
            source = ToolExecutionError(
                spec.native_name, "declined", "declined by the approver", "ConfirmationDeclined"
            )
            raise DSPyToolError(source, _DECLINED_TEXT)
        return True

    def _fn(**kwargs: Any) -> Any:
        module = _calling_module()  # DSPy's settings are per thread: read before bridging
        approved = _decide(module, kwargs)
        try:
            return _run_coroutine_sync(
                execute_spec(
                    spec,
                    kwargs,
                    framework=FRAMEWORK_NAME,
                    native_context=module,
                    metadata={"module": type(module).__name__} if module is not None else None,
                    approved=approved,
                    trace_ids=otel_ids(),
                )
            )
        except ToolExecutionError as exc:
            if not model_mode:
                raise
            if isinstance(exc, ToolConfirmationRequiredError):
                raise DSPyToolError(exc, _PENDING_APPROVAL_TEXT) from None
            raise DSPyToolError(exc, failure_text(exc, error_formatter)) from None

    # Borrow the real __signature__/__annotations__ from the async wrapper so
    # introspection sees the actual parameters; only the body differs.
    async_fn = spec.callable_for_signature()
    _fn.__name__ = spec.native_name
    _fn.__doc__ = spec.description
    _fn.__signature__ = async_fn.__signature__  # type: ignore[attr-defined]
    _fn.__annotations__ = dict(getattr(async_fn, "__annotations__", {}))

    return DSPyTool(
        func=_fn,
        name=spec.native_name,
        desc=spec.description,
        args=args,
        arg_types=arg_types,
        arg_desc=arg_desc,
    )


class DSPyAdapter(BaseFrameworkAdapter):
    """Route Gantry-selected tools into DSPy.

    Static slice (``dspy.Tool`` objects) plus a per-run builder (a fresh
    ``dspy.ReAct`` per call). Every call routes through ``gantry.execute`` so
    retries, timeouts, circuit breakers, and the security policy all apply::

        from agent_gantry import AgentGantry
        from agent_gantry.dspy import DSPyAdapter
        import dspy

        dspy.configure(lm=dspy.LM("openai/gpt-5.5"))

        gantry = AgentGantry()
        # ... register tools, await gantry.sync() ...

        tools = await DSPyAdapter(gantry).select("what's the weather in Tokyo?", limit=3)
        react = dspy.ReAct("question -> answer", tools=tools)
        pred = react(question="what's the weather in Tokyo?")

    :meth:`live` requires ``signature=`` (the DSPy task signature) and returns
    the :meth:`agent_builder` builder; call ``await builder.build(query)`` per
    task.

    Args:
        gantry: The registry to select from and execute against.
        default_limit: Default selection budget.
        error_mode: ``"model"`` (default) raises :class:`DSPyToolError`, which
            ``dspy.ReAct`` records as an error tool result; ``"raise"`` raises
            the plain ``ToolExecutionError`` and never asks ``approve``.
        error_formatter: Rewrites the failure text the model sees (fail-closed,
            capped).
        approve: Decides confirmation-gated calls: a callable taking an
            :class:`ApprovalRequest` and returning ``True`` to allow. ``None``
            (default) tells the model the call needs approval.
    """

    live_tier = PER_RUN_TIER
    _live_delegate = "agent_builder"
    _live_required_kwargs = ("signature",)

    def __init__(
        self,
        gantry: AgentGantry,
        *,
        default_limit: int = DEFAULT_TOOL_LIMIT,
        error_mode: ErrorMode = "model",
        error_formatter: ErrorFormatter | None = None,
        approve: Approver | None = None,
    ) -> None:
        super().__init__(gantry, default_limit=default_limit)
        self._error_mode = check_error_mode(error_mode)
        self._error_formatter = error_formatter
        if approve is not None and not callable(approve):
            raise ValueError("approve must be a callable or None")
        self._approve = approve

    @staticmethod
    def convert(
        spec: ToolSpec,
        *,
        error_mode: ErrorMode = "model",
        error_formatter: ErrorFormatter | None = None,
        approve: Approver | None = None,
    ) -> Any:
        """Wrap a single :class:`ToolSpec` as a ``dspy.Tool``."""
        return _spec_to_dspy(
            spec, error_mode=error_mode, error_formatter=error_formatter, approve=approve
        )

    def _convert(self, spec: ToolSpec, error_mode: ErrorMode | None = None) -> Any:
        mode = self._error_mode if error_mode is None else check_error_mode(error_mode)
        return _spec_to_dspy(
            spec, error_mode=mode, error_formatter=self._error_formatter, approve=self._approve
        )

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
        """Select tools for ``query`` once, as ``dspy.Tool`` objects."""
        specs = await GantryToolset(self._gantry).select(
            query,
            tools_already_used=tools_already_used,
            **self._selection_kwargs(limit, score_threshold, namespaces, required, always_include),
        )
        return [self._convert(spec, error_mode) for spec in specs]

    def agent_builder(
        self,
        signature: Any,
        *,
        max_iters: int = 20,
        limit: int | None = None,
        score_threshold: float = 0.0,
        namespaces: list[str] | None = None,
        required: list[str] | None = None,
        always_include: list[str] | None = None,
        **react_kwargs: Any,
    ) -> Any:
        """Return a builder that builds a fresh ``dspy.ReAct`` per call with re-selected tools.

        ``signature`` and ``react_kwargs`` are forwarded to ``dspy.ReAct``.
        Call ``await builder.build(query)`` per task. The builder uses this
        adapter's ``error_mode``, ``error_formatter`` and ``approve``.
        """
        builder = GantryLiveDSPyReAct(
            self._gantry,
            signature,
            max_iters=max_iters,
            **self._selection_kwargs(limit, score_threshold, namespaces, required, always_include),
            **react_kwargs,
        )
        builder._adapter = self
        return builder


class GantryLiveDSPyReAct(_PerCallBuilder):
    """Build a fresh ``dspy.ReAct`` per call, tools re-selected by Gantry.

    Obtain one via ``DSPyAdapter(gantry).agent_builder(signature, ...)``.

    Args:
        gantry: The :class:`~agent_gantry.core.gantry.AgentGantry` to select from.
        signature: DSPy task signature (an ``"input -> output"`` string or a
            ``dspy.Signature`` subclass), forwarded to ``dspy.ReAct``.
        max_iters: Max tool-calling turns per ``dspy.ReAct`` run.
        limit: Max tools to surface per call. Defaults to ``DEFAULT_TOOL_LIMIT``.
        score_threshold: Minimum semantic relevance score. Defaults to ``0.0``.
        **react_kwargs: Extra kwargs forwarded to ``dspy.ReAct``.
    """

    _adapter_cls = DSPyAdapter
    _adapter: DSPyAdapter

    def __init__(
        self, gantry: AgentGantry, signature: Any, *, max_iters: int = 20, **kwargs: Any
    ) -> None:
        super().__init__(gantry, **kwargs)
        self._signature = signature
        self._framework_kwargs["max_iters"] = max_iters

    async def select_tools(self, query: str) -> list[Any]:
        """This call's tools; the pinned ones if retrieval fails."""
        adapter = self._adapter
        kwargs = self._select_kwargs
        try:
            specs = await GantryToolset(adapter._gantry).select_or_empty(query, **kwargs)
        except (MissingRequiredToolError, SelectorFallbackError):
            raise
        except Exception:
            logger.warning(
                "DSPy: semantic retrieval failed; offering pinned tools only", exc_info=True
            )
            specs = _with_native_names(
                adapter._gantry,
                _resolve_pins(
                    adapter._gantry,
                    [],
                    required=kwargs.get("required"),
                    always_include=kwargs.get("always_include"),
                    namespaces=kwargs.get("namespaces"),
                ),
            )
        return [adapter._convert(spec) for spec in specs]

    def _construct(self, tools: list[Any]) -> Any:
        try:
            from dspy import ReAct
        except ImportError as exc:  # pragma: no cover - exercised via importorskip
            raise ImportError(_INSTALL_HINT) from exc
        return ReAct(self._signature, tools=tools, **self._framework_kwargs)


__all__ = [
    "ApprovalRequest",
    "Approver",
    "DSPyAdapter",
    "DSPyToolError",
    "GantryLiveDSPyReAct",
    "current_dspy_module",
]
