"""
Microsoft Agent Framework middleware powered by Agent-Gantry.

These middlewares plug into the ``Agent(middleware=[...])`` slot introduced
in Agent Framework 1.0 GA and let Gantry's security and observability
primitives govern tool execution without changing the rest of the agent
setup.

Available middleware:

- :class:`GantryApprovalMiddleware` — routes AF tool invocations through
  Gantry's :class:`~agent_gantry.core.security.SecurityPolicy`. A tool that
  requires human approval terminates the round with a
  ``function_approval_request`` result (activating AF's native approval
  flow; an approved replay executes, a rejected one reports without
  executing). A tool that is outright disallowed is never invoked — the
  model receives an explicit "Permission denied by security policy" result
  carrying the reason (raising instead would surface as AF's opaque
  ``"Error: Function failed."``).
- :class:`GantryObservabilityMiddleware` — records every function
  invocation onto Gantry's telemetry via ``telemetry.span(...)`` so token
  savings, latency, and success rate are captured uniformly whether the
  tool runs inside AF or directly through ``gantry.execute``.

The middlewares are defined without a hard import of ``agent_framework``
so the module can be imported (and type-checked) in environments where AF
is not installed. Actually attaching a middleware to an ``Agent`` still
requires the package.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import TYPE_CHECKING, Any

from agent_gantry.core.security import (
    ConfirmationRequiredError,
    PermissionDeniedError,
    SecurityPolicy,
    accepts_confirmation_approved,
)

if TYPE_CHECKING:
    from agent_gantry.core.gantry import AgentGantry

logger = logging.getLogger(__name__)


def _import_af_chat_middleware() -> Any:
    """Lazy import of ``agent_framework.chat_middleware``."""
    try:
        from agent_framework import chat_middleware
    except ImportError as exc:  # pragma: no cover - depends on install
        raise ImportError(
            "Agent-Framework chat middleware requires the 'agent-framework' "
            "package. Install with: pip install 'agent-gantry[agent-frameworks]'"
        ) from exc
    return chat_middleware


def _import_af_middleware_bits() -> tuple[Any, Any]:
    """Lazy import of AF symbols; raises a helpful ImportError otherwise.

    AF 1.x exposes middleware base classes under two names depending on context:
    - ``ChatMiddlewareLayer`` is the mixin for chat-client level middleware.
    - ``FunctionMiddleware`` is the base for tool/function execution middleware;
      it may be absent in some AF 1.x point releases, in which case we fall
      back to ``ChatMiddlewareLayer`` (they share the same ``process`` protocol).
    """
    try:
        from agent_framework import MiddlewareTermination

        # Prefer FunctionMiddleware (tool-execution middleware); fall back to
        # ChatMiddlewareLayer if not present in this AF 1.x release.
        try:
            from agent_framework import FunctionMiddleware as _MiddlewareBase
        except ImportError:
            from agent_framework import (
                ChatMiddlewareLayer as _MiddlewareBase,  # type: ignore[assignment]
            )
    except Exception as exc:  # pragma: no cover - depends on install
        raise ImportError(
            "Agent-Framework middleware requires the 'agent-framework' package. "
            "Install with: pip install 'agent-gantry[agent-frameworks]'"
        ) from exc
    return _MiddlewareBase, MiddlewareTermination


@lru_cache(maxsize=1)
def _build_middleware_classes() -> tuple[type, type]:
    """Build the middleware subclasses once per process.

    AF's ``FunctionMiddleware`` is only importable when the extra is
    installed, so we can't subclass at module import time without making
    ``agent-framework`` a hard dependency. ``lru_cache`` makes this a
    module-level singleton: the classes are constructed on the first call
    and reused thereafter, preserving type identity across
    ``isinstance`` checks and eliminating per-instantiation overhead.
    """
    function_middleware, middleware_termination = _import_af_middleware_bits()
    try:
        from agent_framework import Content as _AFContent
    except ImportError:  # pragma: no cover - Content ships with every AF 1.x
        _AFContent = None  # noqa: N806 - mirrors the class-name import above

    def _approval_request(call_id: str, name: str, args: Any) -> Any:
        """Build the ``function_approval_request`` Content AF's flow expects.

        Returns ``None`` when it can't be constructed (AF double without
        ``Content``); callers then fall back to a plain-string result so the
        approval *reason* still survives the termination.
        """
        if _AFContent is None:
            return None
        try:
            if hasattr(args, "model_dump"):
                args = args.model_dump()
            function_call = _AFContent.from_function_call(
                call_id=call_id, name=name, arguments=dict(args) if args else {}
            )
            return _AFContent.from_function_approval_request(
                id=call_id, function_call=function_call
            )
        except Exception:  # pragma: no cover - best-effort; string fallback
            logger.debug(
                "GantryApprovalMiddleware: could not build a "
                "function_approval_request Content; falling back to a plain "
                "result message.",
                exc_info=True,
            )
            return None

    class GantryApprovalMiddlewareImpl(function_middleware):  # type: ignore[misc,valid-type]
        """AF function middleware that enforces Gantry's SecurityPolicy.

        See :class:`GantryApprovalMiddleware` for the public entry point
        and full docstring.
        """

        def __init__(self, policy: SecurityPolicy) -> None:
            super().__init__()
            self._policy = policy

        async def process(self, context: Any, call_next: Any) -> None:  # noqa: D401
            """Run the SecurityPolicy gate, then delegate to ``call_next``."""
            function = context.function
            name = getattr(function, "name", getattr(function, "__name__", "?"))
            args = context.arguments or {}
            metadata = getattr(context, "metadata", None) or {}
            # AF replays an approved/rejected call with the human's response in
            # metadata — that replay is how a confirmation-gated tool actually
            # runs, so honour it before re-running the policy gate.
            approval_response = metadata.get("approval_response")
            approved: bool | None = getattr(approval_response, "approved", None)

            if approved is False:
                logger.info(
                    "GantryApprovalMiddleware: human rejected approval for '%s'",
                    name,
                )
                context.result = (
                    f"Tool '{name}' was not executed: the approval request "
                    "was rejected by the human reviewer."
                )
                return

            try:
                # Pass raw arguments through: SecurityPolicy type-checks per
                # value, and downstream policies may rely on numeric/boolean
                # typing rather than stringified values.
                #
                # An approved replay passes ``confirmation_approved=True`` so
                # only the confirmation-pattern gate is skipped and every
                # *denial* check (rate limit, allowed domains) still runs —
                # swallowing the ConfirmationRequiredError instead would skip
                # them too, because the policy raises it before the domain
                # check is reached, silently turning a human's confirmation
                # into a domain-allowlist bypass. The keyword is passed only
                # when the policy's signature declares it (duck-typed
                # policies predating it fall back to the swallow branch
                # below, matching their pre-keyword behaviour).
                if accepts_confirmation_approved(self._policy):
                    self._policy.check_permission(
                        name, args, confirmation_approved=approved is True
                    )
                else:
                    self._policy.check_permission(name, args)
            except ConfirmationRequiredError as err:
                if approved is True:
                    # Replay of an approved request on a policy without the
                    # ``confirmation_approved`` keyword: the human said yes.
                    logger.info(
                        "GantryApprovalMiddleware: '%s' approved by human; "
                        "proceeding.",
                        name,
                    )
                else:
                    logger.info(
                        "GantryApprovalMiddleware: '%s' requires human "
                        "approval (%s)",
                        name,
                        err,
                    )
                    # Surface the approval request to AF's native approval
                    # flow: AF only activates it when ``context.result`` is a
                    # ``function_approval_request`` Content at termination —
                    # a bare MiddlewareTermination would reach the model as a
                    # null function result with the reason discarded.
                    call_id = str(metadata.get("call_id") or f"gantry-{name}")
                    request = _approval_request(call_id, name, args)
                    context.result = request if request is not None else str(err)
                    raise middleware_termination(str(err)) from err
            except PermissionDeniedError as err:
                logger.warning(
                    "GantryApprovalMiddleware: denied execution of '%s'", name
                )
                # An explicit result — NOT a raise: AF converts an exception
                # from function middleware into an opaque "Error: Function
                # failed." (detail only with include_detailed_errors), which
                # discards the denial reason and reads as a tool crash. The
                # tool is never invoked; the model sees exactly why.
                context.result = (
                    f"Permission denied by security policy: tool '{name}' was "
                    f"not executed. {err}"
                )
                return

            await call_next()

    class GantryObservabilityMiddlewareImpl(function_middleware):  # type: ignore[misc,valid-type]
        """Record timing + success signals onto Gantry's telemetry span."""

        def __init__(self, gantry: AgentGantry) -> None:
            super().__init__()
            self._gantry = gantry

        async def process(self, context: Any, call_next: Any) -> None:
            name = getattr(
                context.function, "name", getattr(context.function, "__name__", "?")
            )
            telemetry = getattr(self._gantry, "_telemetry", None)
            if telemetry is None:
                await call_next()
                return

            # Wrap the downstream call in a Gantry telemetry span so AF
            # tool invocations appear alongside every other Gantry-traced
            # operation. Adapters like the OpenTelemetry and console
            # adapters all implement ``span`` as an async context manager.
            try:
                span_cm = telemetry.span(
                    "af_function_invocation", {"tool_name": name}
                )
            except Exception:  # pragma: no cover - telemetry best-effort
                logger.debug(
                    "GantryObservabilityMiddleware: telemetry.span failed",
                    exc_info=True,
                )
                await call_next()
                return

            async with span_cm:
                await call_next()

    return GantryApprovalMiddlewareImpl, GantryObservabilityMiddlewareImpl


class GantryApprovalMiddleware:
    """Factory wrapper: returns a concrete AF middleware instance.

    Using a factory here avoids importing ``agent_framework`` at module
    import time, which keeps the package importable in environments that
    don't need AF. Instantiation fails loudly with an actionable
    ``ImportError`` when AF is missing.

    Example:
        .. code-block:: python

            from agent_gantry.integrations.agent_framework_bridge import (
                GantryToolBridge,
            )
            from agent_gantry.integrations.agent_framework_middleware import (
                GantryApprovalMiddleware,
            )
            from agent_gantry.core.security import SecurityPolicy

            policy = SecurityPolicy(
                require_confirmation=["delete_*", "refund_*"],
                allowed_domains=["example.com"],
            )
            bridge = GantryToolBridge(gantry)
            agent = await bridge.build_agent(
                client,
                query,
                name="SupportAgent",
                instructions="...",
                middleware=[GantryApprovalMiddleware(policy)],
            )
    """

    def __new__(cls, policy: SecurityPolicy) -> Any:
        approval_cls, _ = _build_middleware_classes()
        return approval_cls(policy)


class GantryObservabilityMiddleware:
    """Factory wrapper for the observability middleware.

    Records every AF function invocation as a ``gantry.af_function_invocation``
    telemetry span, exposing tool name, latency, and success signals through
    whatever telemetry adapter is configured on the Gantry instance
    (console, OpenTelemetry, custom).

    **agent-framework ≥ 1.6.0 note — instrumentation enabled by default:**
    Agent Framework 1.6.0 ships instrumentation enabled by default, meaning
    AF will emit its own OpenTelemetry spans for every function invocation
    (covering invocation timing and argument/result shapes).  When a user
    also attaches ``GantryObservabilityMiddleware`` *and* configures an OTel
    exporter, both span trees will appear in the tracing backend — one from
    AF (invocation lifecycle) and one from Gantry (tool intent and result).
    This is intentional: the two spans are complementary rather than
    duplicates.  If you want only one source of truth, either omit this
    middleware (rely on AF's built-in instrumentation) or suppress AF's
    instrumentation via Gantry's helper::

        from agent_gantry import disable_af_instrumentation
        disable_af_instrumentation()   # call once before agent construction

    .. note::
        AF 1.6.0 has a known upstream bug affecting *concurrent* ``Agent.run()``
        calls via ``asyncio.gather()`` / ``TaskGroup``: a ``ContextVar`` token
        is reset in a different asyncio context than it was created, raising
        ``ValueError``.  Sequential workflows (``WorkflowAgent``,
        ``SequentialBuilder``, ``HandoffBuilder``) are **not** affected.
        For concurrent patterns on AF 1.6.0 call
        :func:`~agent_gantry.disable_af_instrumentation` (or pass
        ``disable_af_instrumentation=True`` to ``GantryToolBridge``) once
        at startup to disable AF's instrumentation.

    See :class:`GantryApprovalMiddleware` for the factory-pattern rationale.
    """

    def __new__(cls, gantry: AgentGantry) -> Any:
        _, observability_cls = _build_middleware_classes()
        return observability_cls(gantry)


class GantryToolChoiceMiddleware:
    """Chat middleware that modulates ``tool_choice`` per round.

    AF's ``tool_choice`` is a single global setting on the agent; a
    common pattern is "force a tool call for the first N rounds of a
    pipeline so the model can't bail to text, then allow text on the
    summarisation turn". This middleware solves that by re-deriving
    ``tool_choice`` on every chat-completion round from a user-supplied
    callable.

    The callable receives the AF chat-middleware ``context`` and may
    return any of:

    - ``"auto"`` (let the model choose),
    - ``"required"`` (force a tool call),
    - ``"none"`` (text only),
    - A dict in AF / OpenAI tool-choice shape (e.g.
      ``{"type": "function", "function": {"name": "..."}}``),
    - ``None`` (don't touch the existing value).

    The callable may be sync or async. Counting rounds is left to the
    caller — the simplest pattern is to close over a counter::

        rounds = {"n": 0}
        def choice(ctx):
            rounds["n"] += 1
            return "required" if rounds["n"] <= 5 else "auto"
        agent = Agent(client, "...", middleware=[
            provider.as_chat_middleware(),
            GantryToolChoiceMiddleware(choice),
        ])
    """

    def __new__(cls, decider: Any) -> Any:
        chat_middleware = _import_af_chat_middleware()
        import inspect

        is_async = inspect.iscoroutinefunction(decider)

        @chat_middleware
        async def _tool_choice_mw(context: Any, call_next: Any) -> None:
            try:
                choice = (
                    await decider(context) if is_async else decider(context)
                )
            except Exception:
                logger.exception(
                    "GantryToolChoiceMiddleware: decider raised; leaving "
                    "tool_choice unchanged."
                )
                choice = None

            if choice is not None:
                options = getattr(context, "options", None)
                if isinstance(options, dict):
                    options["tool_choice"] = choice
                elif options is not None:
                    try:
                        setattr(options, "tool_choice", choice)
                    except (AttributeError, TypeError, ValueError):
                        if hasattr(options, "model_copy"):
                            try:
                                context.options = options.model_copy(
                                    update={"tool_choice": choice}
                                )
                            except Exception:
                                logger.warning(
                                    "GantryToolChoiceMiddleware: could not "
                                    "set tool_choice on options of type %r.",
                                    type(options).__name__,
                                )

            await call_next()

        return _tool_choice_mw


__all__ = [
    "GantryApprovalMiddleware",
    "GantryObservabilityMiddleware",
    "GantryToolChoiceMiddleware",
]
