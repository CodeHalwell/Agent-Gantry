"""
Microsoft Agent Framework middleware powered by Agent-Gantry.

These middlewares plug into the ``Agent(middleware=[...])`` slot introduced
in Agent Framework 1.0 GA and let Gantry's security and observability
primitives govern tool execution without changing the rest of the agent
setup.

Available middleware:

- :class:`GantryApprovalMiddleware` — routes AF tool invocations through
  Gantry's :class:`~agent_gantry.core.security.SecurityPolicy`, raising
  ``MiddlewareTermination`` for tools that require human approval and
  ``PermissionDeniedError`` for tools that are outright disallowed.
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
            try:
                # Pass raw arguments through: SecurityPolicy type-checks per
                # value, and downstream policies may rely on numeric/boolean
                # typing rather than stringified values.
                self._policy.check_permission(name, args)
            except ConfirmationRequiredError as err:
                logger.info(
                    "GantryApprovalMiddleware: '%s' requires human approval (%s)",
                    name,
                    err,
                )
                # Surface the approval request to AF's native approval flow.
                raise middleware_termination(str(err)) from err
            except PermissionDeniedError:
                logger.warning(
                    "GantryApprovalMiddleware: denied execution of '%s'", name
                )
                raise

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
    """Factory wrapper for the observability middleware. See
    :class:`GantryApprovalMiddleware` for the rationale."""

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
