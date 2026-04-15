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
  invocation onto Gantry's telemetry span so token savings, latency, and
  success rate are captured uniformly whether the tool runs inside AF or
  directly through ``gantry.execute``.

The middlewares are defined without a hard import of ``agent_framework``
so the module can be imported (and type-checked) in environments where AF
is not installed. Actually attaching a middleware to an ``Agent`` still
requires the package.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

from agent_gantry.core.security import (
    ConfirmationRequiredError,
    PermissionDeniedError,
    SecurityPolicy,
)

if TYPE_CHECKING:
    from agent_gantry.core.gantry import AgentGantry

logger = logging.getLogger(__name__)


def _import_af_middleware_bits() -> tuple[Any, Any]:
    """Lazy import of AF symbols; raises a helpful ImportError otherwise."""
    try:
        from agent_framework import (  # type: ignore[import-not-found]
            FunctionMiddleware,
            MiddlewareTermination,
        )
    except Exception as exc:  # pragma: no cover - depends on install
        raise ImportError(
            "Agent-Framework middleware requires the 'agent-framework' package. "
            "Install with: pip install 'agent-gantry[agent-frameworks]'"
        ) from exc
    return FunctionMiddleware, MiddlewareTermination


def _build_middleware_classes() -> tuple[type, type]:
    """Construct the middleware subclasses lazily against the installed AF."""
    FunctionMiddleware, MiddlewareTermination = _import_af_middleware_bits()

    class _GantryApprovalMiddleware(FunctionMiddleware):  # type: ignore[misc,valid-type]
        """AF function middleware that enforces Gantry's SecurityPolicy.

        The middleware intercepts every Gantry-bridged tool call, applies
        :meth:`SecurityPolicy.check_permission` to its name+arguments, and:

        * lets execution proceed via ``await call_next()`` if the policy
          is satisfied;
        * raises :class:`~agent_gantry.core.security.PermissionDeniedError`
          if the call is outright denied (domain policy, rate limit);
        * raises ``agent_framework.MiddlewareTermination`` when the tool
          matches a ``require_confirmation`` pattern, so the agent surfaces
          the approval request rather than silently executing. AF presents
          a ``FunctionApprovalRequestContent`` to the caller in this case.

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

        def __init__(
            self,
            policy: SecurityPolicy,
            *,
            gantry: AgentGantry | None = None,
            deny_on_missing_capabilities: bool = False,
        ) -> None:
            super().__init__()
            self._policy = policy
            self._gantry = gantry
            self._deny_on_missing_capabilities = deny_on_missing_capabilities

        async def process(self, context: Any, call_next: Any) -> None:  # noqa: D401
            """Run the SecurityPolicy gate, then delegate to ``call_next``."""
            function = context.function
            name = getattr(function, "name", getattr(function, "__name__", "?"))
            args = context.arguments or {}
            try:
                self._policy.check_permission(
                    name,
                    {k: str(v) for k, v in args.items()},
                )
            except ConfirmationRequiredError as err:
                logger.info(
                    "GantryApprovalMiddleware: '%s' requires human approval (%s)",
                    name,
                    err,
                )
                # Surface the approval request to AF's native approval flow.
                raise MiddlewareTermination(str(err)) from err
            except PermissionDeniedError:
                logger.warning(
                    "GantryApprovalMiddleware: denied execution of '%s'", name
                )
                raise

            await call_next()

    class _GantryObservabilityMiddleware(FunctionMiddleware):  # type: ignore[misc,valid-type]
        """Record timing + success signals onto Gantry's telemetry."""

        def __init__(self, gantry: AgentGantry) -> None:
            super().__init__()
            self._gantry = gantry

        async def process(self, context: Any, call_next: Any) -> None:
            name = getattr(
                context.function, "name", getattr(context.function, "__name__", "?")
            )
            start = time.perf_counter()
            try:
                await call_next()
            finally:
                elapsed_ms = (time.perf_counter() - start) * 1000.0
                telemetry = getattr(self._gantry, "_telemetry", None)
                if telemetry is not None:
                    try:
                        telemetry.record(
                            "af_function_invocation",
                            {
                                "tool_name": name,
                                "duration_ms": elapsed_ms,
                                "has_result": context.result is not None,
                            },
                        )
                    except Exception:  # pragma: no cover - telemetry best-effort
                        logger.debug(
                            "GantryObservabilityMiddleware: telemetry.record failed",
                            exc_info=True,
                        )
                logger.debug(
                    "GantryObservabilityMiddleware: '%s' took %.2fms",
                    name,
                    elapsed_ms,
                )

    return _GantryApprovalMiddleware, _GantryObservabilityMiddleware


class GantryApprovalMiddleware:
    """Factory wrapper: returns a concrete AF middleware instance.

    Using a factory here avoids importing ``agent_framework`` at module
    import time, which keeps the package importable in environments that
    don't need AF. Instantiation fails loudly with an actionable
    ``ImportError`` when AF is missing.
    """

    def __new__(
        cls,
        policy: SecurityPolicy,
        *,
        gantry: AgentGantry | None = None,
        deny_on_missing_capabilities: bool = False,
    ) -> Any:
        ApprovalCls, _ = _build_middleware_classes()
        return ApprovalCls(
            policy,
            gantry=gantry,
            deny_on_missing_capabilities=deny_on_missing_capabilities,
        )


class GantryObservabilityMiddleware:
    """Factory wrapper for the observability middleware. See
    :class:`GantryApprovalMiddleware` for the rationale."""

    def __new__(cls, gantry: AgentGantry) -> Any:
        _, ObservabilityCls = _build_middleware_classes()
        return ObservabilityCls(gantry)


__all__ = [
    "GantryApprovalMiddleware",
    "GantryObservabilityMiddleware",
]
