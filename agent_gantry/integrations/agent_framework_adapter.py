"""Unified Microsoft Agent Framework adapter for Agent-Gantry.

:class:`AgentFrameworkAdapter` is the one-class entry point for the Microsoft
Agent Framework (AF 1.x) integration, mirroring the per-framework
``<Framework>Adapter`` classes. Bound to a single gantry, it constructs the
underlying Gantry AF primitives on demand:

- :meth:`context_provider` → ``GantryContextProvider`` (per-run / per-call
  dynamic tool injection — the first-class AF ``ContextProvider``),
- :meth:`tool_bridge` → ``GantryToolBridge`` (query-time tool retrieval and
  agent building),
- :meth:`approval_middleware` / :meth:`observability_middleware` /
  :meth:`tool_choice_middleware` → the AF middleware.

Importing this module never requires ``agent-framework``; the framework is
imported lazily by the underlying provider / bridge / middleware when used.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from agent_gantry.integrations.agent_framework_bridge import GantryToolBridge
from agent_gantry.integrations.agent_framework_middleware import (
    GantryApprovalMiddleware,
    GantryObservabilityMiddleware,
    GantryToolChoiceMiddleware,
)
from agent_gantry.integrations.agent_framework_provider import GantryContextProvider

if TYPE_CHECKING:
    from agent_gantry.core.gantry import AgentGantry
    from agent_gantry.core.security import SecurityPolicy


class AgentFrameworkAdapter:
    """Route Gantry into the Microsoft Agent Framework (AF 1.x).

    Example::

        from agent_framework import Agent
        from agent_framework.openai import OpenAIChatClient
        from agent_gantry.agent_framework import AgentFrameworkAdapter

        af = AgentFrameworkAdapter(gantry)

        # Per-run tool injection:
        provider = af.context_provider(top_k=5)
        agent = Agent(OpenAIChatClient(), "...", context_providers=[provider])

        # Per-call (multi-step) tool injection:
        provider = af.context_provider(top_k=3, query_strategy="per_call")
        provider.attach_to(agent)
    """

    def __init__(self, gantry: AgentGantry, *, default_top_k: int = 5) -> None:
        self._gantry = gantry
        self._default_top_k = default_top_k

    def context_provider(self, *, top_k: int | None = None, **kwargs: Any) -> Any:
        """Build a ``GantryContextProvider`` (per-run / per-call tool injection).

        ``top_k`` defaults to the adapter's ``default_top_k``. Every other keyword
        is forwarded to the provider (``query_strategy``, ``score_threshold``,
        ``required``, ``always_include``, ``static_tools``, ``verbose``,
        ``query_generator``, ...).
        """
        return GantryContextProvider(
            self._gantry,
            top_k=self._default_top_k if top_k is None else top_k,
            **kwargs,
        )

    def tool_bridge(self, **kwargs: Any) -> Any:
        """Build a ``GantryToolBridge`` (query-time tool retrieval / agent building)."""
        return GantryToolBridge(self._gantry, **kwargs)

    def approval_middleware(self, policy: SecurityPolicy) -> Any:
        """Build AF function middleware that enforces Gantry's ``SecurityPolicy``."""
        return GantryApprovalMiddleware(policy)

    def observability_middleware(self) -> Any:
        """Build AF function middleware that records Gantry telemetry spans."""
        return GantryObservabilityMiddleware(self._gantry)

    def tool_choice_middleware(self, decider: Any) -> Any:
        """Build AF chat middleware that modulates ``tool_choice`` per round."""
        return GantryToolChoiceMiddleware(decider)


__all__ = ["AgentFrameworkAdapter"]
