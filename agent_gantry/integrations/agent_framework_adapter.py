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
from agent_gantry.integrations.frameworks.base import DEFAULT_TOOL_LIMIT

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

    Not a :class:`~agent_gantry.integrations.frameworks.base.BaseFrameworkAdapter`
    subclass (it has no ``select``/``convert`` staticmethods — see
    ``tests/frameworks/test_conformance.py``), but it participates in the same
    uniform ``live_tier`` / :meth:`live` facade the 13 ``<Framework>Adapter``
    classes expose, since AF genuinely supports per-call (multi-round) dynamic
    tool re-selection via ``query_strategy="per_call"``.
    """

    #: AF's ``GantryContextProvider`` supports genuine per-round re-selection
    #: (``query_strategy="per_call"``), matching the other per-turn adapters.
    live_tier = "per-turn"

    def __init__(self, gantry: AgentGantry, *, default_top_k: int = DEFAULT_TOOL_LIMIT) -> None:
        self._gantry = gantry
        self._default_top_k = default_top_k

    def live(
        self,
        *,
        limit: int | None = None,
        score_threshold: float = 0.0,
        namespaces: list[str] | None = None,
        **framework_kwargs: Any,
    ) -> Any:
        """Per-turn uniform entry point: delegates to :meth:`context_provider`.

        Same ``limit``/``score_threshold``/``namespaces``/``**framework_kwargs``
        shape as every ``<Framework>Adapter.live()`` (``namespaces`` is
        forwarded as an ordinary AF ``query_kwargs`` entry, to
        ``GantryToolBridge.get_tools``). Unlike calling
        :meth:`context_provider` directly, this defaults ``query_strategy``
        to ``"per_call"`` (not :meth:`context_provider`'s own
        back-compatible ``"per_run"`` default) — ``live()`` is meant to
        surface the *deepest* tier AF supports, and per-call is what makes
        AF's re-selection genuinely per-turn. Pass ``query_strategy="per_run"``
        in ``framework_kwargs`` to opt back out.

        Returns a ``GantryContextProvider``. Plug it into
        ``Agent(context_providers=[<result>])`` and, for ``per_call``, also
        attach ``<result>.as_chat_middleware()`` — or call
        ``<result>.attach_to(agent)`` to do both in one step.
        """
        framework_kwargs.setdefault("query_strategy", "per_call")
        if namespaces is not None:
            framework_kwargs.setdefault("namespaces", namespaces)
        return self.context_provider(
            top_k=limit, score_threshold=score_threshold, **framework_kwargs
        )

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
