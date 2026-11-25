"""Unified Microsoft Agent Framework adapter for dotfetch.

:class:`AgentFrameworkAdapter` is the one-class entry point for the Microsoft
Agent Framework (AF 1.x) integration, mirroring the per-framework
``<Framework>Adapter`` classes. Bound to a single fetcher, it constructs the
underlying dotfetch AF primitives on demand:

- :meth:`context_provider` → ``DotFetchContextProvider`` (per-run / per-turn
  dynamic tool injection — the first-class AF ``ContextProvider``),
- :meth:`tool_bridge` → ``DotFetchToolBridge`` (query-time tool retrieval and
  agent building),
- :meth:`approval_middleware` / :meth:`observability_middleware` /
  :meth:`tool_choice_middleware` → the AF middleware.

Importing this module never requires ``agent-framework``; the framework is
imported lazily by the underlying provider / bridge / middleware when used.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from dotfetch.integrations.agent_framework_bridge import (
    DotFetchToolBridge,
    _build_callable_for_tool,
)
from dotfetch.integrations.agent_framework_middleware import (
    DotFetchApprovalMiddleware,
    DotFetchObservabilityMiddleware,
    DotFetchToolChoiceMiddleware,
)
from dotfetch.integrations.agent_framework_provider import DotFetchContextProvider
from dotfetch.integrations.frameworks.base import (
    DEFAULT_TOOL_LIMIT,
    BaseFrameworkAdapter,
    ToolSpec,
)
from dotfetch.schema.tool import ToolDefinition

if TYPE_CHECKING:
    from dotfetch.core.fetcher import DotFetch
    from dotfetch.core.security import SecurityPolicy


class AgentFrameworkAdapter(BaseFrameworkAdapter):
    """Route dotfetch into the Microsoft Agent Framework (AF 1.x).

    Example::

        from agent_framework import Agent
        from agent_framework.openai import OpenAIChatClient
        from dotfetch.agent_framework import AgentFrameworkAdapter

        af = AgentFrameworkAdapter(fetcher)

        # Per-run tool injection:
        provider = af.context_provider(top_k=5)
        agent = Agent(OpenAIChatClient(), "...", context_providers=[provider])

        # Per-turn (multi-step) tool injection:
        provider = af.context_provider(top_k=3, query_strategy="per_turn")
        provider.attach_to(agent)

    :meth:`select` and :meth:`convert` share the native-adapter selection API,
    including required tools and namespace filters. The bridge, provider and
    middleware factories remain available for AF-specific configuration.
    """

    #: AF's ``DotFetchContextProvider`` supports genuine per-round re-selection
    #: (``query_strategy="per_turn"``), matching the other per-turn adapters.
    live_tier = "per-turn"

    def __init__(self, fetcher: DotFetch, *, default_top_k: int = DEFAULT_TOOL_LIMIT) -> None:
        super().__init__(fetcher, default_limit=default_top_k)
        self._default_top_k = default_top_k

    @staticmethod
    def convert(spec: ToolSpec) -> Any:
        """Convert a selected spec into a native AF FunctionTool.

        Requires agent-framework-core. The lower-level tool bridge retains its
        optional bare-callable mode for applications that need it.
        """
        fields = {
            "name": spec.name,
            "namespace": spec._namespace,
            "description": spec.description,
            "parameters_schema": spec.parameters,
            "requires_confirmation": spec.requires_confirmation,
        }
        definition = spec.tool.model_copy(update=fields) if spec.tool is not None else ToolDefinition(**fields)
        return _build_callable_for_tool(
            definition, spec._fetcher, as_function_tool=True, native_name=spec.native_name,
        )

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
        ``DotFetchToolBridge.get_tools``). Unlike calling
        :meth:`context_provider` directly, this defaults ``query_strategy``
        to ``"per_turn"`` (not :meth:`context_provider`'s own
        back-compatible ``"per_run"`` default) — ``live()`` is meant to
        surface the *deepest* tier AF supports, and ``per_turn`` re-selects on
        every model step rather than once per run. Pass ``query_strategy="per_run"``
        in ``framework_kwargs`` to opt back out.

        Returns a ``DotFetchContextProvider``. Plug it into
        ``Agent(context_providers=[<result>])`` and, for ``per_turn``, also
        attach ``<result>.as_chat_middleware()`` — or call
        ``<result>.attach_to(agent)`` to do both in one step.
        """
        framework_kwargs.setdefault("query_strategy", "per_turn")
        if namespaces is not None:
            framework_kwargs.setdefault("namespaces", namespaces)
        return self.context_provider(
            top_k=limit, score_threshold=score_threshold, **framework_kwargs
        )

    def context_provider(self, *, top_k: int | None = None, **kwargs: Any) -> Any:
        """Build a ``DotFetchContextProvider`` (per-run / per-turn tool injection).

        ``top_k`` defaults to the adapter's ``default_top_k``. Every other keyword
        is forwarded to the provider (``query_strategy``, ``score_threshold``,
        ``required``, ``always_include``, ``static_tools``, ``verbose``,
        ``query_generator``, ...).
        """
        return DotFetchContextProvider(
            self._fetcher,
            top_k=self._default_top_k if top_k is None else top_k,
            **kwargs,
        )

    def tool_bridge(self, **kwargs: Any) -> DotFetchToolBridge:
        """Build a ``DotFetchToolBridge`` (query-time tool retrieval / agent building)."""
        return DotFetchToolBridge(self._fetcher, **kwargs)

    def approval_middleware(self, policy: SecurityPolicy) -> Any:
        """Build AF function middleware that enforces dotfetch's ``SecurityPolicy``."""
        return DotFetchApprovalMiddleware(policy)

    def observability_middleware(self) -> Any:
        """Build AF function middleware that records dotfetch telemetry spans."""
        return DotFetchObservabilityMiddleware(self._fetcher)

    def tool_choice_middleware(self, decider: Any) -> Any:
        """Build AF chat middleware that modulates ``tool_choice`` per round."""
        return DotFetchToolChoiceMiddleware(decider)


__all__ = ["AgentFrameworkAdapter"]
