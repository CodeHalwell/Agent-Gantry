"""dotfetch × Microsoft Agent Framework.

Clean per-framework import::

    from dotfetch.agent_framework import AgentFrameworkAdapter

``AgentFrameworkAdapter`` is the one-class entry point — its methods build the
``DotFetchContextProvider`` (per-run / per-turn dynamic tool injection), the
``DotFetchToolBridge``, and the approval / observability / tool-choice middleware.
Those underlying types are also re-exported here for type annotations and for
holding the objects the adapter returns.

Importing this module never requires ``agent-framework``; the framework is
imported lazily by the underlying bridge / provider / middleware when used.
"""

from __future__ import annotations

from dotfetch.integrations.agent_framework_adapter import AgentFrameworkAdapter
from dotfetch.integrations.agent_framework_bridge import (
    DotFetchToolBridge,
    RetrievalCandidate,
    RetrievalDecision,
)
from dotfetch.integrations.agent_framework_middleware import (
    DotFetchApprovalMiddleware,
    DotFetchObservabilityMiddleware,
    DotFetchToolChoiceMiddleware,
)
from dotfetch.integrations.agent_framework_provider import (
    DotFetchContextProvider,
    MissingRequiredToolError,
)

__all__ = [
    "AgentFrameworkAdapter",
    "DotFetchApprovalMiddleware",
    "DotFetchContextProvider",
    "DotFetchObservabilityMiddleware",
    "DotFetchToolBridge",
    "DotFetchToolChoiceMiddleware",
    "MissingRequiredToolError",
    "RetrievalCandidate",
    "RetrievalDecision",
]
