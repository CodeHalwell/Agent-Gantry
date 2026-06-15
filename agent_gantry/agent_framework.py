"""Agent-Gantry × Microsoft Agent Framework.

Clean per-framework import::

    from agent_gantry.agent_framework import AgentFrameworkAdapter

``AgentFrameworkAdapter`` is the one-class entry point — its methods build the
``GantryContextProvider`` (per-run / per-call dynamic tool injection), the
``GantryToolBridge``, and the approval / observability / tool-choice middleware.
Those underlying types are also re-exported here for type annotations and for
holding the objects the adapter returns.

Importing this module never requires ``agent-framework``; the framework is
imported lazily by the underlying bridge / provider / middleware when used.
"""

from __future__ import annotations

from agent_gantry.integrations.agent_framework_adapter import AgentFrameworkAdapter
from agent_gantry.integrations.agent_framework_bridge import (
    GantryToolBridge,
    RetrievalCandidate,
    RetrievalDecision,
)
from agent_gantry.integrations.agent_framework_middleware import (
    GantryApprovalMiddleware,
    GantryObservabilityMiddleware,
    GantryToolChoiceMiddleware,
)
from agent_gantry.integrations.agent_framework_provider import (
    GantryContextProvider,
    MissingRequiredToolError,
)

__all__ = [
    "AgentFrameworkAdapter",
    "GantryApprovalMiddleware",
    "GantryContextProvider",
    "GantryObservabilityMiddleware",
    "GantryToolBridge",
    "GantryToolChoiceMiddleware",
    "MissingRequiredToolError",
    "RetrievalCandidate",
    "RetrievalDecision",
]
