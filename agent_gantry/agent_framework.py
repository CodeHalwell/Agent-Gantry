"""Agent-Gantry × Microsoft Agent Framework.

Clean per-framework imports::

    from agent_gantry.agent_framework import GantryContextProvider

Re-exports the first-class Microsoft Agent Framework integration — the
``GantryContextProvider`` (per-run / per-call dynamic tool injection), the
``GantryToolBridge``, and the approval / observability / tool-choice middleware.
Importing this module does not require ``agent-framework`` until you call into
it (the framework is imported lazily by the underlying bridge/provider).
"""

from __future__ import annotations

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
    "GantryApprovalMiddleware",
    "GantryContextProvider",
    "GantryObservabilityMiddleware",
    "GantryToolBridge",
    "GantryToolChoiceMiddleware",
    "MissingRequiredToolError",
    "RetrievalCandidate",
    "RetrievalDecision",
]
