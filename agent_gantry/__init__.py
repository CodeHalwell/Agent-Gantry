"""
Agent-Gantry: Universal Tool Orchestration Platform

Intelligent, secure tool orchestration for LLM-based agent systems.

Core Philosophy: Context is precious. Execution is sacred. Trust is earned.
"""

from agent_gantry.core.gantry import AgentGantry, create_default_gantry
from agent_gantry.integrations.agent_framework_bridge import (
    RetrievalCandidate,
    RetrievalDecision,
)
from agent_gantry.integrations.agent_framework_provider import (
    GantryContextProvider,
    MissingRequiredToolError,
)
from agent_gantry.integrations.semantic_tools import (
    set_default_gantry,
    with_semantic_tools,
)
from agent_gantry.schema.execution import ToolCall, ToolResult
from agent_gantry.schema.query import ConversationContext, ToolQuery
from agent_gantry.schema.tool import (
    ToolCapability,
    ToolCost,
    ToolDefinition,
    ToolHealth,
    ToolSource,
)


def disable_af_instrumentation() -> bool:
    """Disable Agent Framework ≥1.6.0 default ContextVar instrumentation.

    Call once at process startup before constructing any agents when running
    concurrent workflows (``asyncio.gather()`` / ``TaskGroup``) on AF 1.6.0.
    Sequential workflows (WorkflowAgent, SequentialBuilder, HandoffBuilder)
    are **not** affected and do not require this call.

    The import is deferred so that the ``agent-framework`` package is only
    required when the function is actually called, keeping ``agent_gantry``
    importable in environments where AF is not installed.

    Returns ``True`` if instrumentation was disabled, ``False`` otherwise
    (AF not installed, AF version is older than 1.6.0, or the telemetry
    module does not expose ``disable_instrumentation``).
    """
    from agent_gantry.integrations.agent_framework_bridge import (
        disable_af_instrumentation as _impl,
    )
    return _impl()


__version__ = "0.5.0"
__all__ = [
    "AgentGantry",
    "GantryContextProvider",
    "MissingRequiredToolError",
    "RetrievalCandidate",
    "RetrievalDecision",
    "create_default_gantry",
    "disable_af_instrumentation",
    "with_semantic_tools",
    "set_default_gantry",
    "ToolCall",
    "ToolResult",
    "ToolQuery",
    "ConversationContext",
    "ToolCapability",
    "ToolCost",
    "ToolDefinition",
    "ToolHealth",
    "ToolSource",
]
