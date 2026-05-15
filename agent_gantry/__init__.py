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

__version__ = "0.3.0"
__all__ = [
    "AgentGantry",
    "GantryContextProvider",
    "MissingRequiredToolError",
    "RetrievalCandidate",
    "RetrievalDecision",
    "create_default_gantry",
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
