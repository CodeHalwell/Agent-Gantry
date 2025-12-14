"""
Schema modules for Agent-Gantry.

Contains data models for tools, queries, events, configuration, and A2A protocol.
"""

from agent_gantry.schema.a2a import (
    AgentCard,
    AgentSkill,
    TaskMessage,
    TaskMessagePart,
    TaskRequest,
    TaskResponse,
)
from agent_gantry.schema.config import A2AAgentConfig, A2AConfig, AgentGantryConfig
from agent_gantry.schema.events import (
    ExecutionEvent,
    HealthChangeEvent,
    RetrievalEvent,
)
from agent_gantry.schema.query import (
    ConversationContext,
    RetrievalResult,
    ScoredTool,
    ToolQuery,
)
from agent_gantry.schema.tool import (
    SchemaDialect,
    ToolCapability,
    ToolCost,
    ToolDefinition,
    ToolDependency,
    ToolHealth,
    ToolSource,
)

__all__ = [
    # Tool models
    "SchemaDialect",
    "ToolCapability",
    "ToolCost",
    "ToolDefinition",
    "ToolDependency",
    "ToolHealth",
    "ToolSource",
    # Query models
    "ConversationContext",
    "RetrievalResult",
    "ScoredTool",
    "ToolQuery",
    # Event models
    "ExecutionEvent",
    "HealthChangeEvent",
    "RetrievalEvent",
    # Config
    "AgentGantryConfig",
    "A2AAgentConfig",
    "A2AConfig",
    # A2A models
    "AgentCard",
    "AgentSkill",
    "TaskMessage",
    "TaskMessagePart",
    "TaskRequest",
    "TaskResponse",
]
