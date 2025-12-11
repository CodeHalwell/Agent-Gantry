"""
Schema modules for Agent-Gantry.

Contains data models for tools, queries, events, and configuration.
"""

from agent_gantry.schema.config import AgentGantryConfig
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
]
