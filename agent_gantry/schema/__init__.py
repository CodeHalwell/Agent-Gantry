"""
Schema modules for Agent-Gantry.

Contains data models for tools, queries, configuration, and the MCP/A2A protocols.
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
from agent_gantry.schema.mcp import MCPServerCost, MCPServerDefinition, MCPServerHealth
from agent_gantry.schema.query import (
    ConversationContext,
    RetrievalResult,
    ScoredTool,
    ToolQuery,
)
from agent_gantry.schema.skill import Skill, SkillCategory, SkillSearchResult
from agent_gantry.schema.tool import (
    SchemaDialect,
    ToolCapability,
    ToolCost,
    ToolDefinition,
    ToolHealth,
    ToolSource,
)

__all__ = [
    # Tool models
    "SchemaDialect",
    "ToolCapability",
    "ToolCost",
    "ToolDefinition",
    "ToolHealth",
    "ToolSource",
    # Skill models
    "Skill",
    "SkillCategory",
    "SkillSearchResult",
    # Query models
    "ConversationContext",
    "RetrievalResult",
    "ScoredTool",
    "ToolQuery",
    # Config
    "AgentGantryConfig",
    "A2AAgentConfig",
    "A2AConfig",
    # MCP models
    "MCPServerCost",
    "MCPServerDefinition",
    "MCPServerHealth",
    # A2A models
    "AgentCard",
    "AgentSkill",
    "TaskMessage",
    "TaskMessagePart",
    "TaskRequest",
    "TaskResponse",
]
