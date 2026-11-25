"""
Schema modules for dotfetch.

Contains data models for tools, queries, configuration, and the MCP/A2A protocols.
"""

from dotfetch.schema.a2a import (
    AgentCard,
    AgentSkill,
    TaskMessage,
    TaskMessagePart,
    TaskRequest,
    TaskResponse,
)
from dotfetch.schema.config import A2AAgentConfig, A2AConfig, DotFetchConfig
from dotfetch.schema.mcp import MCPServerCost, MCPServerDefinition, MCPServerHealth
from dotfetch.schema.query import (
    ConversationContext,
    RetrievalResult,
    ScoredTool,
    ToolQuery,
)
from dotfetch.schema.skill import Skill, SkillCategory, SkillSearchResult
from dotfetch.schema.tool import (
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
    "DotFetchConfig",
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
