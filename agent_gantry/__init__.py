"""
Agent-Gantry: Universal Tool Orchestration Platform

Intelligent, secure tool orchestration for LLM-based agent systems.

Core Philosophy: Context is precious. Execution is sacred. Trust is earned.
"""

from agent_gantry.core.gantry import AgentGantry
from agent_gantry.schema.tool import (
    ToolCapability,
    ToolCost,
    ToolDefinition,
    ToolHealth,
    ToolSource,
)

__version__ = "0.1.0"
__all__ = [
    "AgentGantry",
    "ToolCapability",
    "ToolCost",
    "ToolDefinition",
    "ToolHealth",
    "ToolSource",
]
