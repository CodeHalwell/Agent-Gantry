"""
Agent-Gantry: Universal Tool Orchestration Platform

Intelligent, secure tool orchestration for LLM-based agent systems.

Core Philosophy: Context is precious. Execution is sacred. Trust is earned.
"""

import logging as _logging
from typing import Any

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
from agent_gantry.observability.console import enable_console_logging
from agent_gantry.schema.execution import ToolCall, ToolCallEvent, ToolResult
from agent_gantry.schema.query import ConversationContext, ToolQuery
from agent_gantry.schema.skill import Skill, SkillCategory, SkillSearchResult
from agent_gantry.schema.tool import (
    ToolCapability,
    ToolCost,
    ToolDefinition,
    ToolHealth,
    ToolSource,
)
from agent_gantry.utils.render import render_result

# Library logging hygiene: attach a NullHandler to the package root logger so
# importing agent_gantry never configures handlers or levels on the
# application's behalf. Output is opt-in via the consumer's own logging config
# or agent_gantry.enable_console_logging(). Guarded so a module reload (REPLs,
# hot-reload dev servers, some test setups) doesn't stack duplicate handlers.
_ag_root_logger = _logging.getLogger("agent_gantry")
if not any(isinstance(_h, _logging.NullHandler) for _h in _ag_root_logger.handlers):
    _ag_root_logger.addHandler(_logging.NullHandler())


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


def extract_tool_calls(response: "Any", dialect: str = "openai") -> "list[Any]":
    """Pull every tool call out of a provider response (parallel calls included).

    See :func:`agent_gantry.adapters.tool_spec.round_trip.extract_tool_calls`.
    """
    from agent_gantry.adapters.tool_spec.round_trip import extract_tool_calls as _impl

    return _impl(response, dialect)


def __getattr__(name: str) -> "Any":
    """Lazily surface the streaming accumulator at package level."""
    if name == "StreamingToolCallAccumulator":
        from agent_gantry.adapters.tool_spec import round_trip

        return round_trip.StreamingToolCallAccumulator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__version__ = "0.11.0"
__all__ = [
    "AgentGantry",
    "StreamingToolCallAccumulator",
    "GantryContextProvider",
    "MissingRequiredToolError",
    "RetrievalCandidate",
    "RetrievalDecision",
    "create_default_gantry",
    "disable_af_instrumentation",
    "enable_console_logging",
    "extract_tool_calls",
    "render_result",
    "with_semantic_tools",
    "set_default_gantry",
    "ToolCall",
    "ToolCallEvent",
    "ToolResult",
    "ToolQuery",
    "ConversationContext",
    "Skill",
    "SkillCategory",
    "SkillSearchResult",
    "ToolCapability",
    "ToolCost",
    "ToolDefinition",
    "ToolHealth",
    "ToolSource",
]
