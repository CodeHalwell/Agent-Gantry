"""Agent-Gantry × the OpenAI Agents SDK.

Clean per-framework import::

    from agent_gantry.openai_agents import OpenAIAgentsAdapter, current_run_context

    adapter = OpenAIAgentsAdapter(gantry)
    agent = Agent(name="assistant", tools=await adapter.dynamic_tools(limit=5))

Importing this module never requires the OpenAI Agents SDK to be installed; it is
imported lazily when you build a tool.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from agent_gantry.integrations.frameworks.native import ErrorFormatter, ErrorMode
from agent_gantry.integrations.frameworks.openai_agents import (
    OpenAIAgentsAdapter,
    current_run_context,
)

if TYPE_CHECKING:  # resolved lazily by ``__getattr__`` below
    from agent_gantry.integrations.frameworks.openai_agents_live import GantryDynamicTools

__all__ = [
    "ErrorFormatter",
    "ErrorMode",
    "GantryDynamicTools",
    "OpenAIAgentsAdapter",
    "current_run_context",
]


def __getattr__(name: str) -> Any:
    if name == "GantryDynamicTools":
        from agent_gantry.integrations.frameworks.openai_agents_live import GantryDynamicTools

        return GantryDynamicTools
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
