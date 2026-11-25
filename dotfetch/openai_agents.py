"""dotfetch × the OpenAI Agents SDK.

Clean per-framework import::

    from dotfetch.openai_agents import OpenAIAgentsAdapter, current_run_context

    adapter = OpenAIAgentsAdapter(fetcher)
    agent = Agent(name="assistant", tools=await adapter.dynamic_tools(limit=5))

Importing this module never requires the OpenAI Agents SDK to be installed; it is
imported lazily when you build a tool.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from dotfetch.integrations.frameworks.native import ErrorFormatter, ErrorMode
from dotfetch.integrations.frameworks.openai_agents import (
    OpenAIAgentsAdapter,
    current_run_context,
)

if TYPE_CHECKING:  # resolved lazily by ``__getattr__`` below
    from dotfetch.integrations.frameworks.openai_agents_live import DotFetchDynamicTools

__all__ = [
    "ErrorFormatter",
    "ErrorMode",
    "DotFetchDynamicTools",
    "OpenAIAgentsAdapter",
    "current_run_context",
]


def __getattr__(name: str) -> Any:
    if name == "DotFetchDynamicTools":
        from dotfetch.integrations.frameworks.openai_agents_live import DotFetchDynamicTools

        return DotFetchDynamicTools
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
