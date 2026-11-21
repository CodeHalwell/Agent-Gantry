"""Agent-Gantry × AWS Strands Agents.

Clean per-framework import::

    from agent_gantry.strands import StrandsAdapter

Importing this module never requires Strands Agents to be installed; it is
imported lazily when you call an adapter method.
"""

from __future__ import annotations

from agent_gantry.integrations.frameworks.native import ErrorFormatter, ErrorMode
from agent_gantry.integrations.frameworks.strands import (
    APPROVAL_INTERRUPT,
    StrandsAdapter,
    current_tool_context,
)

__all__ = [
    "APPROVAL_INTERRUPT",
    "ErrorFormatter",
    "ErrorMode",
    "StrandsAdapter",
    "current_tool_context",
]
