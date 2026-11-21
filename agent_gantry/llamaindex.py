"""Agent-Gantry × LlamaIndex.

Clean per-framework import::

    from agent_gantry.llamaindex import LlamaIndexAdapter

Importing this module never requires LlamaIndex to be installed; the framework is
imported lazily when you call an adapter method.
"""

from __future__ import annotations

from agent_gantry.integrations.frameworks.llamaindex import (
    LlamaIndexAdapter,
    LlamaIndexToolError,
    current_workflow_context,
)
from agent_gantry.integrations.frameworks.native import ErrorFormatter, ErrorMode

__all__ = [
    "ErrorFormatter",
    "ErrorMode",
    "LlamaIndexAdapter",
    "LlamaIndexToolError",
    "current_workflow_context",
]
