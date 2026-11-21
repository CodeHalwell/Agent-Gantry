"""Agent-Gantry × LlamaIndex.

Clean per-framework import::

    from agent_gantry.llamaindex import LlamaIndexAdapter

Importing this module never requires LlamaIndex to be installed; the framework is
imported lazily when you call an adapter method.
"""

from __future__ import annotations

from typing import Any

from agent_gantry.integrations.frameworks.llamaindex import (
    LlamaIndexAdapter,
    LlamaIndexToolError,
    current_workflow_context,
    gantry_function_agent_class,
)
from agent_gantry.integrations.frameworks.native import ErrorFormatter, ErrorMode

__all__ = [
    "ErrorFormatter",
    "ErrorMode",
    "LlamaIndexAdapter",
    "LlamaIndexToolError",
    "current_workflow_context",
]


def __getattr__(name: str) -> Any:
    # ``GantryFunctionAgent`` is built on first access, so importing this module
    # never needs LlamaIndex (and so it is not listed in ``__all__``).
    if name == "GantryFunctionAgent":
        return gantry_function_agent_class()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
