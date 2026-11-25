"""dotfetch × LlamaIndex.

Clean per-framework import::

    from dotfetch.llamaindex import LlamaIndexAdapter

Importing this module never requires LlamaIndex to be installed; the framework is
imported lazily when you call an adapter method.
"""

from __future__ import annotations

from typing import Any

from dotfetch.integrations.frameworks.llamaindex import (
    LlamaIndexAdapter,
    LlamaIndexToolError,
    current_workflow_context,
    fetcher_function_agent_class,
)
from dotfetch.integrations.frameworks.native import ErrorFormatter, ErrorMode

__all__ = [
    "ErrorFormatter",
    "ErrorMode",
    "LlamaIndexAdapter",
    "LlamaIndexToolError",
    "current_workflow_context",
]


def __getattr__(name: str) -> Any:
    # ``DotFetchFunctionAgent`` is built on first access, so importing this module
    # never needs LlamaIndex (and so it is not listed in ``__all__``).
    if name == "DotFetchFunctionAgent":
        return fetcher_function_agent_class()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
