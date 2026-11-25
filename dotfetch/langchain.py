"""dotfetch × LangChain.

Clean per-framework import::

    from dotfetch.langchain import LangChainAdapter, current_tool_runtime

Importing this module never requires LangChain to be installed; the framework is
imported lazily when you call an adapter method.
"""

from __future__ import annotations

from dotfetch.integrations.frameworks.langchain import (
    DotFetchToolRuntime,
    LangChainAdapter,
    current_tool_runtime,
)
from dotfetch.integrations.frameworks.native import ErrorFormatter, ErrorMode

__all__ = [
    "ErrorFormatter",
    "ErrorMode",
    "DotFetchToolRuntime",
    "LangChainAdapter",
    "current_tool_runtime",
]
