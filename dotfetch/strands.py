"""dotfetch × AWS Strands Agents.

Clean per-framework import::

    from dotfetch.strands import StrandsAdapter

Importing this module never requires Strands Agents to be installed; it is
imported lazily when you call an adapter method.
"""

from __future__ import annotations

from dotfetch.integrations.frameworks.native import ErrorFormatter, ErrorMode
from dotfetch.integrations.frameworks.strands import (
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
