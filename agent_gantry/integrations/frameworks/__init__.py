"""Native per-framework tool adapters built on a shared selection core.

Each module exports ``for_<framework>(gantry, query, ...)`` (and a matching
``spec_to_<framework>`` converter) that turns Gantry-selected tools into that
framework's native tool objects. Imports of the third-party framework are lazy,
so importing this package never requires those frameworks to be installed.
"""

from __future__ import annotations

from agent_gantry.integrations.frameworks.base import (
    GantryToolset,
    ToolExecutionError,
    ToolSpec,
    spec_from_tool,
)

__all__ = [
    "GantryToolset",
    "ToolExecutionError",
    "ToolSpec",
    "spec_from_tool",
]
