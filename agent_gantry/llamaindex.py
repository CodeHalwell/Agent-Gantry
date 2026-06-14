"""Agent-Gantry × LlamaIndex.

Clean per-framework imports::

    from agent_gantry.llamaindex import for_llamaindex, gantry_tool_retriever

Re-exports LlamaIndex's static adapter (select + convert) and the deep per-turn "live" provider. Importing
this module does not require LlamaIndex until you actually call into it.
"""

from __future__ import annotations

from agent_gantry._framework_ns import make_lazy_getattr
from agent_gantry.integrations.frameworks.llamaindex import (
    for_llamaindex,
    spec_to_llamaindex,
)

__getattr__ = make_lazy_getattr({'gantry_tool_retriever': 'llamaindex_live', 'gantry_function_agent': 'llamaindex_live'})

__all__ = [
    "for_llamaindex",
    "spec_to_llamaindex",
    "gantry_tool_retriever",  # noqa: F822 (lazy via __getattr__)
    "gantry_function_agent",  # noqa: F822 (lazy via __getattr__)
]
