"""Agent-Gantry × LangGraph.

Clean per-framework imports::

    from agent_gantry.langgraph import for_langgraph, create_gantry_react_agent

Re-exports LangGraph's static adapter (select + convert) and the deep per-turn "live" provider. Importing
this module does not require LangGraph until you actually call into it.
"""

from __future__ import annotations

from agent_gantry._framework_ns import make_lazy_getattr
from agent_gantry.integrations.frameworks.langgraph import (
    for_langgraph,
    spec_to_langgraph,
)

__getattr__ = make_lazy_getattr({'create_gantry_react_agent': 'langgraph_live', 'acreate_gantry_react_agent': 'langgraph_live', 'select_tools_for_state': 'langgraph_live'})

__all__ = [
    "for_langgraph",
    "spec_to_langgraph",
    "create_gantry_react_agent",  # noqa: F822 (lazy via __getattr__)
    "acreate_gantry_react_agent",  # noqa: F822 (lazy via __getattr__)
    "select_tools_for_state",  # noqa: F822 (lazy via __getattr__)
]
