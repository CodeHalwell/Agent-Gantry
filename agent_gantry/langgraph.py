"""Agent-Gantry × LangGraph.

Clean per-framework import::

    from agent_gantry.langgraph import LangGraphAdapter

Importing this module never requires LangGraph to be installed; the framework is
imported lazily when you call an adapter method.
"""

from __future__ import annotations

from agent_gantry.integrations.frameworks.langgraph import LangGraphAdapter

__all__ = ["LangGraphAdapter"]
