"""Agent-Gantry × LangChain.

Clean per-framework import::

    from agent_gantry.langchain import LangChainAdapter

Importing this module never requires LangChain to be installed; the framework is
imported lazily when you call an adapter method.
"""

from __future__ import annotations

from agent_gantry.integrations.frameworks.langchain import LangChainAdapter

__all__ = ["LangChainAdapter"]
