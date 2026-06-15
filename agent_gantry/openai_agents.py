"""Agent-Gantry × the OpenAI Agents SDK.

Clean per-framework import::

    from agent_gantry.openai_agents import OpenAIAgentsAdapter

Importing this module never requires the OpenAI Agents SDK to be installed; it is
imported lazily when you call an adapter method.
"""

from __future__ import annotations

from agent_gantry.integrations.frameworks.openai_agents import OpenAIAgentsAdapter

__all__ = ["OpenAIAgentsAdapter"]
