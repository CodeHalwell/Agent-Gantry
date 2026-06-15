"""Agent-Gantry × Anthropic (Claude).

Clean per-provider import::

    from agent_gantry.anthropic import AnthropicAdapter

``AnthropicAdapter(gantry).tools(query)`` returns Anthropic Messages API tool
schemas (``{"name", "description", "input_schema"}``). You still hand the schemas
to the vendor SDK yourself.
"""

from __future__ import annotations

from agent_gantry.integrations.llm_adapters import AnthropicAdapter

__all__ = ["AnthropicAdapter"]
