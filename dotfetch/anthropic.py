"""dotfetch × Anthropic (Claude).

Clean per-provider import::

    from dotfetch.anthropic import AnthropicAdapter

``AnthropicAdapter(fetcher).tools(query)`` returns Anthropic Messages API tool
schemas (``{"name", "description", "input_schema"}``). You still hand the schemas
to the vendor SDK yourself.
"""

from __future__ import annotations

from dotfetch.integrations.llm_adapters import AnthropicAdapter

__all__ = ["AnthropicAdapter"]
