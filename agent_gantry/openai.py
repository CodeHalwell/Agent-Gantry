"""Agent-Gantry × OpenAI (Chat Completions / Responses).

Clean per-provider import::

    from agent_gantry.openai import OpenAIAdapter

``OpenAIAdapter(gantry).tools(query)`` returns OpenAI Chat Completions tool
schemas; ``.responses_tools(query)`` returns the OpenAI Responses API shape.
Also works for Azure OpenAI and OpenRouter (OpenAI-compatible). You still hand
the schemas to the vendor SDK yourself.
"""

from __future__ import annotations

from agent_gantry.integrations.llm_adapters import OpenAIAdapter

__all__ = ["OpenAIAdapter"]
