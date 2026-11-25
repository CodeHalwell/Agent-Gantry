"""dotfetch × OpenAI (Chat Completions / Responses).

Clean per-provider import::

    from dotfetch.openai import OpenAIAdapter

``OpenAIAdapter(fetcher).tools(query)`` returns OpenAI Chat Completions tool
schemas; ``.responses_tools(query)`` returns the OpenAI Responses API shape.
Also works for Azure OpenAI and OpenRouter (OpenAI-compatible). You still hand
the schemas to the vendor SDK yourself.
"""

from __future__ import annotations

from dotfetch.integrations.llm_adapters import OpenAIAdapter

__all__ = ["OpenAIAdapter"]
