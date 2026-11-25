"""dotfetch × Groq.

Clean per-provider import::

    from dotfetch.groq import GroqAdapter

``GroqAdapter(fetcher).tools(query)`` returns OpenAI-compatible tool schemas for
Groq's fast-inference endpoint. You still hand the schemas to the vendor SDK
yourself.
"""

from __future__ import annotations

from dotfetch.integrations.llm_adapters import GroqAdapter

__all__ = ["GroqAdapter"]
