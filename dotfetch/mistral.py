"""dotfetch × Mistral.

Clean per-provider import::

    from dotfetch.mistral import MistralAdapter

``MistralAdapter(fetcher).tools(query)`` returns OpenAI-compatible tool schemas
for Mistral's endpoint (the ``mistralai`` package is quarantined on PyPI; use the
OpenAI SDK with ``base_url="https://api.mistral.ai/v1"``). You still hand the
schemas to the vendor SDK yourself.
"""

from __future__ import annotations

from dotfetch.integrations.llm_adapters import MistralAdapter

__all__ = ["MistralAdapter"]
