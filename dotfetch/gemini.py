"""dotfetch × Google Gemini (google-genai).

Clean per-provider import::

    from dotfetch.gemini import GeminiAdapter

``GeminiAdapter(fetcher).tools(query)`` returns Gemini function-declaration
schemas. Wrap them in ``google.genai.types.FunctionDeclaration`` / ``Tool``
yourself before handing them to the SDK.
"""

from __future__ import annotations

from dotfetch.integrations.llm_adapters import GeminiAdapter

__all__ = ["GeminiAdapter"]
