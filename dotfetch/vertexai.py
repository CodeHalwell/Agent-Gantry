"""dotfetch × Google Vertex AI.

Clean per-provider import::

    from dotfetch.vertexai import VertexAIAdapter

``VertexAIAdapter(fetcher).tools(query)`` returns Gemini function-declaration
schemas. Wrap them in ``vertexai.generative_models.FunctionDeclaration`` /
``Tool`` yourself before handing them to the SDK.
"""

from __future__ import annotations

from dotfetch.integrations.llm_adapters import VertexAIAdapter

__all__ = ["VertexAIAdapter"]
