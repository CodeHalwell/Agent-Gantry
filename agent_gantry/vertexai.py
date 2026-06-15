"""Agent-Gantry × Google Vertex AI.

Clean per-provider import::

    from agent_gantry.vertexai import VertexAIAdapter

``VertexAIAdapter(gantry).tools(query)`` returns Gemini function-declaration
schemas. Wrap them in ``vertexai.generative_models.FunctionDeclaration`` /
``Tool`` yourself before handing them to the SDK.
"""

from __future__ import annotations

from agent_gantry.integrations.llm_adapters import VertexAIAdapter

__all__ = ["VertexAIAdapter"]
