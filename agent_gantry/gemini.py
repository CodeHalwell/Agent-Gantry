"""Agent-Gantry × Google Gemini (google-genai).

Clean per-provider import::

    from agent_gantry.gemini import GeminiAdapter

``GeminiAdapter(gantry).tools(query)`` returns Gemini function-declaration
schemas. Wrap them in ``google.genai.types.FunctionDeclaration`` / ``Tool``
yourself before handing them to the SDK.
"""

from __future__ import annotations

from agent_gantry.integrations.llm_adapters import GeminiAdapter

__all__ = ["GeminiAdapter"]
