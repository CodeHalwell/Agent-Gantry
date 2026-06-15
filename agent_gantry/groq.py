"""Agent-Gantry × Groq.

Clean per-provider import::

    from agent_gantry.groq import GroqAdapter

``GroqAdapter(gantry).tools(query)`` returns OpenAI-compatible tool schemas for
Groq's fast-inference endpoint. You still hand the schemas to the vendor SDK
yourself.
"""

from __future__ import annotations

from agent_gantry.integrations.llm_adapters import GroqAdapter

__all__ = ["GroqAdapter"]
