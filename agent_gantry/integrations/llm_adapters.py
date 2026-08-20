"""LLM provider SDK adapters for Agent-Gantry.

One class per LLM SDK. Each wraps :meth:`AgentGantry.retrieve_tools` with the
provider's schema *dialect* baked in, so a single call returns tool schemas in
that provider's shape::

    from agent_gantry.openai import OpenAIAdapter
    from agent_gantry.anthropic import AnthropicAdapter

    tools = await OpenAIAdapter(gantry).tools("what's the weather?", limit=3)
    claude_tools = await AnthropicAdapter(gantry).tools("what's the weather?", limit=3)

You still hand ``tools`` to the vendor SDK yourself — these adapters never import
the vendor SDK, they only produce schemas. Tool *execution* still runs through
``gantry.execute``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from agent_gantry.integrations.frameworks.base import DEFAULT_TOOL_LIMIT

if TYPE_CHECKING:
    from agent_gantry.core.gantry import AgentGantry


class _LLMToolAdapter:
    """Base adapter: select tools and emit them in one provider's dialect.

    Subclasses set :attr:`dialect`. ``tools`` defaults ``limit`` to the adapter's
    ``default_limit`` and forwards any extra keyword to ``retrieve_tools``.
    """

    dialect: str = "openai"

    def __init__(self, gantry: AgentGantry, *, default_limit: int = DEFAULT_TOOL_LIMIT) -> None:
        self._gantry = gantry
        self._default_limit = default_limit

    async def tools(
        self,
        query: str,
        *,
        limit: int | None = None,
        score_threshold: float = 0.0,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Select tools for ``query`` and return them as ``self.dialect`` schemas.

        Keywords are split by :meth:`AgentGantry.retrieve_tools`: those naming a
        ``ToolQuery`` field configure retrieval, the rest are passed to the
        dialect adapter — so ``tools(q, strict=True)`` reaches OpenAI's strict
        mode, and ``namespaces=[...]`` still filters the query.
        """
        return await self._gantry.retrieve_tools(
            query,
            limit=self._default_limit if limit is None else limit,
            dialect=self.dialect,
            score_threshold=score_threshold,
            **kwargs,
        )


class OpenAIAdapter(_LLMToolAdapter):
    """OpenAI Chat Completions (also Azure OpenAI / OpenRouter — OpenAI-compatible)."""

    dialect = "openai"

    async def responses_tools(
        self,
        query: str,
        *,
        limit: int | None = None,
        score_threshold: float = 0.0,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Select tools in the OpenAI **Responses API** shape (flattened schema)."""
        return await self._gantry.retrieve_tools(
            query,
            limit=self._default_limit if limit is None else limit,
            dialect="openai_responses",
            score_threshold=score_threshold,
            **kwargs,
        )


class AnthropicAdapter(_LLMToolAdapter):
    """Anthropic (Claude) Messages API — ``{"name", "description", "input_schema"}``."""

    dialect = "anthropic"


class GeminiAdapter(_LLMToolAdapter):
    """Google Gemini (``google-genai``) function declarations."""

    dialect = "gemini"


class GroqAdapter(_LLMToolAdapter):
    """Groq (OpenAI-compatible fast inference)."""

    dialect = "groq"


class VertexAIAdapter(_LLMToolAdapter):
    """Google Vertex AI — Gemini function declarations.

    Separate class from :class:`GeminiAdapter` (the two target different vendor
    SDK entry points: ``google-genai`` vs ``vertexai``) but they emit the same
    ``"gemini"`` schema dialect.
    """

    dialect = "gemini"


class MistralAdapter(_LLMToolAdapter):
    """Mistral (OpenAI-compatible endpoint)."""

    dialect = "mistral"


__all__ = [
    "OpenAIAdapter",
    "AnthropicAdapter",
    "GeminiAdapter",
    "GroqAdapter",
    "VertexAIAdapter",
    "MistralAdapter",
]
