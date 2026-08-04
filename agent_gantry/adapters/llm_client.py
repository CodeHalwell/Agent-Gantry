"""
LLM client for intent classification and other LLM-based features.

Supports multiple providers: OpenAI, Anthropic, Google, Mistral, Groq.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agent_gantry.schema.config import LLMConfig

logger = logging.getLogger(__name__)


class LLMClient:
    """
    Unified LLM client for intent classification.

    Supports multiple providers with a consistent interface.
    Uses async clients where available to avoid blocking the event loop.
    """

    def __init__(self, config: LLMConfig) -> None:
        """
        Initialize the LLM client.

        Args:
            config: LLM configuration
        """
        self._config = config
        self._client: Any = None
        self._provider = config.provider
        self._model = config.model
        self._initialize_client()

    def _initialize_client(self) -> None:
        """Initialize the provider-specific async client."""
        api_key = self._config.api_key or self._get_api_key_from_env()

        if self._provider == "openai":
            from openai import AsyncOpenAI

            base_url = self._config.base_url
            self._client = AsyncOpenAI(
                api_key=api_key,
                base_url=base_url,
            )
        elif self._provider == "anthropic":
            from anthropic import AsyncAnthropic

            self._client = AsyncAnthropic(api_key=api_key)
        elif self._provider == "google":
            from google import genai

            self._client = genai.Client(api_key=api_key)
        elif self._provider == "mistral":
            # mistralai (the official Mistral SDK) is quarantined on PyPI (2026-05-12).
            # Mistral's API is OpenAI-compatible; use AsyncOpenAI with the Mistral
            # base URL. The chat.completions interface is identical.
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI(
                api_key=api_key,
                base_url="https://api.mistral.ai/v1",
            )
        elif self._provider == "groq":
            from groq import AsyncGroq

            self._client = AsyncGroq(api_key=api_key)
        else:
            raise ValueError(f"Unsupported LLM provider: {self._provider}")

    def _get_api_key_from_env(self) -> str:
        """Get API key from environment based on provider."""
        env_vars = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "google": "GOOGLE_API_KEY",
            "mistral": "MISTRAL_API_KEY",
            "groq": "GROQ_API_KEY",
        }
        env_var = env_vars.get(self._provider)
        if not env_var:
            raise ValueError(f"Unknown provider: {self._provider}")

        api_key = os.getenv(env_var)
        if not api_key:
            raise ValueError(
                f"API key not found. Set {env_var} environment variable "
                f"or provide api_key in LLMConfig."
            )
        return api_key

    async def classify_intent(
        self,
        query: str,
        conversation_summary: str | None = None,
        available_intents: list[str] | None = None,
    ) -> str:
        """
        Classify the intent of a query using the LLM.

        Uses async clients to avoid blocking the event loop.

        Args:
            query: The user's query
            conversation_summary: Optional conversation context
            available_intents: List of available intent values

        Returns:
            The classified intent as a string
        """
        # Build the classification prompt
        context = ""
        if conversation_summary:
            context = f"\n\nConversation context:\n{conversation_summary}"

        intents_list = available_intents or [
            "data_query",
            "data_mutation",
            "analysis",
            "communication",
            "file_operations",
            "customer_support",
            "admin",
            "unknown",
        ]

        prompt = f"""Classify the intent of the following user query into one of these categories:

{chr(10).join(f"- {intent}" for intent in intents_list)}

User query: {query}{context}

Respond with ONLY the intent category name (e.g., "data_query"), nothing else."""

        # Call the appropriate provider using async methods
        if self._provider in ("openai", "groq"):
            # AsyncOpenAI and AsyncGroq have native async support
            params: dict[str, Any] = {
                "model": self._model,
                "messages": [{"role": "user", "content": prompt}],
            }
            if self._provider == "openai" and self._model.startswith(
                ("gpt-5", "o1", "o3", "o4")
            ):
                # OpenAI reasoning models (gpt-5 family, o-series) reject the
                # legacy max_tokens parameter (they require
                # max_completion_tokens) and support only the default
                # temperature — sending either fails the request, which would
                # silently degrade every classification to the UNKNOWN
                # fallback. Reasoning tokens also count toward the completion
                # budget, so give headroom above the configured cap (sized
                # for one-word answers from non-reasoning models).
                params["max_completion_tokens"] = max(self._config.max_tokens, 512)
            else:
                params["max_tokens"] = self._config.max_tokens
                params["temperature"] = self._config.temperature
            response = await self._client.chat.completions.create(**params)
            result = response.choices[0].message.content.strip()
        elif self._provider == "anthropic":
            # AsyncAnthropic has native async support
            response = await self._client.messages.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self._config.max_tokens,
                temperature=self._config.temperature,
            )
            result = response.content[0].text.strip()
        elif self._provider == "google":
            # google-genai >= 1.0 exposes a native async interface via client.aio
            response = await self._client.aio.models.generate_content(
                model=self._model,
                contents=prompt,
            )
            result = response.text.strip()
        elif self._provider == "mistral":
            # Mistral's API is OpenAI-compatible; self._client is AsyncOpenAI
            # with base_url="https://api.mistral.ai/v1" (set in _initialize_client).
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self._config.max_tokens,
                temperature=self._config.temperature,
            )
            result = response.choices[0].message.content.strip()
        else:
            raise ValueError(f"Unsupported provider: {self._provider}")

        # Clean up the result (remove quotes, lowercase)
        result = result.strip("\"'").lower().strip()

        # Validate result is in available intents
        if result not in intents_list:
            # Try to find an exact substring match
            for intent in intents_list:
                if intent == result:
                    return intent
            # Default to unknown if no match
            return "unknown"

        return str(result)

    async def health_check(self) -> bool:
        """
        Check if the LLM client is healthy.

        Returns:
            True if the client is initialized and ready
        """
        return self._client is not None
