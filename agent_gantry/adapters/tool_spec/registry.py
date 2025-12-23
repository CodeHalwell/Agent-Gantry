"""
Dialect registry for tool specification adapters.

Maintains a registry of adapters keyed by dialect/provider names, enabling
dynamic dispatch of tool specifications based on the target LLM provider.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agent_gantry.adapters.tool_spec.base import ToolSpecAdapter

logger = logging.getLogger(__name__)


class DialectRegistry:
    """
    Registry for tool specification adapters.

    Manages adapters keyed by dialect names ('openai', 'anthropic', 'gemini', etc.)
    and provides lookup functionality with auto-detection support.

    Example:
        registry = DialectRegistry()
        registry.register(OpenAIAdapter())
        adapter = registry.get('openai')
        schema = adapter.to_provider_schema(tool)
    """

    _instance: DialectRegistry | None = None
    _adapters: dict[str, ToolSpecAdapter]

    def __init__(self) -> None:
        """Initialize an empty registry."""
        self._adapters = {}

    @classmethod
    def default(cls) -> DialectRegistry:
        """
        Get the default global registry instance.

        Returns:
            The singleton DialectRegistry instance
        """
        if cls._instance is None:
            cls._instance = cls()
            cls._instance._register_default_adapters()
        return cls._instance

    def _register_default_adapters(self) -> None:
        """Register all built-in adapters."""
        # Import here to avoid circular imports
        from agent_gantry.adapters.tool_spec.providers import (
            AnthropicAdapter,
            GeminiAdapter,
            GroqAdapter,
            MistralAdapter,
            OpenAIAdapter,
            OpenAIResponsesAdapter,
        )

        self.register(OpenAIAdapter())
        self.register(OpenAIResponsesAdapter())
        self.register(AnthropicAdapter())
        self.register(GeminiAdapter())
        self.register(MistralAdapter())
        self.register(GroqAdapter())

    def register(self, adapter: ToolSpecAdapter) -> None:
        """
        Register an adapter for its dialect.

        Args:
            adapter: The adapter to register
        """
        dialect = adapter.dialect_name
        if dialect in self._adapters:
            logger.warning(f"Overwriting existing adapter for dialect '{dialect}'")
        self._adapters[dialect] = adapter
        logger.debug(f"Registered adapter for dialect '{dialect}'")

    def get(self, dialect: str) -> ToolSpecAdapter:
        """
        Get an adapter by dialect name.

        Args:
            dialect: The dialect/provider name

        Returns:
            The registered adapter

        Raises:
            KeyError: If no adapter is registered for the dialect
        """
        if dialect == "auto":
            # Default to OpenAI for 'auto' dialect
            return self._adapters.get("openai", self._get_fallback_adapter())
        if dialect not in self._adapters:
            raise KeyError(
                f"No adapter registered for dialect '{dialect}'. "
                f"Available: {list(self._adapters.keys())}"
            )
        return self._adapters[dialect]

    def _get_fallback_adapter(self) -> ToolSpecAdapter:
        """Get a fallback adapter when none is registered."""
        from agent_gantry.adapters.tool_spec.providers import OpenAIAdapter

        return OpenAIAdapter()

    def has(self, dialect: str) -> bool:
        """
        Check if an adapter is registered for a dialect.

        Args:
            dialect: The dialect/provider name

        Returns:
            True if an adapter is registered
        """
        return dialect in self._adapters or dialect == "auto"

    def list_dialects(self) -> list[str]:
        """
        List all registered dialect names.

        Returns:
            List of registered dialect names
        """
        return list(self._adapters.keys())

    def unregister(self, dialect: str) -> bool:
        """
        Unregister an adapter.

        Args:
            dialect: The dialect to unregister

        Returns:
            True if an adapter was removed
        """
        if dialect in self._adapters:
            del self._adapters[dialect]
            return True
        return False

    def clear(self) -> None:
        """Clear all registered adapters."""
        self._adapters.clear()


def get_adapter(dialect: str = "auto") -> ToolSpecAdapter:
    """
    Convenience function to get an adapter from the default registry.

    Args:
        dialect: The dialect/provider name (default: 'auto')

    Returns:
        The registered adapter
    """
    return DialectRegistry.default().get(dialect)


def to_dialect(
    tool: Any,  # ToolDefinition
    dialect: str = "auto",
    **options: Any,
) -> dict[str, Any]:
    """
    Convenience function to convert a tool to provider format.

    Args:
        tool: The ToolDefinition to convert
        dialect: Target dialect/provider name
        **options: Provider-specific options

    Returns:
        Provider-specific tool schema
    """
    adapter = get_adapter(dialect)
    return adapter.to_provider_schema(tool, **options)
