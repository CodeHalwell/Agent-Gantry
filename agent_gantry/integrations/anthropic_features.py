"""
Anthropic-specific features and helpers.

Provides easy access to Anthropic's beta features including:
- Interleaved thinking (between tool calls, enabled via ``interleaved-thinking-2025-05-14`` beta
  header — deprecated and silently ignored on Opus 4.6+, Sonnet 4.6+, and Opus 4.7, where
  adaptive thinking supersedes it automatically)
- Extended thinking (chain-of-thought reasoning, enabled via ``thinking={type: "enabled",
  budget_tokens: N}`` in the request body — no beta header required)
- Adaptive thinking (recommended for Opus 4.6+, Sonnet 4.6+, Opus 4.7: pass
  ``thinking={type: "adaptive", effort: "medium"}`` for the model to self-regulate)
- Tool use integration with Agent-Gantry
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Literal

from agent_gantry import AgentGantry
from agent_gantry.schema.execution import ToolCall
from agent_gantry.schema.query import ConversationContext, ToolQuery


@dataclass
class AnthropicFeatures:
    """Configuration for Anthropic beta features."""

    enable_interleaved_thinking: bool = False
    enable_extended_thinking: bool = False
    thinking_budget_tokens: int | None = None
    # Adaptive thinking: model self-regulates depth; recommended for Opus 4.6+, Sonnet 4.6+,
    # Opus 4.7. Mutually exclusive with enable_extended_thinking.
    adaptive_thinking_effort: Literal["low", "medium", "high"] | None = None


class AnthropicClient:
    """
    Enhanced Anthropic client with Agent-Gantry integration.

    Supports:
    - Interleaved thinking (``interleaved-thinking-2025-05-14`` beta header; required for
      Opus 4.5 / Sonnet 4.5 and earlier Claude 4 models; silently ignored on Opus 4.6+,
      Sonnet 4.6+, and Opus 4.7 where adaptive thinking is automatic)
    - Extended thinking (``thinking={type: "enabled", budget_tokens: N}`` in the request
      body; no beta header required on any current model)
    - Adaptive thinking (``thinking={type: "adaptive", effort: "medium"}``; recommended
      for Opus 4.6+, Sonnet 4.6+, Opus 4.7; set ``adaptive_thinking_effort`` in
      ``AnthropicFeatures`` or use ``enable_thinking="adaptive"`` in
      ``create_anthropic_client``)
    - Automatic tool retrieval and execution
    """

    def __init__(
        self,
        api_key: str | None = None,
        gantry: AgentGantry | None = None,
        features: AnthropicFeatures | None = None,
    ) -> None:
        """
        Initialize the Anthropic client with Agent-Gantry integration.

        Args:
            api_key: Anthropic API key (or use ANTHROPIC_API_KEY env var)
            gantry: AgentGantry instance for tool retrieval
            features: Feature configuration
        """
        from anthropic import AsyncAnthropic

        self._api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self._api_key:
            raise ValueError("API key required. Set ANTHROPIC_API_KEY or pass api_key parameter.")

        self._gantry = gantry
        self._features = features or AnthropicFeatures()

        # The interleaved-thinking beta header is required for Opus 4.5 / Sonnet 4.5 and
        # earlier Claude 4 models. On Opus 4.6+, Sonnet 4.6+, and Opus 4.7 the header is
        # deprecated and silently ignored — adaptive thinking is automatic on those models.
        # Extended thinking does NOT need a beta header; it is activated via the `thinking`
        # parameter in each create_message() call (see below).
        extra_headers = {}
        if self._features.enable_interleaved_thinking:
            extra_headers["anthropic-beta"] = "interleaved-thinking-2025-05-14"

        self._client = AsyncAnthropic(
            api_key=self._api_key,
            default_headers=extra_headers if extra_headers else None,
        )

    async def create_message(
        self,
        model: str,
        messages: list[dict[str, Any]],
        max_tokens: int = 4096,
        query: str | None = None,
        auto_retrieve_tools: bool = True,
        tool_limit: int = 5,
        **kwargs: Any,
    ) -> Any:
        """
        Create a message with optional tool retrieval.

        Args:
            model: Model identifier (e.g., "claude-sonnet-4-6")
            messages: Message history
            max_tokens: Maximum tokens to generate
            query: Query for tool retrieval (defaults to last user message)
            auto_retrieve_tools: Whether to automatically retrieve tools
            tool_limit: Maximum number of tools to retrieve
            **kwargs: Additional arguments passed to messages.create()

        Returns:
            Anthropic message response
        """
        # Extract query from messages if not provided
        if not query and auto_retrieve_tools:
            for msg in reversed(messages):
                if msg.get("role") == "user":
                    content = msg.get("content", "")
                    if isinstance(content, str):
                        query = content
                        break

        # Retrieve tools if enabled
        tools = None
        if auto_retrieve_tools and self._gantry and query:
            retrieval_result = await self._gantry.retrieve(
                ToolQuery(
                    context=ConversationContext(query=query),
                    limit=tool_limit,
                )
            )
            tools = [t.tool.to_dialect("anthropic") for t in retrieval_result.tools]

        # Build thinking payload — adaptive takes precedence over extended when both are set
        if self._features.adaptive_thinking_effort and "thinking" not in kwargs:
            kwargs["thinking"] = {
                "type": "adaptive",
                "effort": self._features.adaptive_thinking_effort,
            }
        elif self._features.enable_extended_thinking and self._features.thinking_budget_tokens:
            if "thinking" not in kwargs:
                kwargs["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": self._features.thinking_budget_tokens,
                }

        # Create message
        response = await self._client.messages.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            tools=tools if tools else [],
            **kwargs,
        )

        return response

    async def execute_tool_calls(
        self,
        response: Any,
    ) -> list[dict[str, Any]]:
        """
        Execute tool calls from an Anthropic response.

        Args:
            response: Anthropic message response

        Returns:
            List of tool results in Anthropic format
        """
        if not self._gantry:
            raise ValueError("AgentGantry instance required for tool execution")

        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                # Execute via Agent-Gantry
                result = await self._gantry.execute(
                    ToolCall(
                        tool_name=block.name,
                        arguments=block.input,
                    )
                )

                # Format result for Anthropic
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": str(result.result)
                        if result.status == "success"
                        else f"Error: {result.error}",
                    }
                )

        return tool_results

    def extract_thinking(
        self,
        response: Any,
    ) -> list[str]:
        """
        Extract thinking blocks from an interleaved thinking response.

        Args:
            response: Anthropic message response

        Returns:
            List of thinking text blocks
        """
        thinking_blocks = []
        for block in response.content:
            if hasattr(block, "type") and block.type == "thinking":
                if hasattr(block, "thinking"):
                    thinking_blocks.append(block.thinking)
                elif hasattr(block, "text"):
                    thinking_blocks.append(block.text)

        return thinking_blocks

    async def chat_with_thinking(
        self,
        model: str,
        messages: list[dict[str, Any]],
        show_thinking: bool = True,
        **kwargs: Any,
    ) -> tuple[Any, list[str]]:
        """
        Chat with interleaved thinking enabled.

        Args:
            model: Model identifier
            messages: Message history
            show_thinking: Whether to extract and return thinking blocks
            **kwargs: Additional arguments

        Returns:
            Tuple of (response, thinking_blocks)
        """
        if not self._features.enable_interleaved_thinking:
            raise ValueError(
                "Interleaved thinking not enabled. "
                "Set enable_interleaved_thinking=True in AnthropicFeatures"
            )

        response = await self.create_message(
            model=model,
            messages=messages,
            **kwargs,
        )

        thinking_blocks = []
        if show_thinking:
            thinking_blocks = self.extract_thinking(response)

        return response, thinking_blocks


async def create_anthropic_client(
    api_key: str | None = None,
    gantry: AgentGantry | None = None,
    enable_thinking: Literal["interleaved", "extended", "adaptive", None] = None,
    thinking_budget_tokens: int | None = None,
    adaptive_effort: Literal["low", "medium", "high"] = "medium",
) -> AnthropicClient:
    """
    Convenience function to create an Anthropic client with features.

    Args:
        api_key: Anthropic API key
        gantry: AgentGantry instance
        enable_thinking: Type of thinking to enable — ``"interleaved"`` (beta header, pre-4.6
            models), ``"extended"`` (fixed budget via ``thinking_budget_tokens``), or
            ``"adaptive"`` (model self-regulates depth; recommended for Opus 4.6+, Sonnet 4.6+,
            Opus 4.7)
        thinking_budget_tokens: Budget for extended thinking (ignored for other modes)
        adaptive_effort: Effort level for adaptive thinking — ``"low"``, ``"medium"`` (default),
            or ``"high"`` (ignored unless ``enable_thinking="adaptive"``)

    Returns:
        Configured AnthropicClient

    Example:
        >>> client = await create_anthropic_client(
        ...     gantry=gantry,
        ...     enable_thinking="adaptive",
        ...     adaptive_effort="medium",
        ... )
        >>> response, _ = await client.chat_with_thinking(
        ...     model="claude-sonnet-4-6",
        ...     messages=[{"role": "user", "content": "Explain quantum computing"}],
        ... )
    """
    features = AnthropicFeatures(
        enable_interleaved_thinking=(enable_thinking == "interleaved"),
        enable_extended_thinking=(enable_thinking == "extended"),
        thinking_budget_tokens=thinking_budget_tokens,
        adaptive_thinking_effort=adaptive_effort if enable_thinking == "adaptive" else None,
    )

    return AnthropicClient(
        api_key=api_key,
        gantry=gantry,
        features=features,
    )
