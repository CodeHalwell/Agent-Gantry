"""
Base protocol for tool specification adapters.

Defines the interface for adapters that convert ToolDefinition to provider-specific
formats and map provider tool-call payloads to unified ToolCall objects.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Any, Protocol

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from agent_gantry.schema.execution import ToolCall
    from agent_gantry.schema.tool import ToolDefinition


class ToolCallPayload(BaseModel):
    """
    Unified representation of a provider tool-call payload.

    This model captures the essential information from any provider's tool call
    format, enabling bidirectional mapping between provider-specific and
    unified representations.
    """

    tool_name: str = Field(..., description="The name of the tool being called")
    tool_call_id: str | None = Field(
        default=None, description="Provider-specific identifier for this tool call"
    )
    arguments: dict[str, Any] = Field(
        default_factory=dict, description="Arguments to pass to the tool"
    )
    raw_payload: dict[str, Any] | None = Field(
        default=None, description="Original provider payload for debugging"
    )


class ToolSpecAdapter(Protocol):
    """
    Adapter protocol for converting tool specifications between formats.

    Each implementation handles a specific provider/dialect (OpenAI, Anthropic,
    Gemini, Mistral, Groq, etc.) and provides bidirectional mapping:

    1. ToolDefinition → provider-specific specification/schema
    2. Provider tool-call payload → unified ToolCall for execution

    Implementations should handle parameter schema translation, ensuring proper
    normalization of JSON Schema to each SDK's required format.
    """

    @property
    @abstractmethod
    def dialect_name(self) -> str:
        """
        Return the dialect/provider name.

        Returns:
            String identifier for this dialect (e.g., 'openai', 'anthropic')
        """
        ...

    @abstractmethod
    def to_provider_schema(
        self,
        tool: ToolDefinition,
        **options: Any,
    ) -> dict[str, Any]:
        """
        Convert a ToolDefinition to provider-specific format.

        Args:
            tool: The canonical ToolDefinition to convert
            **options: Provider-specific options (e.g., strict mode)

        Returns:
            Provider-specific tool schema dictionary
        """
        ...

    @abstractmethod
    def from_provider_payload(
        self,
        payload: dict[str, Any],
    ) -> ToolCallPayload:
        """
        Parse a provider tool-call payload into a unified ToolCallPayload.

        Args:
            payload: Raw provider payload from the LLM response

        Returns:
            Unified ToolCallPayload for further processing
        """
        ...

    @abstractmethod
    def to_tool_call(
        self,
        payload: ToolCallPayload,
        timeout_ms: int = 30000,
        retry_count: int = 0,
    ) -> ToolCall:
        """
        Convert a ToolCallPayload to a ToolCall for execution.

        Args:
            payload: Unified tool call payload
            timeout_ms: Execution timeout in milliseconds
            retry_count: Number of retry attempts

        Returns:
            ToolCall ready for execution
        """
        ...

    @abstractmethod
    def format_tool_result(
        self,
        tool_name: str,
        result: Any,
        tool_call_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Format a tool result for sending back to the provider.

        Args:
            tool_name: Name of the executed tool
            result: Result from tool execution
            tool_call_id: Provider-specific tool call identifier

        Returns:
            Provider-formatted tool result
        """
        ...
