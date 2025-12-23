"""
Provider-specific tool specification adapters.

Implementations for OpenAI (Chat Completions & Responses API), Anthropic,
Gemini, Mistral, and Groq.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from agent_gantry.adapters.tool_spec.base import ToolCallPayload
from agent_gantry.schema.execution import ToolCall

if TYPE_CHECKING:
    from agent_gantry.schema.tool import ToolDefinition


class OpenAIAdapter:
    """
    Tool specification adapter for OpenAI Chat Completions API.

    Converts to/from OpenAI function calling format:
    {
        "type": "function",
        "function": {
            "name": "...",
            "description": "...",
            "parameters": {...}
        }
    }

    For the Responses API, use OpenAIResponsesAdapter instead.
    """

    @property
    def dialect_name(self) -> str:
        return "openai"

    def to_provider_schema(
        self,
        tool: ToolDefinition,
        *,
        strict: bool = False,
        **options: Any,
    ) -> dict[str, Any]:
        """
        Convert ToolDefinition to OpenAI function calling format.

        Args:
            tool: The tool definition to convert
            strict: Enable OpenAI's strict mode (default: False)
            **options: Additional provider-specific options

        Returns:
            OpenAI-compatible tool schema
        """
        schema: dict[str, Any] = {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters_schema,
            },
        }
        if strict:
            schema["function"]["strict"] = True
        return schema

    def from_provider_payload(
        self,
        payload: dict[str, Any],
    ) -> ToolCallPayload:
        """
        Parse an OpenAI tool call from the API response.

        Expected format:
        {
            "id": "call_xxx",
            "type": "function",
            "function": {
                "name": "tool_name",
                "arguments": "{\"arg\": \"value\"}"
            }
        }
        """
        tool_call_id = payload.get("id")
        function_data = payload.get("function", {})
        tool_name = function_data.get("name", "")

        # Arguments may be a JSON string or already parsed
        arguments = function_data.get("arguments", {})
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {}

        return ToolCallPayload(
            tool_name=tool_name,
            tool_call_id=tool_call_id,
            arguments=arguments,
            raw_payload=payload,
        )

    def to_tool_call(
        self,
        payload: ToolCallPayload,
        timeout_ms: int = 30000,
        retry_count: int = 0,
    ) -> ToolCall:
        return ToolCall(
            tool_name=payload.tool_name,
            arguments=payload.arguments,
            timeout_ms=timeout_ms,
            retry_count=retry_count,
            trace_id=payload.tool_call_id,
        )

    def format_tool_result(
        self,
        tool_name: str,
        result: Any,
        tool_call_id: str | None = None,
    ) -> dict[str, Any]:
        """Format result for OpenAI tool_outputs."""
        content = result if isinstance(result, str) else json.dumps(result)
        response: dict[str, Any] = {
            "role": "tool",
            "content": content,
            "name": tool_name,
        }
        if tool_call_id:
            response["tool_call_id"] = tool_call_id
        return response


class OpenAIResponsesAdapter:
    """
    Tool specification adapter for OpenAI Responses API.

    The Responses API is a newer API that uses a different format:
    - Tools are specified with "type": "function" and "name" at top level
    - Tool calls come as output items with type "function_call"
    - Tool results use "function_call_output" type

    Tool schema format:
    {
        "type": "function",
        "name": "...",
        "description": "...",
        "parameters": {...}
    }

    Tool call format (from response.output):
    {
        "type": "function_call",
        "call_id": "...",
        "name": "tool_name",
        "arguments": "{\"arg\": \"value\"}"
    }

    Tool result format:
    {
        "type": "function_call_output",
        "call_id": "...",
        "output": "result string"
    }
    """

    @property
    def dialect_name(self) -> str:
        return "openai_responses"

    def to_provider_schema(
        self,
        tool: ToolDefinition,
        *,
        strict: bool = False,
        **options: Any,
    ) -> dict[str, Any]:
        """
        Convert ToolDefinition to OpenAI Responses API function format.

        Args:
            tool: The tool definition to convert
            strict: Enable strict mode (default: False)
            **options: Additional provider-specific options

        Returns:
            OpenAI Responses API compatible tool schema
        """
        schema: dict[str, Any] = {
            "type": "function",
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters_schema,
        }
        if strict:
            schema["strict"] = True
        return schema

    def from_provider_payload(
        self,
        payload: dict[str, Any],
    ) -> ToolCallPayload:
        """
        Parse an OpenAI Responses API function_call from the response output.

        Expected format (from response.output array):
        {
            "type": "function_call",
            "call_id": "call_xxx",
            "name": "tool_name",
            "arguments": "{\"arg\": \"value\"}"
        }
        """
        tool_call_id = payload.get("call_id")
        tool_name = payload.get("name", "")

        # Arguments may be a JSON string or already parsed
        arguments = payload.get("arguments", {})
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {}

        return ToolCallPayload(
            tool_name=tool_name,
            tool_call_id=tool_call_id,
            arguments=arguments,
            raw_payload=payload,
        )

    def to_tool_call(
        self,
        payload: ToolCallPayload,
        timeout_ms: int = 30000,
        retry_count: int = 0,
    ) -> ToolCall:
        return ToolCall(
            tool_name=payload.tool_name,
            arguments=payload.arguments,
            timeout_ms=timeout_ms,
            retry_count=retry_count,
            trace_id=payload.tool_call_id,
        )

    def format_tool_result(
        self,
        tool_name: str,
        result: Any,
        tool_call_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Format result for OpenAI Responses API function_call_output.

        Returns format suitable for sending back as input to responses.create():
        {
            "type": "function_call_output",
            "call_id": "...",
            "output": "result string"
        }
        """
        output = result if isinstance(result, str) else json.dumps(result)
        response: dict[str, Any] = {
            "type": "function_call_output",
            "output": output,
        }
        if tool_call_id:
            response["call_id"] = tool_call_id
        return response


class AnthropicAdapter:
    """
    Tool specification adapter for Anthropic (Claude).

    Converts to/from Anthropic tool format:
    {
        "name": "...",
        "description": "...",
        "input_schema": {...}
    }
    """

    @property
    def dialect_name(self) -> str:
        return "anthropic"

    def to_provider_schema(
        self,
        tool: ToolDefinition,
        **options: Any,
    ) -> dict[str, Any]:
        """
        Convert ToolDefinition to Anthropic tool format.

        Args:
            tool: The tool definition to convert
            **options: Additional provider-specific options

        Returns:
            Anthropic-compatible tool schema
        """
        return {
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.parameters_schema,
        }

    def from_provider_payload(
        self,
        payload: dict[str, Any],
    ) -> ToolCallPayload:
        """
        Parse an Anthropic tool_use block from the API response.

        Expected format:
        {
            "type": "tool_use",
            "id": "toolu_xxx",
            "name": "tool_name",
            "input": {"arg": "value"}
        }
        """
        tool_call_id = payload.get("id")
        tool_name = payload.get("name", "")
        arguments = payload.get("input", {})

        return ToolCallPayload(
            tool_name=tool_name,
            tool_call_id=tool_call_id,
            arguments=arguments if isinstance(arguments, dict) else {},
            raw_payload=payload,
        )

    def to_tool_call(
        self,
        payload: ToolCallPayload,
        timeout_ms: int = 30000,
        retry_count: int = 0,
    ) -> ToolCall:
        return ToolCall(
            tool_name=payload.tool_name,
            arguments=payload.arguments,
            timeout_ms=timeout_ms,
            retry_count=retry_count,
            trace_id=payload.tool_call_id,
        )

    def format_tool_result(
        self,
        tool_name: str,
        result: Any,
        tool_call_id: str | None = None,
    ) -> dict[str, Any]:
        """Format result for Anthropic tool_result."""
        content = result if isinstance(result, str) else json.dumps(result)
        response: dict[str, Any] = {
            "type": "tool_result",
            "content": content,
        }
        if tool_call_id:
            response["tool_use_id"] = tool_call_id
        return response


class GeminiAdapter:
    """
    Tool specification adapter for Google Gemini.

    Converts to/from Gemini function declaration format:
    {
        "name": "...",
        "description": "...",
        "parameters": {...}
    }
    """

    @property
    def dialect_name(self) -> str:
        return "gemini"

    def to_provider_schema(
        self,
        tool: ToolDefinition,
        **options: Any,
    ) -> dict[str, Any]:
        """
        Convert ToolDefinition to Gemini function declaration format.

        Args:
            tool: The tool definition to convert
            **options: Additional provider-specific options

        Returns:
            Gemini-compatible function declaration
        """
        return {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters_schema,
        }

    def from_provider_payload(
        self,
        payload: dict[str, Any],
    ) -> ToolCallPayload:
        """
        Parse a Gemini function call from the API response.

        Expected format (from functionCall):
        {
            "name": "tool_name",
            "args": {"arg": "value"}
        }
        """
        tool_name = payload.get("name", "")
        arguments = payload.get("args", {})

        return ToolCallPayload(
            tool_name=tool_name,
            tool_call_id=None,  # Gemini doesn't provide call IDs
            arguments=arguments if isinstance(arguments, dict) else {},
            raw_payload=payload,
        )

    def to_tool_call(
        self,
        payload: ToolCallPayload,
        timeout_ms: int = 30000,
        retry_count: int = 0,
    ) -> ToolCall:
        return ToolCall(
            tool_name=payload.tool_name,
            arguments=payload.arguments,
            timeout_ms=timeout_ms,
            retry_count=retry_count,
        )

    def format_tool_result(
        self,
        tool_name: str,
        result: Any,
        tool_call_id: str | None = None,
    ) -> dict[str, Any]:
        """Format result for Gemini function response."""
        response_content = result if isinstance(result, dict) else {"result": result}
        return {
            "functionResponse": {
                "name": tool_name,
                "response": response_content,
            }
        }


class MistralAdapter:
    """
    Tool specification adapter for Mistral AI.

    Mistral uses OpenAI-compatible function calling format:
    {
        "type": "function",
        "function": {
            "name": "...",
            "description": "...",
            "parameters": {...}
        }
    }
    """

    @property
    def dialect_name(self) -> str:
        return "mistral"

    def to_provider_schema(
        self,
        tool: ToolDefinition,
        **options: Any,
    ) -> dict[str, Any]:
        """
        Convert ToolDefinition to Mistral function format.

        Mistral uses OpenAI-compatible format.
        """
        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters_schema,
            },
        }

    def from_provider_payload(
        self,
        payload: dict[str, Any],
    ) -> ToolCallPayload:
        """
        Parse a Mistral tool call from the API response.

        Format is OpenAI-compatible:
        {
            "id": "...",
            "type": "function",
            "function": {
                "name": "tool_name",
                "arguments": "{\"arg\": \"value\"}"
            }
        }
        """
        tool_call_id = payload.get("id")
        function_data = payload.get("function", {})
        tool_name = function_data.get("name", "")

        arguments = function_data.get("arguments", {})
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {}

        return ToolCallPayload(
            tool_name=tool_name,
            tool_call_id=tool_call_id,
            arguments=arguments,
            raw_payload=payload,
        )

    def to_tool_call(
        self,
        payload: ToolCallPayload,
        timeout_ms: int = 30000,
        retry_count: int = 0,
    ) -> ToolCall:
        return ToolCall(
            tool_name=payload.tool_name,
            arguments=payload.arguments,
            timeout_ms=timeout_ms,
            retry_count=retry_count,
            trace_id=payload.tool_call_id,
        )

    def format_tool_result(
        self,
        tool_name: str,
        result: Any,
        tool_call_id: str | None = None,
    ) -> dict[str, Any]:
        """Format result for Mistral tool response."""
        content = result if isinstance(result, str) else json.dumps(result)
        response: dict[str, Any] = {
            "role": "tool",
            "content": content,
            "name": tool_name,
        }
        if tool_call_id:
            response["tool_call_id"] = tool_call_id
        return response


class GroqAdapter:
    """
    Tool specification adapter for Groq.

    Groq uses OpenAI-compatible function calling format:
    {
        "type": "function",
        "function": {
            "name": "...",
            "description": "...",
            "parameters": {...}
        }
    }
    """

    @property
    def dialect_name(self) -> str:
        return "groq"

    def to_provider_schema(
        self,
        tool: ToolDefinition,
        **options: Any,
    ) -> dict[str, Any]:
        """
        Convert ToolDefinition to Groq function format.

        Groq uses OpenAI-compatible format.
        """
        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters_schema,
            },
        }

    def from_provider_payload(
        self,
        payload: dict[str, Any],
    ) -> ToolCallPayload:
        """
        Parse a Groq tool call from the API response.

        Format is OpenAI-compatible:
        {
            "id": "call_xxx",
            "type": "function",
            "function": {
                "name": "tool_name",
                "arguments": "{\"arg\": \"value\"}"
            }
        }
        """
        tool_call_id = payload.get("id")
        function_data = payload.get("function", {})
        tool_name = function_data.get("name", "")

        arguments = function_data.get("arguments", {})
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {}

        return ToolCallPayload(
            tool_name=tool_name,
            tool_call_id=tool_call_id,
            arguments=arguments,
            raw_payload=payload,
        )

    def to_tool_call(
        self,
        payload: ToolCallPayload,
        timeout_ms: int = 30000,
        retry_count: int = 0,
    ) -> ToolCall:
        return ToolCall(
            tool_name=payload.tool_name,
            arguments=payload.arguments,
            timeout_ms=timeout_ms,
            retry_count=retry_count,
            trace_id=payload.tool_call_id,
        )

    def format_tool_result(
        self,
        tool_name: str,
        result: Any,
        tool_call_id: str | None = None,
    ) -> dict[str, Any]:
        """Format result for Groq tool response."""
        content = result if isinstance(result, str) else json.dumps(result)
        response: dict[str, Any] = {
            "role": "tool",
            "content": content,
            "name": tool_name,
        }
        if tool_call_id:
            response["tool_call_id"] = tool_call_id
        return response
