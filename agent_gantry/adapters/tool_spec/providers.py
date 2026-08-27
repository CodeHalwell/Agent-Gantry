"""
Provider-specific tool specification adapters.

Implementations for OpenAI (Chat Completions & Responses API), Anthropic,
Gemini, Mistral, Groq, and Microsoft Agent Framework.
"""

from __future__ import annotations

import copy
import json
import logging
from typing import TYPE_CHECKING, Any

from agent_gantry.adapters.tool_spec.base import ToolCallPayload
from agent_gantry.adapters.tool_spec.schema_utils import (
    sanitize_gemini_schema,
    strict_json_schema,
    unsupported_strict_paths,
)
from agent_gantry.schema.execution import ToolCall

if TYPE_CHECKING:
    from agent_gantry.schema.tool import ToolDefinition

_logger = logging.getLogger(__name__)


def _emitted_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """A caller-owned deep copy of a tool's parameter schema.

    The registry holds one canonical ``ToolDefinition.parameters_schema``
    per tool, so a payload that aliases it lets any caller that augments
    the emitted schema corrupt every later conversion of that tool — and
    the executor's own validation, which reads the same object.
    :func:`strict_json_schema` and :func:`sanitize_gemini_schema` already
    deep-copy their input; every pass-through path has to do it here.
    """
    return copy.deepcopy(schema)


def _strict_parameters(tool: ToolDefinition, dialect: str) -> tuple[dict[str, Any], bool]:
    """Strict-mode ``parameters`` for ``tool``, and whether strict is usable.

    OpenAI rejects the whole request when a tool marked ``strict: true``
    carries an object with arbitrary keys — a ``dict[str, int]`` parameter,
    an untyped ``dict`` — because strict mode cannot express one. Publishing
    the tool without the flag keeps it callable (merely unconstrained),
    where honouring the caller's ``strict=True`` verbatim would take down
    every request the tool appears in.
    """
    unsupported = unsupported_strict_paths(tool.parameters_schema)
    if unsupported:
        _logger.warning(
            "Tool %r cannot use %s strict mode: %s describes an object with "
            "arbitrary keys, which strict mode cannot express. Emitting the "
            "tool without strict:true so the request still succeeds — declare "
            "the object's properties explicitly to make it strict-compatible.",
            tool.name,
            dialect,
            ", ".join(unsupported),
        )
        return _emitted_schema(tool.parameters_schema), False
    return strict_json_schema(tool.parameters_schema), True


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
            strict: Enable OpenAI's structured-outputs strict mode. The
                parameter schema is rewritten to satisfy it: every object gets
                ``additionalProperties: false`` and lists all of its properties
                in ``required``, with formerly-optional properties widened to
                admit ``null``. Setting the flag without that rewrite makes the
                API reject any tool that has an optional parameter. The
                ToolDefinition's own schema is never mutated. Defaults to False.
                A tool whose schema contains an object with arbitrary keys (a
                ``dict[str, int]`` parameter, an untyped ``dict``) has no
                strict-mode representation; it is emitted unmodified and
                *without* ``strict: true``, with a warning, since claiming
                strict there makes OpenAI reject the entire request.
                Source: https://platform.openai.com/docs/guides/function-calling
            **options: Additional provider-specific options

        Returns:
            OpenAI-compatible tool schema
        """
        # ``_strict_parameters`` returns its own copy on both branches, so
        # only the non-strict path needs one made here.
        if strict:
            parameters, use_strict = _strict_parameters(tool, self.dialect_name)
        else:
            parameters, use_strict = _emitted_schema(tool.parameters_schema), False
        schema: dict[str, Any] = {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": parameters,
            },
        }
        if use_strict:
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
                _logger.warning(
                    "OpenAIAdapter: malformed JSON in tool arguments "
                    "for '%s', defaulting to empty dict: %s",
                    tool_name,
                    arguments[:200],
                )
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
        *,
        is_error: bool = False,
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
            strict: Enable strict mode. The parameter schema is rewritten to
                satisfy it (``additionalProperties: false``, every property in
                ``required``, formerly-optional properties widened to admit
                ``null``); the flag alone would make the API reject any tool
                with an optional parameter. The ToolDefinition's own schema is
                never mutated. Defaults to False. A tool whose schema contains
                an object with arbitrary keys (a ``dict[str, int]`` parameter,
                an untyped ``dict``) has no strict-mode representation; it is
                emitted unmodified and *without* ``strict: true``, with a
                warning, since claiming strict there makes the API reject the
                entire request.
            **options: Additional provider-specific options

        Returns:
            OpenAI Responses API compatible tool schema
        """
        # ``_strict_parameters`` returns its own copy on both branches, so
        # only the non-strict path needs one made here.
        if strict:
            parameters, use_strict = _strict_parameters(tool, self.dialect_name)
        else:
            parameters, use_strict = _emitted_schema(tool.parameters_schema), False
        schema: dict[str, Any] = {
            "type": "function",
            "name": tool.name,
            "description": tool.description,
            "parameters": parameters,
        }
        if use_strict:
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
                _logger.warning(
                    "OpenAIResponsesAdapter: malformed JSON in tool arguments "
                    "for '%s', defaulting to empty dict: %s",
                    tool_name,
                    arguments[:200],
                )
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
        *,
        is_error: bool = False,
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
        *,
        strict: bool = False,
        **options: Any,
    ) -> dict[str, Any]:
        """
        Convert ToolDefinition to Anthropic tool format.

        Args:
            tool: The tool definition to convert
            strict: Enable Anthropic strict tool use — uses grammar-constrained
                sampling so Claude's ``input`` always matches ``input_schema``
                exactly.  When ``True`` this method automatically injects
                ``additionalProperties: false`` into the ``input_schema`` copy
                (required by the Anthropic API for strict mode to take effect).
                The original schema on the ToolDefinition is never mutated.
                Defaults to False.
                Source: https://platform.claude.com/docs/en/agents-and-tools/tool-use/strict-tool-use
            **options: Additional provider-specific options

        Returns:
            Anthropic-compatible tool schema
        """
        # Always emit a deep copy: a ``{**schema}`` spread would leave every
        # nested property dict aliasing the shared
        # ToolDefinition.parameters_schema, so a caller adjusting one nested
        # subschema on the payload would still corrupt the registered tool.
        input_schema: dict[str, Any] = _emitted_schema(tool.parameters_schema)
        use_strict = False
        if strict:
            # Same gate as the OpenAI adapters: strict mode requires
            # ``additionalProperties: false``, so a schema that *needs*
            # arbitrary keys cannot be strict. Setting the flag anyway would
            # replace a typed mapping (``dict[str, int]``) or a ``**kwargs``
            # handler's open object with one accepting no keys at all —
            # silently discarding the parameter's data rather than merely
            # leaving it unconstrained.
            unsupported = unsupported_strict_paths(tool.parameters_schema)
            if unsupported:
                _logger.warning(
                    "Tool %r cannot use %s strict mode: %s describes an "
                    "object with arbitrary keys, which strict mode cannot "
                    "express. Emitting the tool without strict:true so the "
                    "keys stay usable — declare the object's properties "
                    "explicitly to make it strict-compatible.",
                    tool.name,
                    self.dialect_name,
                    ", ".join(unsupported),
                )
            else:
                use_strict = True
                # Anthropic requires additionalProperties: false for strict
                # mode to take effect.
                input_schema["additionalProperties"] = False

        schema: dict[str, Any] = {
            "name": tool.name,
            "description": tool.description,
            "input_schema": input_schema,
        }
        if use_strict:
            schema["strict"] = True
        return schema

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
        *,
        is_error: bool = False,
    ) -> dict[str, Any]:
        """Format result for Anthropic tool_result.

        Pass ``is_error=True`` when the tool execution failed so the model can
        distinguish error content from normal tool output.
        Source: https://platform.claude.com/docs/en/api/messages (tool_result block)
        """
        content = result if isinstance(result, str) else json.dumps(result)
        response: dict[str, Any] = {
            "type": "tool_result",
            "content": content,
        }
        if is_error:
            response["is_error"] = True
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

        The parameter schema is sanitized first: Gemini and Vertex AI reject
        unknown JSON-Schema keywords rather than ignoring them, so keywords
        that are valid everywhere else (``additionalProperties``, ``default``,
        ``title``, …) would turn into request errors. This matters for
        Agent-Gantry's own tools — introspecting a ``**kwargs`` handler emits
        ``additionalProperties``, which alone breaks the Gemini path. Local
        ``$ref``/``$defs`` pairs (what Pydantic emits for nested models) are
        inlined, since the SDKs will not follow the pointers.

        Args:
            tool: The tool definition to convert
            sanitize: Set ``False`` to emit ``parameters_schema`` verbatim,
                for callers on an SDK version that accepts more keywords.
                Defaults to True.
            **options: Additional provider-specific options

        Returns:
            Gemini-compatible function declaration
        """
        sanitize = options.get("sanitize", True)
        return {
            "name": tool.name,
            "description": tool.description,
            "parameters": (
                sanitize_gemini_schema(tool.parameters_schema)
                if sanitize
                else _emitted_schema(tool.parameters_schema)
            ),
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
            "args": {"arg": "value"},
            "id": "..."   # present in google-genai >= 1.x when parallel calls are made
        }
        """
        tool_name = payload.get("name", "")
        arguments = payload.get("args", {})
        # google-genai >= 1.x includes an "id" on parallel function calls
        tool_call_id = payload.get("id")

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
        *,
        is_error: bool = False,
    ) -> dict[str, Any]:
        """Format result for Gemini function response."""
        response_content = result if isinstance(result, dict) else {"result": result}
        payload: dict[str, Any] = {
            "functionResponse": {
                "name": tool_name,
                "response": response_content,
            }
        }
        # The id field belongs inside functionResponse (it is a field on the
        # FunctionResponse proto message, not on Part itself — Part has no id
        # field). Echo back the call id so the model can correlate parallel
        # function calls with their results.
        # Source: google-genai types.py — FunctionResponse.id field description:
        # "The id of the function call this response is for"
        # https://ai.google.dev/gemini-api/docs/function-calling
        if tool_call_id:
            payload["functionResponse"]["id"] = tool_call_id
        return payload


class MistralAdapter(OpenAIAdapter):
    """
    Tool specification adapter for Mistral AI.

    Mistral uses OpenAI-compatible function calling format,
    so this inherits all behavior from OpenAIAdapter.
    """

    @property
    def dialect_name(self) -> str:
        return "mistral"


class GroqAdapter(OpenAIAdapter):
    """
    Tool specification adapter for Groq.

    Groq uses OpenAI-compatible function calling format,
    so this inherits all behavior from OpenAIAdapter.
    """

    @property
    def dialect_name(self) -> str:
        return "groq"


class AgentFrameworkAdapter(OpenAIAdapter):
    """
    Tool specification adapter for Microsoft Agent Framework (1.0 GA).

    Microsoft Agent Framework uses OpenAI-compatible function calling format
    for its tool schemas, so this inherits from ``OpenAIAdapter`` and adds:

    - Optional ``include_metadata`` support for Gantry provenance info
    - Simplified AF internal payload format (``{"name": ..., "arguments": ...}``)

    For direct tool wrapping (Python callables), use the higher-level
    ``agent_gantry.integrations.agent_framework_bridge`` module instead.
    """

    @property
    def dialect_name(self) -> str:
        return "agent_framework"

    def to_provider_schema(
        self,
        tool: ToolDefinition,
        **options: Any,
    ) -> dict[str, Any]:
        """
        Convert ToolDefinition to Microsoft Agent Framework tool schema.

        Delegates to OpenAIAdapter for the base schema and optionally appends
        Gantry-specific metadata for provenance tracking.
        """
        schema = super().to_provider_schema(tool, **options)
        if options.get("include_metadata", False):
            schema["function"]["metadata"] = {
                "namespace": tool.namespace,
                "version": tool.version,
                "source": tool.source.value if hasattr(tool.source, "value") else str(tool.source),
            }
        return schema

    def from_provider_payload(
        self,
        payload: dict[str, Any],
    ) -> ToolCallPayload:
        """
        Parse a Microsoft Agent Framework tool call payload.

        Handles both the standard OpenAI-style format (via super()) and the
        simplified format used by AF's internal dispatch:
        ``{"name": "tool_name", "arguments": {"arg": "value"}}``
        """
        # Standard OpenAI format — delegate to parent
        if "function" in payload:
            return super().from_provider_payload(payload)

        # Simplified AF internal format
        tool_call_id = payload.get("id") or payload.get("call_id")
        tool_name = payload.get("name", "")
        arguments = payload.get("arguments", {})

        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                _logger.warning(
                    "AgentFrameworkAdapter: malformed JSON in tool arguments "
                    "for '%s', defaulting to empty dict: %s",
                    tool_name,
                    arguments[:200] if isinstance(arguments, str) else arguments,
                )
                arguments = {}

        return ToolCallPayload(
            tool_name=tool_name,
            tool_call_id=tool_call_id,
            arguments=arguments,
            raw_payload=payload,
        )
