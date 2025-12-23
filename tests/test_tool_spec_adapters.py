"""
Tests for tool specification adapters and dialect registry.

These tests verify the adapter layer that converts ToolDefinition to
provider-specific formats and maps provider tool-call payloads to
unified ToolCall objects.
"""

from __future__ import annotations

import pytest

from agent_gantry.adapters.tool_spec import (
    DialectRegistry,
    ToolCallPayload,
    get_adapter,
)
from agent_gantry.adapters.tool_spec.providers import (
    AnthropicAdapter,
    GeminiAdapter,
    GroqAdapter,
    MistralAdapter,
    OpenAIAdapter,
    OpenAIResponsesAdapter,
)
from agent_gantry.schema.tool import SchemaDialect, ToolDefinition


@pytest.fixture
def sample_tool() -> ToolDefinition:
    """Create a sample tool definition for testing."""
    return ToolDefinition(
        name="get_weather",
        description="Get the current weather for a specified city.",
        parameters_schema={
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "The city name to get weather for",
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "Temperature unit",
                },
            },
            "required": ["city"],
        },
        tags=["weather", "api"],
    )


class TestToolCallPayload:
    """Tests for ToolCallPayload model."""

    def test_create_minimal_payload(self) -> None:
        """Test creating a payload with minimal fields."""
        payload = ToolCallPayload(tool_name="test_tool")
        assert payload.tool_name == "test_tool"
        assert payload.tool_call_id is None
        assert payload.arguments == {}
        assert payload.raw_payload is None

    def test_create_full_payload(self) -> None:
        """Test creating a payload with all fields."""
        payload = ToolCallPayload(
            tool_name="test_tool",
            tool_call_id="call_123",
            arguments={"arg1": "value1"},
            raw_payload={"id": "call_123", "type": "function"},
        )
        assert payload.tool_name == "test_tool"
        assert payload.tool_call_id == "call_123"
        assert payload.arguments == {"arg1": "value1"}
        assert payload.raw_payload == {"id": "call_123", "type": "function"}


class TestOpenAIAdapter:
    """Tests for OpenAI adapter."""

    def test_dialect_name(self) -> None:
        """Test dialect name property."""
        adapter = OpenAIAdapter()
        assert adapter.dialect_name == "openai"

    def test_to_provider_schema(self, sample_tool: ToolDefinition) -> None:
        """Test converting ToolDefinition to OpenAI schema."""
        adapter = OpenAIAdapter()
        schema = adapter.to_provider_schema(sample_tool)

        assert schema["type"] == "function"
        assert schema["function"]["name"] == "get_weather"
        assert "description" in schema["function"]
        assert "parameters" in schema["function"]
        assert schema["function"]["parameters"]["type"] == "object"

    def test_to_provider_schema_strict_mode(self, sample_tool: ToolDefinition) -> None:
        """Test converting with strict mode enabled."""
        adapter = OpenAIAdapter()
        schema = adapter.to_provider_schema(sample_tool, strict=True)

        assert schema["function"]["strict"] is True

    def test_from_provider_payload(self) -> None:
        """Test parsing OpenAI tool call payload."""
        adapter = OpenAIAdapter()
        payload = adapter.from_provider_payload({
            "id": "call_abc123",
            "type": "function",
            "function": {
                "name": "get_weather",
                "arguments": '{"city": "London"}',
            },
        })

        assert payload.tool_name == "get_weather"
        assert payload.tool_call_id == "call_abc123"
        assert payload.arguments == {"city": "London"}

    def test_from_provider_payload_invalid_json(self) -> None:
        """Test parsing payload with invalid JSON arguments."""
        adapter = OpenAIAdapter()
        payload = adapter.from_provider_payload({
            "id": "call_abc123",
            "type": "function",
            "function": {
                "name": "get_weather",
                "arguments": "invalid json",
            },
        })

        assert payload.tool_name == "get_weather"
        assert payload.arguments == {}

    def test_to_tool_call(self) -> None:
        """Test converting payload to ToolCall."""
        adapter = OpenAIAdapter()
        payload = ToolCallPayload(
            tool_name="get_weather",
            tool_call_id="call_abc123",
            arguments={"city": "London"},
        )
        call = adapter.to_tool_call(payload, timeout_ms=5000)

        assert call.tool_name == "get_weather"
        assert call.arguments == {"city": "London"}
        assert call.timeout_ms == 5000

    def test_format_tool_result(self) -> None:
        """Test formatting tool result for OpenAI."""
        adapter = OpenAIAdapter()
        result = adapter.format_tool_result(
            tool_name="get_weather",
            result={"temperature": 20, "unit": "celsius"},
            tool_call_id="call_abc123",
        )

        assert result["role"] == "tool"
        assert result["name"] == "get_weather"
        assert result["tool_call_id"] == "call_abc123"
        assert "content" in result


class TestOpenAIResponsesAdapter:
    """Tests for OpenAI Responses API adapter."""

    def test_dialect_name(self) -> None:
        """Test dialect name property."""
        adapter = OpenAIResponsesAdapter()
        assert adapter.dialect_name == "openai_responses"

    def test_to_provider_schema(self, sample_tool: ToolDefinition) -> None:
        """Test converting ToolDefinition to OpenAI Responses API schema."""
        adapter = OpenAIResponsesAdapter()
        schema = adapter.to_provider_schema(sample_tool)

        # Responses API has name at top level, not nested in function
        assert schema["type"] == "function"
        assert schema["name"] == "get_weather"
        assert "description" in schema
        assert "parameters" in schema
        assert schema["parameters"]["type"] == "object"
        # Should NOT have nested "function" key
        assert "function" not in schema

    def test_to_provider_schema_strict_mode(self, sample_tool: ToolDefinition) -> None:
        """Test converting with strict mode enabled."""
        adapter = OpenAIResponsesAdapter()
        schema = adapter.to_provider_schema(sample_tool, strict=True)

        assert schema["strict"] is True

    def test_from_provider_payload(self) -> None:
        """Test parsing OpenAI Responses API function_call payload."""
        adapter = OpenAIResponsesAdapter()
        payload = adapter.from_provider_payload({
            "type": "function_call",
            "call_id": "call_abc123",
            "name": "get_weather",
            "arguments": '{"city": "London"}',
        })

        assert payload.tool_name == "get_weather"
        assert payload.tool_call_id == "call_abc123"
        assert payload.arguments == {"city": "London"}

    def test_from_provider_payload_invalid_json(self) -> None:
        """Test parsing payload with invalid JSON arguments."""
        adapter = OpenAIResponsesAdapter()
        payload = adapter.from_provider_payload({
            "type": "function_call",
            "call_id": "call_abc123",
            "name": "get_weather",
            "arguments": "invalid json",
        })

        assert payload.tool_name == "get_weather"
        assert payload.arguments == {}

    def test_to_tool_call(self) -> None:
        """Test converting payload to ToolCall."""
        adapter = OpenAIResponsesAdapter()
        payload = ToolCallPayload(
            tool_name="get_weather",
            tool_call_id="call_abc123",
            arguments={"city": "London"},
        )
        call = adapter.to_tool_call(payload, timeout_ms=5000)

        assert call.tool_name == "get_weather"
        assert call.arguments == {"city": "London"}
        assert call.timeout_ms == 5000

    def test_format_tool_result(self) -> None:
        """Test formatting tool result for OpenAI Responses API."""
        adapter = OpenAIResponsesAdapter()
        result = adapter.format_tool_result(
            tool_name="get_weather",
            result={"temperature": 20, "unit": "celsius"},
            tool_call_id="call_abc123",
        )

        # Responses API uses function_call_output format
        assert result["type"] == "function_call_output"
        assert result["call_id"] == "call_abc123"
        assert "output" in result

    def test_format_tool_result_string(self) -> None:
        """Test formatting string result for OpenAI Responses API."""
        adapter = OpenAIResponsesAdapter()
        result = adapter.format_tool_result(
            tool_name="get_weather",
            result="Sunny, 72°F",
            tool_call_id="call_abc123",
        )

        assert result["type"] == "function_call_output"
        assert result["output"] == "Sunny, 72°F"


class TestAnthropicAdapter:
    """Tests for Anthropic adapter."""

    def test_dialect_name(self) -> None:
        """Test dialect name property."""
        adapter = AnthropicAdapter()
        assert adapter.dialect_name == "anthropic"

    def test_to_provider_schema(self, sample_tool: ToolDefinition) -> None:
        """Test converting ToolDefinition to Anthropic schema."""
        adapter = AnthropicAdapter()
        schema = adapter.to_provider_schema(sample_tool)

        assert schema["name"] == "get_weather"
        assert "description" in schema
        assert "input_schema" in schema
        assert schema["input_schema"]["type"] == "object"

    def test_from_provider_payload(self) -> None:
        """Test parsing Anthropic tool_use block."""
        adapter = AnthropicAdapter()
        payload = adapter.from_provider_payload({
            "type": "tool_use",
            "id": "toolu_abc123",
            "name": "get_weather",
            "input": {"city": "Paris"},
        })

        assert payload.tool_name == "get_weather"
        assert payload.tool_call_id == "toolu_abc123"
        assert payload.arguments == {"city": "Paris"}

    def test_format_tool_result(self) -> None:
        """Test formatting tool result for Anthropic."""
        adapter = AnthropicAdapter()
        result = adapter.format_tool_result(
            tool_name="get_weather",
            result="Sunny, 25°C",
            tool_call_id="toolu_abc123",
        )

        assert result["type"] == "tool_result"
        assert result["tool_use_id"] == "toolu_abc123"
        assert "content" in result


class TestGeminiAdapter:
    """Tests for Gemini adapter."""

    def test_dialect_name(self) -> None:
        """Test dialect name property."""
        adapter = GeminiAdapter()
        assert adapter.dialect_name == "gemini"

    def test_to_provider_schema(self, sample_tool: ToolDefinition) -> None:
        """Test converting ToolDefinition to Gemini schema."""
        adapter = GeminiAdapter()
        schema = adapter.to_provider_schema(sample_tool)

        assert schema["name"] == "get_weather"
        assert "description" in schema
        assert "parameters" in schema

    def test_from_provider_payload(self) -> None:
        """Test parsing Gemini function call."""
        adapter = GeminiAdapter()
        payload = adapter.from_provider_payload({
            "name": "get_weather",
            "args": {"city": "Tokyo"},
        })

        assert payload.tool_name == "get_weather"
        assert payload.tool_call_id is None  # Gemini doesn't provide call IDs
        assert payload.arguments == {"city": "Tokyo"}

    def test_format_tool_result(self) -> None:
        """Test formatting tool result for Gemini."""
        adapter = GeminiAdapter()
        result = adapter.format_tool_result(
            tool_name="get_weather",
            result={"temperature": 18},
        )

        assert "functionResponse" in result
        assert result["functionResponse"]["name"] == "get_weather"
        assert result["functionResponse"]["response"] == {"temperature": 18}


class TestMistralAdapter:
    """Tests for Mistral adapter."""

    def test_dialect_name(self) -> None:
        """Test dialect name property."""
        adapter = MistralAdapter()
        assert adapter.dialect_name == "mistral"

    def test_to_provider_schema(self, sample_tool: ToolDefinition) -> None:
        """Test converting ToolDefinition to Mistral schema."""
        adapter = MistralAdapter()
        schema = adapter.to_provider_schema(sample_tool)

        # Mistral uses OpenAI-compatible format
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "get_weather"

    def test_from_provider_payload(self) -> None:
        """Test parsing Mistral tool call."""
        adapter = MistralAdapter()
        payload = adapter.from_provider_payload({
            "id": "call_123",
            "type": "function",
            "function": {
                "name": "get_weather",
                "arguments": '{"city": "Berlin"}',
            },
        })

        assert payload.tool_name == "get_weather"
        assert payload.arguments == {"city": "Berlin"}


class TestGroqAdapter:
    """Tests for Groq adapter."""

    def test_dialect_name(self) -> None:
        """Test dialect name property."""
        adapter = GroqAdapter()
        assert adapter.dialect_name == "groq"

    def test_to_provider_schema(self, sample_tool: ToolDefinition) -> None:
        """Test converting ToolDefinition to Groq schema."""
        adapter = GroqAdapter()
        schema = adapter.to_provider_schema(sample_tool)

        # Groq uses OpenAI-compatible format
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "get_weather"

    def test_from_provider_payload(self) -> None:
        """Test parsing Groq tool call."""
        adapter = GroqAdapter()
        payload = adapter.from_provider_payload({
            "id": "call_groq_123",
            "type": "function",
            "function": {
                "name": "get_weather",
                "arguments": '{"city": "Sydney"}',
            },
        })

        assert payload.tool_name == "get_weather"
        assert payload.arguments == {"city": "Sydney"}


class TestDialectRegistry:
    """Tests for dialect registry."""

    def test_default_registry_has_adapters(self) -> None:
        """Test that default registry has built-in adapters."""
        registry = DialectRegistry.default()

        assert registry.has("openai")
        assert registry.has("openai_responses")
        assert registry.has("anthropic")
        assert registry.has("gemini")
        assert registry.has("mistral")
        assert registry.has("groq")
        assert registry.has("auto")

    def test_get_adapter(self) -> None:
        """Test getting an adapter by dialect name."""
        registry = DialectRegistry.default()

        openai_adapter = registry.get("openai")
        assert openai_adapter.dialect_name == "openai"

        anthropic_adapter = registry.get("anthropic")
        assert anthropic_adapter.dialect_name == "anthropic"

    def test_get_auto_defaults_to_openai(self) -> None:
        """Test that 'auto' dialect defaults to OpenAI."""
        registry = DialectRegistry.default()
        adapter = registry.get("auto")
        assert adapter.dialect_name == "openai"

    def test_get_unknown_dialect_raises(self) -> None:
        """Test that getting unknown dialect raises KeyError."""
        registry = DialectRegistry()  # Fresh registry without defaults
        with pytest.raises(KeyError):
            registry.get("unknown_dialect")

    def test_register_custom_adapter(self) -> None:
        """Test registering a custom adapter."""
        registry = DialectRegistry()
        adapter = OpenAIAdapter()
        registry.register(adapter)

        assert registry.has("openai")
        assert registry.get("openai") is adapter

    def test_list_dialects(self) -> None:
        """Test listing registered dialects."""
        registry = DialectRegistry.default()
        dialects = registry.list_dialects()

        assert "openai" in dialects
        assert "anthropic" in dialects
        assert "gemini" in dialects

    def test_unregister(self) -> None:
        """Test unregistering an adapter."""
        registry = DialectRegistry()
        registry.register(OpenAIAdapter())
        assert registry.has("openai")

        result = registry.unregister("openai")
        assert result is True
        assert not registry.has("openai")

    def test_clear(self) -> None:
        """Test clearing all adapters."""
        registry = DialectRegistry()
        registry.register(OpenAIAdapter())
        registry.register(AnthropicAdapter())

        registry.clear()
        assert registry.list_dialects() == []


class TestGetAdapterFunction:
    """Tests for the get_adapter convenience function."""

    def test_get_adapter_default(self) -> None:
        """Test get_adapter with default (auto) dialect."""
        adapter = get_adapter()
        assert adapter.dialect_name == "openai"

    def test_get_adapter_specific(self) -> None:
        """Test get_adapter with specific dialect."""
        adapter = get_adapter("anthropic")
        assert adapter.dialect_name == "anthropic"


class TestToolDefinitionToDialect:
    """Tests for ToolDefinition.to_dialect with registry integration."""

    def test_to_dialect_openai(self, sample_tool: ToolDefinition) -> None:
        """Test to_dialect with OpenAI dialect."""
        schema = sample_tool.to_dialect(SchemaDialect.OPENAI)
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "get_weather"

    def test_to_dialect_anthropic(self, sample_tool: ToolDefinition) -> None:
        """Test to_dialect with Anthropic dialect."""
        schema = sample_tool.to_dialect(SchemaDialect.ANTHROPIC)
        assert schema["name"] == "get_weather"
        assert "input_schema" in schema

    def test_to_dialect_gemini(self, sample_tool: ToolDefinition) -> None:
        """Test to_dialect with Gemini dialect."""
        schema = sample_tool.to_dialect(SchemaDialect.GEMINI)
        assert schema["name"] == "get_weather"
        assert "parameters" in schema

    def test_to_dialect_mistral(self, sample_tool: ToolDefinition) -> None:
        """Test to_dialect with Mistral dialect."""
        schema = sample_tool.to_dialect(SchemaDialect.MISTRAL)
        assert schema["type"] == "function"

    def test_to_dialect_groq(self, sample_tool: ToolDefinition) -> None:
        """Test to_dialect with Groq dialect."""
        schema = sample_tool.to_dialect(SchemaDialect.GROQ)
        assert schema["type"] == "function"

    def test_to_dialect_openai_responses(self, sample_tool: ToolDefinition) -> None:
        """Test to_dialect with OpenAI Responses API dialect."""
        schema = sample_tool.to_dialect(SchemaDialect.OPENAI_RESPONSES)
        assert schema["type"] == "function"
        assert schema["name"] == "get_weather"
        assert "parameters" in schema
        # Should NOT have nested "function" key (unlike chat completions)
        assert "function" not in schema

    def test_to_dialect_string_name(self, sample_tool: ToolDefinition) -> None:
        """Test to_dialect with string dialect name."""
        schema = sample_tool.to_dialect("anthropic")
        assert schema["name"] == "get_weather"
        assert "input_schema" in schema

    def test_to_dialect_auto(self, sample_tool: ToolDefinition) -> None:
        """Test to_dialect with AUTO defaults to OpenAI."""
        schema = sample_tool.to_dialect(SchemaDialect.AUTO)
        assert schema["type"] == "function"

    def test_to_dialect_with_options(self, sample_tool: ToolDefinition) -> None:
        """Test to_dialect passes options to adapter."""
        schema = sample_tool.to_dialect("openai", strict=True)
        assert schema["function"]["strict"] is True


class TestRetrievalResultToDialect:
    """Tests for RetrievalResult.to_dialect with registry integration."""

    @pytest.mark.asyncio
    async def test_retrieval_result_to_dialect(self) -> None:
        """Test converting retrieval results to different dialects."""
        from agent_gantry import AgentGantry

        gantry = AgentGantry()

        @gantry.register
        def test_tool(x: int) -> str:
            """A test tool for validation."""
            return str(x)

        await gantry.sync()

        result = await gantry.retrieve_tools("test", limit=1, dialect="anthropic")

        assert len(result) >= 1
        # Anthropic format has 'name' and 'input_schema'
        assert "name" in result[0]
        assert "input_schema" in result[0]

    @pytest.mark.asyncio
    async def test_retrieve_tools_openai_dialect(self) -> None:
        """Test retrieve_tools with OpenAI dialect (default)."""
        from agent_gantry import AgentGantry

        gantry = AgentGantry()

        @gantry.register
        def calc(a: int, b: int) -> int:
            """Calculate something with two numbers."""
            return a + b

        await gantry.sync()

        # Use low score threshold to ensure match with simple embedder
        result = await gantry.retrieve_tools(
            "calculate", limit=1, dialect="openai", score_threshold=0.0
        )

        assert len(result) >= 1
        assert result[0]["type"] == "function"
        assert "function" in result[0]

    @pytest.mark.asyncio
    async def test_retrieve_tools_gemini_dialect(self) -> None:
        """Test retrieve_tools with Gemini dialect."""
        from agent_gantry import AgentGantry

        gantry = AgentGantry()

        @gantry.register
        def search(query: str) -> str:
            """Search for information."""
            return f"Results for: {query}"

        await gantry.sync()

        result = await gantry.retrieve_tools("search", limit=1, dialect="gemini")

        assert len(result) >= 1
        assert "name" in result[0]
        assert "parameters" in result[0]

    @pytest.mark.asyncio
    async def test_retrieve_tools_openai_responses_dialect(self) -> None:
        """Test retrieve_tools with OpenAI Responses API dialect."""
        from agent_gantry import AgentGantry

        gantry = AgentGantry()

        @gantry.register
        def send_email(to: str, subject: str, body: str) -> str:
            """Send an email to the specified recipient."""
            return f"Email sent to {to}"

        await gantry.sync()

        result = await gantry.retrieve_tools(
            "send email", limit=1, dialect="openai_responses"
        )

        assert len(result) >= 1
        # Responses API format has name at top level
        assert result[0]["type"] == "function"
        assert "name" in result[0]
        assert "parameters" in result[0]
        # Should NOT have nested "function" key
        assert "function" not in result[0]
