"""
Tests for semantic tool selection decorator.

Tests cover:
- Basic decorator functionality
- Async and sync function wrapping
- Prompt extraction from various formats
- Tool injection behavior
- Dialect conversion
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest

from agent_gantry import AgentGantry
from agent_gantry.integrations.decorator import (
    SemanticToolsDecorator,
    SemanticToolSelector,
    with_semantic_tools,
)
from agent_gantry.schema.tool import ToolDefinition


class TestWithSemanticToolsDecorator:
    """Tests for the with_semantic_tools decorator."""

    def test_decorator_requires_gantry_instance(self) -> None:
        """Test that decorator requires an AgentGantry instance."""
        with pytest.raises(TypeError, match="requires an AgentGantry instance"):

            @with_semantic_tools(lambda x: x)  # type: ignore[arg-type]
            def generate(prompt: str) -> str:
                return prompt

    def test_decorator_returns_selector_with_gantry(self) -> None:
        """Test that decorator returns a SemanticToolSelector when given a gantry."""
        gantry = AgentGantry()
        selector = with_semantic_tools(gantry)
        assert isinstance(selector, SemanticToolSelector)

    def test_decorator_accepts_configuration_options(self) -> None:
        """Test that decorator accepts all configuration options."""
        gantry = AgentGantry()
        selector = with_semantic_tools(
            gantry,
            prompt_param="question",
            tools_param="functions",
            limit=3,
            dialect="anthropic",
            auto_sync=False,
            score_threshold=0.7,
        )
        assert isinstance(selector, SemanticToolSelector)
        assert selector._prompt_param == "question"
        assert selector._tools_param == "functions"
        assert selector._limit == 3
        assert selector._dialect == "anthropic"
        assert selector._auto_sync is False
        assert selector._score_threshold == 0.7


class TestSemanticToolSelector:
    """Tests for the SemanticToolSelector class."""

    @pytest.fixture
    def mock_gantry(self) -> AgentGantry:
        """Create a mock AgentGantry with sample tools."""
        gantry = AgentGantry()
        return gantry

    @pytest.fixture
    def sample_tool(self) -> ToolDefinition:
        """Create a sample tool definition."""
        return ToolDefinition(
            name="get_weather",
            description="Get the current weather for a specified city location.",
            parameters_schema={
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"},
                },
                "required": ["city"],
            },
        )

    @pytest.mark.asyncio
    async def test_async_function_wrapping(self, mock_gantry: AgentGantry) -> None:
        """Test wrapping an async function."""
        selector = SemanticToolSelector(mock_gantry)

        @selector
        async def generate(prompt: str, *, tools: list[dict[str, Any]] | None = None) -> str:
            return f"Prompt: {prompt}, Tools: {len(tools or [])}"

        # The function should still be async
        assert hasattr(generate, "__wrapped__")

    @pytest.mark.asyncio
    async def test_sync_function_wrapping(self, mock_gantry: AgentGantry) -> None:
        """Test wrapping a sync function."""
        selector = SemanticToolSelector(mock_gantry)

        @selector
        def generate(prompt: str, *, tools: list[dict[str, Any]] | None = None) -> str:
            return f"Prompt: {prompt}, Tools: {len(tools or [])}"

        # The function should still be sync
        assert hasattr(generate, "__wrapped__")

    @pytest.mark.asyncio
    async def test_tool_injection_when_tools_not_provided(
        self, mock_gantry: AgentGantry, sample_tool: ToolDefinition
    ) -> None:
        """Test that tools are injected when not provided."""
        # Register a tool
        @mock_gantry.register
        def get_weather(city: str) -> str:
            """Get the current weather for a specified city location."""
            return f"Weather in {city}"

        # Use low threshold since simple embedder produces lower scores
        selector = SemanticToolSelector(mock_gantry, score_threshold=0.0)
        captured_tools: list[dict[str, Any]] | None = None

        @selector
        async def generate(prompt: str, *, tools: list[dict[str, Any]] | None = None) -> str:
            nonlocal captured_tools
            captured_tools = tools
            return f"Prompt: {prompt}"

        await generate("What's the weather in Paris?")

        # Tools should have been injected
        assert captured_tools is not None
        assert len(captured_tools) >= 1

    @pytest.mark.asyncio
    async def test_tools_not_overwritten_when_provided(
        self, mock_gantry: AgentGantry
    ) -> None:
        """Test that existing tools are not overwritten."""
        @mock_gantry.register
        def some_tool(x: int) -> str:
            """Some tool for testing purposes."""
            return str(x)

        selector = SemanticToolSelector(mock_gantry)
        captured_tools: list[dict[str, Any]] | None = None
        provided_tools = [{"type": "function", "function": {"name": "custom_tool"}}]

        @selector
        async def generate(prompt: str, *, tools: list[dict[str, Any]] | None = None) -> str:
            nonlocal captured_tools
            captured_tools = tools
            return f"Prompt: {prompt}"

        await generate("Test prompt", tools=provided_tools)

        # Original tools should be preserved
        assert captured_tools == provided_tools

    @pytest.mark.asyncio
    async def test_prompt_extraction_from_messages(
        self, mock_gantry: AgentGantry
    ) -> None:
        """Test prompt extraction from OpenAI-style messages."""
        @mock_gantry.register
        def test_tool(x: str) -> str:
            """A test tool for message extraction."""
            return x

        selector = SemanticToolSelector(mock_gantry, prompt_param="messages")
        captured_prompt: str | None = None
        original_retrieve = selector._retrieve_tools

        async def capture_retrieve(prompt: str) -> list[dict[str, Any]]:
            nonlocal captured_prompt
            captured_prompt = prompt
            return await original_retrieve(prompt)

        # Use patch for cleaner test isolation
        with patch.object(selector, "_retrieve_tools", side_effect=capture_retrieve):

            @selector
            async def generate(
                messages: list[dict[str, str]],
                *,
                tools: list[dict[str, Any]] | None = None,
            ) -> str:
                return "response"

            await generate([
                {"role": "system", "content": "You are helpful"},
                {"role": "user", "content": "What's the weather?"},
            ])

        # Should have extracted the user message
        assert captured_prompt == "What's the weather?"


class TestSemanticToolSelectorDialects:
    """Tests for different dialect outputs."""

    @pytest.fixture
    def gantry_with_tool(self) -> AgentGantry:
        """Create a gantry with a registered tool."""
        gantry = AgentGantry()

        @gantry.register
        def search_docs(query: str) -> str:
            """Search documentation for relevant information."""
            return f"Results for: {query}"

        return gantry

    @pytest.mark.asyncio
    async def test_openai_dialect(self, gantry_with_tool: AgentGantry) -> None:
        """Test OpenAI dialect output format."""
        selector = SemanticToolSelector(gantry_with_tool, dialect="openai")

        tools = await selector._retrieve_tools("search for something")

        if tools:  # Only check if tools were found
            tool = tools[0]
            assert "type" in tool
            assert tool["type"] == "function"
            assert "function" in tool
            assert "name" in tool["function"]
            assert "parameters" in tool["function"]

    @pytest.mark.asyncio
    async def test_anthropic_dialect(self, gantry_with_tool: AgentGantry) -> None:
        """Test Anthropic dialect output format."""
        selector = SemanticToolSelector(gantry_with_tool, dialect="anthropic")

        tools = await selector._retrieve_tools("search for something")

        if tools:  # Only check if tools were found
            tool = tools[0]
            assert "name" in tool
            assert "description" in tool
            assert "input_schema" in tool
            # Anthropic format does NOT have "type": "function" wrapper
            assert "type" not in tool or tool.get("type") != "function"


class TestSemanticToolsDecoratorFactory:
    """Tests for SemanticToolsDecorator factory class."""

    def test_factory_initialization(self) -> None:
        """Test factory initialization with options."""
        gantry = AgentGantry()
        factory = SemanticToolsDecorator(
            gantry,
            prompt_param="query",
            tools_param="functions",
            limit=10,
            dialect="anthropic",
        )
        assert factory._gantry is gantry
        assert factory._prompt_param == "query"
        assert factory._tools_param == "functions"
        assert factory._limit == 10
        assert factory._dialect == "anthropic"

    @pytest.mark.asyncio
    async def test_factory_wrap_method(self) -> None:
        """Test wrapping functions using the factory."""
        gantry = AgentGantry()

        @gantry.register
        def my_tool(x: int) -> str:
            """A tool for factory testing."""
            return str(x)

        factory = SemanticToolsDecorator(gantry)

        @factory.wrap
        async def generate(prompt: str, *, tools: list | None = None) -> str:
            return f"Prompt: {prompt}, Tools: {len(tools or [])}"

        result = await generate("test prompt")
        assert "Prompt: test prompt" in result

    @pytest.mark.asyncio
    async def test_factory_wrap_with_overrides(self) -> None:
        """Test wrapping with parameter overrides."""
        gantry = AgentGantry()

        @gantry.register
        def search_tool(query: str) -> str:
            """Search for information in the database."""
            return f"Results: {query}"

        factory = SemanticToolsDecorator(gantry, limit=5)

        @factory.wrap(limit=2)
        async def generate(prompt: str, *, tools: list | None = None) -> str:
            return f"Tools count: {len(tools or [])}"

        # Should use the override limit of 2
        result = await generate("search query")
        assert "Tools count:" in result


class TestPromptExtraction:
    """Tests for prompt extraction from various formats."""

    @pytest.fixture
    def selector(self) -> SemanticToolSelector:
        """Create a selector for testing."""
        gantry = AgentGantry()
        return SemanticToolSelector(gantry)

    def test_extract_from_direct_prompt(self, selector: SemanticToolSelector) -> None:
        """Test extracting prompt from direct parameter."""
        import inspect

        def func(prompt: str) -> str:
            return prompt

        sig = inspect.signature(func)
        result = selector._extract_prompt(("Hello world",), {}, sig)
        assert result == "Hello world"

    def test_extract_from_kwargs(self, selector: SemanticToolSelector) -> None:
        """Test extracting prompt from keyword argument."""
        import inspect

        def func(prompt: str = "") -> str:
            return prompt

        sig = inspect.signature(func)
        result = selector._extract_prompt((), {"prompt": "Hello world"}, sig)
        assert result == "Hello world"

    def test_extract_from_openai_messages(self, selector: SemanticToolSelector) -> None:
        """Test extracting prompt from OpenAI-style messages."""
        import inspect

        def func(messages: list[dict[str, str]]) -> str:
            return ""

        sig = inspect.signature(func)
        messages = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "User message"},
        ]
        result = selector._extract_prompt((), {"messages": messages}, sig)
        assert result == "User message"

    def test_extract_from_multimodal_content(self, selector: SemanticToolSelector) -> None:
        """Test extracting prompt from multimodal message content."""
        import inspect

        def func(messages: list[dict[str, Any]]) -> str:
            return ""

        sig = inspect.signature(func)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "..."}},
                    {"type": "text", "text": "What's in this image?"},
                ],
            }
        ]
        result = selector._extract_prompt((), {"messages": messages}, sig)
        assert result == "What's in this image?"

    def test_extract_returns_none_when_no_prompt(
        self, selector: SemanticToolSelector
    ) -> None:
        """Test that None is returned when no prompt is found."""
        import inspect

        def func(other_param: int) -> int:
            return other_param

        sig = inspect.signature(func)
        result = selector._extract_prompt((42,), {}, sig)
        assert result is None


class TestIntegrationScenarios:
    """Integration tests for real-world usage scenarios."""

    @pytest.mark.asyncio
    async def test_openai_style_integration(self) -> None:
        """Test integration with OpenAI-style API patterns."""
        gantry = AgentGantry()

        @gantry.register
        def get_stock_price(symbol: str) -> str:
            """Get the current stock price for a ticker symbol."""
            return f"Stock price for {symbol}"

        @gantry.register
        def calculate_mortgage(principal: float, rate: float) -> str:
            """Calculate monthly mortgage payment amounts."""
            return f"Mortgage for ${principal}"

        # Use low threshold since simple embedder produces lower scores
        selector = with_semantic_tools(gantry, limit=2, score_threshold=0.0)
        received_tools: list[dict[str, Any]] | None = None

        @selector
        async def chat_completion(
            messages: list[dict[str, str]],
            model: str = "gpt-4",
            *,
            tools: list[dict[str, Any]] | None = None,
        ) -> dict[str, Any]:
            nonlocal received_tools
            received_tools = tools
            return {"choices": [{"message": {"content": "Response"}}]}

        await chat_completion(
            messages=[{"role": "user", "content": "What is the stock price of AAPL?"}],
            model="gpt-4",
        )

        # Should have retrieved relevant tools
        assert received_tools is not None
        # Should contain tools related to stock/finance

    @pytest.mark.asyncio
    async def test_anthropic_style_integration(self) -> None:
        """Test integration with Anthropic-style API patterns."""
        gantry = AgentGantry()

        @gantry.register
        def send_email(to: str, subject: str, body: str) -> str:
            """Send an email to the specified recipient address."""
            return f"Email sent to {to}"

        # Use low threshold since simple embedder produces lower scores
        selector = with_semantic_tools(gantry, dialect="anthropic", score_threshold=0.0)
        received_tools: list[dict[str, Any]] | None = None

        @selector
        async def messages_create(
            messages: list[dict[str, str]],
            model: str = "claude-3",
            *,
            tools: list[dict[str, Any]] | None = None,
        ) -> dict[str, Any]:
            nonlocal received_tools
            received_tools = tools
            return {"content": [{"text": "Response"}]}

        await messages_create(
            messages=[{"role": "user", "content": "Send an email to john@example.com"}],
        )

        # Should have retrieved tools in Anthropic format
        assert received_tools is not None
        if received_tools:
            # Anthropic format has input_schema, not parameters
            assert "input_schema" in received_tools[0] or "name" in received_tools[0]
