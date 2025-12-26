"""
Tests for new convenience methods and improvements added in the refactoring PR.

Tests:
- AgentGantry.quick_start()
- AgentGantry.search_and_execute()
- set_default_gantry() and decorator improvements
- build_parameters_schema() improvements
- ToolDefinition.to_searchable_text()
"""

import pytest

from agent_gantry import AgentGantry, set_default_gantry, with_semantic_tools
from agent_gantry.schema.introspection import build_parameters_schema
from agent_gantry.schema.tool import ToolDefinition


class TestQuickStart:
    """Tests for AgentGantry.quick_start() convenience method."""

    @pytest.mark.asyncio
    async def test_quick_start_auto_embedder(self):
        """Test quick_start with auto embedder selection."""
        gantry = await AgentGantry.quick_start(embedder="auto")
        assert gantry is not None
        assert gantry._embedder is not None

    @pytest.mark.asyncio
    async def test_quick_start_simple_embedder(self):
        """Test quick_start with explicit simple embedder."""
        gantry = await AgentGantry.quick_start(embedder="simple")
        assert gantry is not None
        from agent_gantry.adapters.embedders.simple import SimpleEmbedder

        assert isinstance(gantry._embedder, SimpleEmbedder)

    @pytest.mark.asyncio
    async def test_quick_start_openai_without_key(self):
        """Test quick_start raises error for OpenAI without API key."""
        with pytest.raises(ValueError, match="OpenAI embedder requires a valid API key"):
            await AgentGantry.quick_start(embedder="openai")

    @pytest.mark.asyncio
    async def test_quick_start_with_tool_registration(self):
        """Test quick_start works with tool registration and sync."""
        gantry = await AgentGantry.quick_start()

        @gantry.register
        def test_tool(x: int) -> int:
            """Test tool."""
            return x * 2

        await gantry.sync()
        tools = await gantry.list_tools()
        assert len(tools) == 1
        assert tools[0].name == "test_tool"


class TestSearchAndExecute:
    """Tests for AgentGantry.search_and_execute() convenience method."""

    @pytest.mark.asyncio
    async def test_search_and_execute_basic(self):
        """Test basic search and execute functionality."""
        gantry = await AgentGantry.quick_start()

        @gantry.register
        def calculate_tax(amount: float) -> float:
            """Calculate 8% sales tax."""
            return amount * 0.08

        await gantry.sync()

        result = await gantry.search_and_execute(
            "calculate tax", arguments={"amount": 100.0}, score_threshold=0.0
        )

        assert result.result == 8.0
        assert result.status.value == "success"

    @pytest.mark.asyncio
    async def test_search_and_execute_no_tools_found(self):
        """Test search_and_execute raises error when no tools match."""
        gantry = await AgentGantry.quick_start()

        @gantry.register
        def unrelated_tool(x: int) -> int:
            """Some unrelated tool."""
            return x

        await gantry.sync()

        with pytest.raises(ValueError, match="No tools found matching query"):
            await gantry.search_and_execute(
                "calculate quantum mechanics", arguments={}, score_threshold=0.9
            )

    @pytest.mark.asyncio
    async def test_search_and_execute_with_namespace(self):
        """Test search_and_execute respects tool namespace."""
        gantry = await AgentGantry.quick_start()

        @gantry.register(namespace="default")
        def multiply(x: int, y: int) -> int:
            """Multiply two numbers."""
            return x * y

        await gantry.sync()

        result = await gantry.search_and_execute(
            "multiply numbers", arguments={"x": 5, "y": 3}, score_threshold=0.0
        )

        assert result.result == 15


class TestDefaultGantryDecorator:
    """Tests for set_default_gantry() and improved decorator."""

    @pytest.mark.asyncio
    async def test_set_default_gantry(self):
        """Test setting and using default gantry."""
        gantry = await AgentGantry.quick_start()

        @gantry.register
        def get_weather(city: str) -> str:
            """Get weather for a city."""
            return f"Sunny in {city}"

        await gantry.sync()

        # Set as default
        set_default_gantry(gantry)

        # Use decorator without explicit gantry
        @with_semantic_tools(limit=1, score_threshold=0.0)
        async def mock_llm_call(prompt: str, tools=None):
            return {"tools_count": len(tools) if tools else 0}

        result = await mock_llm_call("What's the weather?")
        assert result["tools_count"] == 1

    @pytest.mark.asyncio
    async def test_decorator_without_default_gantry_raises(self):
        """Test decorator without default gantry raises helpful error."""
        # Reset default
        from agent_gantry.integrations import decorator

        decorator._DEFAULT_GANTRY = None

        with pytest.raises(ValueError, match="No gantry provided and no default set"):

            @with_semantic_tools(limit=1)
            async def generate(prompt: str, tools=None):
                pass

    @pytest.mark.asyncio
    async def test_decorator_with_explicit_gantry_still_works(self):
        """Test decorator still works with explicit gantry parameter."""
        gantry = await AgentGantry.quick_start()

        @gantry.register
        def test_tool() -> str:
            """Test tool."""
            return "result"

        await gantry.sync()

        @with_semantic_tools(gantry, limit=1, score_threshold=0.0)
        async def generate(prompt: str, tools=None):
            return {"tools_count": len(tools) if tools else 0}

        result = await generate("test")
        assert result["tools_count"] == 1


class TestBuildParametersSchema:
    """Tests for build_parameters_schema improvements."""

    def test_basic_type_mapping(self):
        """Test basic type mapping for int, float, bool, str."""

        def func(a: int, b: float, c: bool, d: str) -> None:
            pass

        schema = build_parameters_schema(func)
        assert schema["properties"]["a"]["type"] == "integer"
        assert schema["properties"]["b"]["type"] == "number"
        assert schema["properties"]["c"]["type"] == "boolean"
        assert schema["properties"]["d"]["type"] == "string"

    def test_required_vs_optional(self):
        """Test detection of required vs optional parameters."""

        def func(required: int, optional: str = "default") -> None:
            pass

        schema = build_parameters_schema(func)
        assert "required" in schema["required"]
        assert "optional" not in schema["required"]

    def test_optional_type_handling(self):
        """Test handling of Optional[T] type hints."""

        def func(x: int | None = None, y: str | None = None) -> None:
            pass

        schema = build_parameters_schema(func)
        assert schema["properties"]["x"]["type"] == "integer"
        assert schema["properties"]["y"]["type"] == "string"

    def test_skips_self_and_cls(self):
        """Test that self and cls parameters are skipped."""

        class TestClass:
            def method(self, x: int) -> None:
                pass

            @classmethod
            def classmethod(cls, x: int) -> None:
                pass

        schema1 = build_parameters_schema(TestClass.method)
        assert "self" not in schema1["properties"]
        assert "x" in schema1["properties"]

        schema2 = build_parameters_schema(TestClass.classmethod)
        assert "cls" not in schema2["properties"]
        assert "x" in schema2["properties"]

    def test_no_type_hints(self):
        """Test function without type hints defaults to string."""

        def func(x, y):
            pass

        schema = build_parameters_schema(func)
        assert schema["properties"]["x"]["type"] == "string"
        assert schema["properties"]["y"]["type"] == "string"


class TestToolSearchableText:
    """Tests for ToolDefinition.to_searchable_text()."""

    def test_includes_all_metadata(self):
        """Test that searchable text includes all relevant metadata."""
        tool = ToolDefinition(
            name="calculate_tax",
            namespace="finance",
            description="Calculate sales tax for a given amount",
            tags=["math", "money", "finance"],
            examples=["calculate tax on $100", "what is 8% tax on 50"],
            parameters_schema={"type": "object", "properties": {}, "required": []},
        )

        text = tool.to_searchable_text()

        # Verify all components are present
        assert "calculate_tax" in text
        assert "finance" in text
        assert "Calculate sales tax" in text
        assert "math" in text
        assert "money" in text
        assert "calculate tax on $100" in text

    def test_empty_tags_and_examples(self):
        """Test searchable text with empty tags and examples."""
        tool = ToolDefinition(
            name="simple_tool",
            namespace="default",
            description="A simple tool",
            parameters_schema={"type": "object", "properties": {}, "required": []},
        )

        text = tool.to_searchable_text()

        assert "simple_tool" in text
        assert "default" in text
        assert "A simple tool" in text

    def test_consistency_with_router(self):
        """Test that searchable text is consistent across uses."""
        tool = ToolDefinition(
            name="test_tool",
            namespace="test",
            description="Test description",
            tags=["tag1", "tag2"],
            examples=["example1"],
            parameters_schema={"type": "object", "properties": {}, "required": []},
        )

        # Call multiple times
        text1 = tool.to_searchable_text()
        text2 = tool.to_searchable_text()

        # Should be identical
        assert text1 == text2
