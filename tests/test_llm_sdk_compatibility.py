"""
Tests for LLM SDK compatibility (Late 2025).

These tests verify that Agent-Gantry is compatible with major Python LLM SDKs.
Tests cover SDK imports, client initialization patterns, and key endpoint methods.
"""

from __future__ import annotations

import pytest


class TestOpenAICompatibility:
    """Tests for OpenAI SDK compatibility."""

    def test_openai_import(self) -> None:
        """Test that OpenAI SDK can be imported."""
        pytest.importorskip("openai")
        from openai import OpenAI

        assert OpenAI is not None

    def test_azure_openai_import(self) -> None:
        """Test that AzureOpenAI can be imported."""
        pytest.importorskip("openai")
        from openai import AzureOpenAI

        assert AzureOpenAI is not None

    def test_openai_client_initialization(self) -> None:
        """Test OpenAI client initialization pattern."""
        pytest.importorskip("openai")
        from openai import OpenAI

        # Test that client can be instantiated with api_key
        client = OpenAI(api_key="test-key")
        assert client is not None
        assert hasattr(client, "chat")
        assert hasattr(client.chat, "completions")

    def test_azure_openai_client_initialization(self) -> None:
        """Test AzureOpenAI client initialization pattern."""
        pytest.importorskip("openai")
        from openai import AzureOpenAI

        # Test that Azure client can be instantiated
        client = AzureOpenAI(
            api_key="test-key",
            api_version="2024-10-21",
            azure_endpoint="https://test.openai.azure.com",
        )
        assert client is not None
        assert hasattr(client, "chat")
        assert hasattr(client.chat, "completions")

    def test_openai_has_required_endpoints(self) -> None:
        """Test OpenAI client has required endpoint methods."""
        pytest.importorskip("openai")
        from openai import OpenAI

        client = OpenAI(api_key="test-key")

        # Chat completions
        assert hasattr(client.chat.completions, "create")

        # Audio transcriptions
        assert hasattr(client, "audio")
        assert hasattr(client.audio, "transcriptions")

    def test_openai_compatible_base_url_override(self) -> None:
        """Test OpenAI client with custom base_url for OpenRouter etc."""
        pytest.importorskip("openai")
        from openai import OpenAI

        # OpenRouter
        client = OpenAI(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
        )
        assert client is not None
        assert hasattr(client.chat.completions, "create")


class TestAnthropicCompatibility:
    """Tests for Anthropic SDK compatibility."""

    def test_anthropic_import(self) -> None:
        """Test that Anthropic SDK can be imported."""
        pytest.importorskip("anthropic")
        from anthropic import Anthropic

        assert Anthropic is not None

    def test_anthropic_client_initialization(self) -> None:
        """Test Anthropic client initialization pattern."""
        pytest.importorskip("anthropic")
        from anthropic import Anthropic

        client = Anthropic(api_key="test-key")
        assert client is not None
        assert hasattr(client, "messages")

    def test_anthropic_has_required_endpoints(self) -> None:
        """Test Anthropic client has required endpoint methods."""
        pytest.importorskip("anthropic")
        from anthropic import Anthropic

        client = Anthropic(api_key="test-key")

        # Messages
        assert hasattr(client.messages, "create")

        # Beta features
        assert hasattr(client, "beta")


class TestGoogleGenAICompatibility:
    """Tests for Google GenAI SDK compatibility."""

    def test_google_genai_import(self) -> None:
        """Test that Google GenAI SDK can be imported."""
        pytest.importorskip("google.genai")
        from google import genai

        assert genai is not None

    def test_google_genai_client_initialization(self) -> None:
        """Test Google GenAI client initialization pattern."""
        pytest.importorskip("google.genai")
        from google import genai

        # Client requires valid API key for initialization
        client = genai.Client(api_key="test-key")
        assert client is not None
        assert hasattr(client, "models")

    def test_google_genai_has_required_methods(self) -> None:
        """Test Google GenAI client has required methods."""
        pytest.importorskip("google.genai")
        from google import genai

        client = genai.Client(api_key="test-key")

        # Models endpoint
        assert hasattr(client.models, "generate_content")
        assert hasattr(client.models, "generate_content_stream")


class TestGoogleVertexAICompatibility:
    """Tests for Google Vertex AI SDK compatibility."""

    def test_vertexai_import(self) -> None:
        """Test that Vertex AI SDK can be imported."""
        pytest.importorskip("vertexai")
        import vertexai

        assert vertexai is not None
        assert hasattr(vertexai, "init")

    def test_generative_model_import(self) -> None:
        """Test that GenerativeModel can be imported."""
        pytest.importorskip("vertexai")
        from vertexai.generative_models import GenerativeModel

        assert GenerativeModel is not None

    def test_vertexai_function_declaration_import(self) -> None:
        """Test that FunctionDeclaration can be imported for tool use."""
        pytest.importorskip("vertexai")
        from vertexai.generative_models import FunctionDeclaration, Tool

        assert FunctionDeclaration is not None
        assert Tool is not None


class TestMistralCompatibility:
    """Tests for Mistral SDK compatibility."""

    def test_mistral_import(self) -> None:
        """Test that Mistral SDK can be imported."""
        pytest.importorskip("mistralai")
        from mistralai import Mistral

        assert Mistral is not None

    def test_mistral_client_initialization(self) -> None:
        """Test Mistral client initialization pattern."""
        pytest.importorskip("mistralai")
        from mistralai import Mistral

        client = Mistral(api_key="test-key")
        assert client is not None
        assert hasattr(client, "chat")

    def test_mistral_has_required_endpoints(self) -> None:
        """Test Mistral client has required endpoint methods."""
        pytest.importorskip("mistralai")
        from mistralai import Mistral

        client = Mistral(api_key="test-key")

        # Chat complete
        assert hasattr(client.chat, "complete")

        # FIM (Fill-in-the-Middle) for code completion
        assert hasattr(client, "fim")
        assert hasattr(client.fim, "complete")

        # Agents
        assert hasattr(client, "agents")
        assert hasattr(client.agents, "complete")


class TestGroqCompatibility:
    """Tests for Groq SDK compatibility."""

    def test_groq_import(self) -> None:
        """Test that Groq SDK can be imported."""
        pytest.importorskip("groq")
        from groq import Groq

        assert Groq is not None

    def test_groq_client_initialization(self) -> None:
        """Test Groq client initialization pattern."""
        pytest.importorskip("groq")
        from groq import Groq

        client = Groq(api_key="test-key")
        assert client is not None
        assert hasattr(client, "chat")
        assert hasattr(client.chat, "completions")

    def test_groq_has_required_endpoints(self) -> None:
        """Test Groq client has required endpoint methods."""
        pytest.importorskip("groq")
        from groq import Groq

        client = Groq(api_key="test-key")

        # Chat completions (OpenAI-compatible)
        assert hasattr(client.chat.completions, "create")


class TestAgentGantryToolFormatCompatibility:
    """Tests for Agent-Gantry tool format compatibility with LLM providers."""

    @pytest.mark.asyncio
    async def test_openai_tool_format(self) -> None:
        """Test that Agent-Gantry produces valid OpenAI tool format."""
        from agent_gantry import AgentGantry

        gantry = AgentGantry()

        @gantry.register
        def test_tool(x: int, y: str) -> str:
            """A test tool for format validation."""
            return f"{x}: {y}"

        await gantry.sync()

        tools = await gantry.retrieve_tools("test tool", limit=1)

        assert len(tools) >= 1
        tool = tools[0]

        # Verify OpenAI tool format
        assert "type" in tool
        assert tool["type"] == "function"
        assert "function" in tool
        assert "name" in tool["function"]
        assert "description" in tool["function"]
        assert "parameters" in tool["function"]

        # Verify parameters schema
        params = tool["function"]["parameters"]
        assert params["type"] == "object"
        assert "properties" in params
        assert "x" in params["properties"]
        assert "y" in params["properties"]

    @pytest.mark.asyncio
    async def test_anthropic_tool_format_conversion(self) -> None:
        """Test converting Agent-Gantry tools to Anthropic format."""
        from agent_gantry import AgentGantry

        gantry = AgentGantry()

        @gantry.register
        def analyze_text(text: str) -> str:
            """Analyze the given text."""
            return f"Analysis of: {text}"

        await gantry.sync()

        openai_tools = await gantry.retrieve_tools("analyze text", limit=1)

        # Convert to Anthropic format
        anthropic_tools = [
            {
                "name": tool["function"]["name"],
                "description": tool["function"]["description"],
                "input_schema": tool["function"]["parameters"],
            }
            for tool in openai_tools
        ]

        assert len(anthropic_tools) >= 1
        tool = anthropic_tools[0]

        # Verify Anthropic tool format
        assert "name" in tool
        assert "description" in tool
        assert "input_schema" in tool
        assert tool["input_schema"]["type"] == "object"

    @pytest.mark.asyncio
    async def test_tool_execution_returns_expected_result(self) -> None:
        """Test that tool execution works correctly."""
        from agent_gantry import AgentGantry
        from agent_gantry.schema.execution import ExecutionStatus, ToolCall

        gantry = AgentGantry()

        @gantry.register
        def add_numbers(a: int, b: int) -> int:
            """Add two numbers together."""
            return a + b

        await gantry.sync()

        result = await gantry.execute(
            ToolCall(tool_name="add_numbers", arguments={"a": 5, "b": 3})
        )

        assert result.status == ExecutionStatus.SUCCESS
        assert result.result == 8


class TestSDKVersionCompatibility:
    """Tests for SDK version compatibility."""

    def test_openai_minimum_version(self) -> None:
        """Test OpenAI SDK meets minimum version."""
        openai = pytest.importorskip("openai")
        version = openai.__version__
        parts = version.split(".")
        major = int(parts[0])
        assert major >= 1, f"OpenAI SDK version {version} is below minimum 1.0.0"

    def test_anthropic_minimum_version(self) -> None:
        """Test Anthropic SDK meets minimum version."""
        anthropic = pytest.importorskip("anthropic")
        version = anthropic.__version__
        parts = version.split(".")
        major = int(parts[0])
        minor = int(parts[1])
        # Version 0.40.0 or higher
        assert (
            major > 0 or minor >= 40
        ), f"Anthropic SDK version {version} is below minimum 0.40.0"

    def test_groq_minimum_version(self) -> None:
        """Test Groq SDK meets minimum version."""
        groq = pytest.importorskip("groq")
        version = groq.__version__
        parts = version.split(".")
        major = int(parts[0])
        minor = int(parts[1])
        # Version 0.13.0 or higher
        assert (
            major > 0 or minor >= 13
        ), f"Groq SDK version {version} is below minimum 0.13.0"

    def test_mistral_minimum_version(self) -> None:
        """Test Mistral SDK meets minimum version."""
        mistralai = pytest.importorskip("mistralai")
        version = getattr(mistralai, "__version__", "1.0.0")
        parts = version.split(".")
        major = int(parts[0])
        assert major >= 1, f"Mistral SDK version {version} is below minimum 1.0.0"
