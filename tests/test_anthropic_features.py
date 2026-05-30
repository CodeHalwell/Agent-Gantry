"""
Tests for Anthropic-specific features.

Tests the Anthropic integration including:
- Interleaved thinking
- Extended thinking
- Tool use with Agent-Gantry
"""

from __future__ import annotations

import sys
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest

from agent_gantry import AgentGantry
from agent_gantry.integrations.anthropic_features import AnthropicFeatures

# Create mock anthropic module
mock_anthropic = Mock()
mock_anthropic.AsyncAnthropic = MagicMock
sys.modules["anthropic"] = mock_anthropic

# Now we can import regardless of whether anthropic is installed
from agent_gantry.integrations.anthropic_features import (
    AnthropicClient,
    create_anthropic_client,
)


@pytest.fixture
def mock_anthropic_response():
    """Create a mock Anthropic response."""
    response = MagicMock()
    response.content = [
        MagicMock(type="text", text="This is the answer"),
    ]
    return response


@pytest.fixture
def mock_anthropic_thinking_response():
    """Create a mock Anthropic response with thinking."""
    response = MagicMock()
    thinking_block = MagicMock()
    thinking_block.type = "thinking"
    thinking_block.thinking = "Let me think about this..."

    text_block = MagicMock()
    text_block.type = "text"
    text_block.text = "Here's my answer"

    response.content = [thinking_block, text_block]
    return response


@pytest.fixture
def mock_anthropic_tool_response():
    """Create a mock Anthropic response with tool use."""
    response = MagicMock()
    tool_block = MagicMock()
    tool_block.type = "tool_use"
    tool_block.id = "tool_123"
    tool_block.name = "test_tool"
    tool_block.input = {"arg": "value"}

    response.content = [tool_block]
    return response


class TestAnthropicFeatures:
    """Tests for AnthropicFeatures configuration."""

    def test_default_features(self):
        """Test default feature configuration."""
        features = AnthropicFeatures()
        assert features.enable_interleaved_thinking is False
        assert features.enable_extended_thinking is False
        assert features.thinking_budget_tokens is None
        assert features.thinking_display is None

    def test_interleaved_thinking_config(self):
        """Test interleaved thinking configuration."""
        features = AnthropicFeatures(enable_interleaved_thinking=True)
        assert features.enable_interleaved_thinking is True
        assert features.enable_extended_thinking is False

    def test_extended_thinking_config(self):
        """Test extended thinking configuration."""
        features = AnthropicFeatures(
            enable_extended_thinking=True,
            thinking_budget_tokens=5000,
        )
        assert features.enable_extended_thinking is True
        assert features.thinking_budget_tokens == 5000

    def test_thinking_display_config(self):
        """Test thinking_display field is stored correctly."""
        features = AnthropicFeatures(
            adaptive_thinking_effort="medium",
            thinking_display="omitted",
        )
        assert features.thinking_display == "omitted"
        assert features.adaptive_thinking_effort == "medium"


class TestAnthropicClient:
    """Tests for AnthropicClient."""

    def test_client_initialization(self):
        """Test client initialization."""
        client = AnthropicClient(api_key="test-key")
        assert client is not None

    def test_client_with_interleaved_thinking(self):
        """Test client initialization with interleaved thinking."""
        features = AnthropicFeatures(enable_interleaved_thinking=True)
        client = AnthropicClient(api_key="test-key", features=features)
        assert client._features.enable_interleaved_thinking is True

    def test_client_with_extended_thinking(self):
        """Test client initialization with extended thinking."""
        features = AnthropicFeatures(enable_extended_thinking=True)
        client = AnthropicClient(api_key="test-key", features=features)
        assert client._features.enable_extended_thinking is True

    def test_extract_thinking(self, mock_anthropic_thinking_response):
        """Test extracting thinking blocks from response."""
        client = AnthropicClient(api_key="test-key")
        thinking = client.extract_thinking(mock_anthropic_thinking_response)

        assert len(thinking) == 1
        assert thinking[0] == "Let me think about this..."

    @pytest.mark.asyncio
    async def test_execute_tool_calls(self, mock_anthropic_tool_response):
        """Test executing tool calls from response."""
        # Create mock gantry
        gantry = MagicMock(spec=AgentGantry)
        mock_result = MagicMock()
        mock_result.status = "success"
        mock_result.result = "Tool executed successfully"
        gantry.execute = AsyncMock(return_value=mock_result)

        client = AnthropicClient(api_key="test-key", gantry=gantry)
        tool_results = await client.execute_tool_calls(mock_anthropic_tool_response)

        # Verify tool was executed
        gantry.execute.assert_called_once()
        assert len(tool_results) == 1
        assert tool_results[0]["type"] == "tool_result"
        assert tool_results[0]["tool_use_id"] == "tool_123"
        assert "Tool executed successfully" in tool_results[0]["content"]

    def test_features_configuration(self):
        """Test that features are properly stored."""
        features = AnthropicFeatures(
            enable_interleaved_thinking=True,
            thinking_budget_tokens=5000,
        )
        client = AnthropicClient(api_key="test-key", features=features)
        assert client._features == features

    @pytest.mark.asyncio
    async def test_thinking_display_adaptive(self):
        """Test that thinking_display is included in adaptive thinking payload."""
        features = AnthropicFeatures(
            adaptive_thinking_effort="high",
            thinking_display="omitted",
        )
        client = AnthropicClient(api_key="test-key", features=features)

        # Mock the underlying create call
        mock_response = MagicMock()
        mock_response.content = []
        client._client.messages.create = AsyncMock(return_value=mock_response)

        await client.create_message(
            model="claude-sonnet-4-6",
            messages=[{"role": "user", "content": "Hello"}],
            auto_retrieve_tools=False,
        )

        call_kwargs = client._client.messages.create.call_args.kwargs
        assert "thinking" in call_kwargs
        assert call_kwargs["thinking"]["type"] == "adaptive"
        assert call_kwargs["thinking"]["effort"] == "high"
        assert call_kwargs["thinking"]["display"] == "omitted"

    @pytest.mark.asyncio
    async def test_thinking_display_extended(self):
        """Test that thinking_display is included in extended thinking payload."""
        features = AnthropicFeatures(
            enable_extended_thinking=True,
            thinking_budget_tokens=8000,
            thinking_display="summarized",
        )
        client = AnthropicClient(api_key="test-key", features=features)

        mock_response = MagicMock()
        mock_response.content = []
        client._client.messages.create = AsyncMock(return_value=mock_response)

        await client.create_message(
            model="claude-sonnet-4-6",
            messages=[{"role": "user", "content": "Hello"}],
            auto_retrieve_tools=False,
        )

        call_kwargs = client._client.messages.create.call_args.kwargs
        assert "thinking" in call_kwargs
        assert call_kwargs["thinking"]["type"] == "enabled"
        assert call_kwargs["thinking"]["budget_tokens"] == 8000
        assert call_kwargs["thinking"]["display"] == "summarized"

    @pytest.mark.asyncio
    async def test_thinking_display_absent_when_none(self):
        """Test that display key is absent when thinking_display is None."""
        features = AnthropicFeatures(
            adaptive_thinking_effort="medium",
            thinking_display=None,
        )
        client = AnthropicClient(api_key="test-key", features=features)

        mock_response = MagicMock()
        mock_response.content = []
        client._client.messages.create = AsyncMock(return_value=mock_response)

        await client.create_message(
            model="claude-sonnet-4-6",
            messages=[{"role": "user", "content": "Hello"}],
            auto_retrieve_tools=False,
        )

        call_kwargs = client._client.messages.create.call_args.kwargs
        assert "thinking" in call_kwargs
        assert "display" not in call_kwargs["thinking"]

    @pytest.mark.asyncio
    async def test_execute_tool_calls_failure_sets_is_error(self, mock_anthropic_tool_response):
        """Test that a failed tool execution sets is_error=True on the result."""
        gantry = MagicMock(spec=AgentGantry)
        mock_result = MagicMock()
        mock_result.status = "failure"
        mock_result.error = "Tool crashed"
        gantry.execute = AsyncMock(return_value=mock_result)

        client = AnthropicClient(api_key="test-key", gantry=gantry)
        tool_results = await client.execute_tool_calls(mock_anthropic_tool_response)

        assert len(tool_results) == 1
        assert tool_results[0]["is_error"] is True
        assert "Tool crashed" in tool_results[0]["content"]

    @pytest.mark.asyncio
    async def test_execute_tool_calls_success_omits_is_error(self, mock_anthropic_tool_response):
        """Test that a successful tool execution does NOT set is_error."""
        gantry = MagicMock(spec=AgentGantry)
        mock_result = MagicMock()
        mock_result.status = "success"
        mock_result.result = {"temperature": 25, "unit": "C"}
        gantry.execute = AsyncMock(return_value=mock_result)

        client = AnthropicClient(api_key="test-key", gantry=gantry)
        tool_results = await client.execute_tool_calls(mock_anthropic_tool_response)

        assert len(tool_results) == 1
        assert "is_error" not in tool_results[0]
        assert tool_results[0]["content"] == '{"temperature": 25, "unit": "C"}'

    @pytest.mark.asyncio
    async def test_create_message_with_output_schema(self):
        """Test that output_schema injects the correct output_config payload."""
        client = AnthropicClient(api_key="test-key")
        mock_response = MagicMock()
        mock_response.content = []
        client._client.messages.create = AsyncMock(return_value=mock_response)

        schema = {
            "type": "object",
            "properties": {"status": {"type": "string"}},
            "required": ["status"],
            "additionalProperties": False,
        }
        await client.create_message(
            model="claude-sonnet-4-6",
            messages=[{"role": "user", "content": "Hello"}],
            auto_retrieve_tools=False,
            output_schema=schema,
        )

        call_kwargs = client._client.messages.create.call_args.kwargs
        assert "output_config" in call_kwargs
        assert call_kwargs["output_config"]["format"]["type"] == "json_schema"
        assert call_kwargs["output_config"]["format"]["schema"] == schema

    @pytest.mark.asyncio
    async def test_create_message_output_config_caller_wins(self):
        """Test that an explicit output_config kwarg takes precedence over output_schema."""
        client = AnthropicClient(api_key="test-key")
        mock_response = MagicMock()
        mock_response.content = []
        client._client.messages.create = AsyncMock(return_value=mock_response)

        caller_config = {"format": {"type": "text"}}
        schema = {"type": "object", "properties": {"x": {"type": "integer"}}}
        await client.create_message(
            model="claude-sonnet-4-6",
            messages=[{"role": "user", "content": "Hello"}],
            auto_retrieve_tools=False,
            output_schema=schema,
            output_config=caller_config,
        )

        call_kwargs = client._client.messages.create.call_args.kwargs
        assert call_kwargs["output_config"] == caller_config

    @pytest.mark.asyncio
    async def test_create_message_no_output_schema(self):
        """Test that output_config is absent when output_schema is not supplied."""
        client = AnthropicClient(api_key="test-key")
        mock_response = MagicMock()
        mock_response.content = []
        client._client.messages.create = AsyncMock(return_value=mock_response)

        await client.create_message(
            model="claude-sonnet-4-6",
            messages=[{"role": "user", "content": "Hello"}],
            auto_retrieve_tools=False,
        )

        call_kwargs = client._client.messages.create.call_args.kwargs
        assert "output_config" not in call_kwargs


class TestCreateAnthropicClient:
    """Tests for create_anthropic_client convenience function."""

    @pytest.mark.asyncio
    async def test_create_client_default(self):
        """Test creating client with defaults."""
        client = await create_anthropic_client(api_key="test-key")
        assert client is not None
        assert client._features.enable_interleaved_thinking is False
        assert client._features.enable_extended_thinking is False

    @pytest.mark.asyncio
    async def test_create_client_with_interleaved_thinking(self):
        """Test creating client with interleaved thinking."""
        client = await create_anthropic_client(
            api_key="test-key",
            enable_thinking="interleaved",
        )
        assert client._features.enable_interleaved_thinking is True
        assert client._features.enable_extended_thinking is False

    @pytest.mark.asyncio
    async def test_create_client_with_extended_thinking(self):
        """Test creating client with extended thinking."""
        client = await create_anthropic_client(
            api_key="test-key",
            enable_thinking="extended",
            thinking_budget_tokens=10000,
        )
        assert client._features.enable_extended_thinking is True
        assert client._features.thinking_budget_tokens == 10000

    @pytest.mark.asyncio
    async def test_create_client_with_gantry(self):
        """Test creating client with AgentGantry instance."""
        gantry = MagicMock(spec=AgentGantry)
        client = await create_anthropic_client(
            api_key="test-key",
            gantry=gantry,
        )
        assert client._gantry == gantry


class TestClaudeOpus48ThinkingGuards:
    """Guard tests documenting claude-opus-4-8 thinking-mode constraints.

    claude-opus-4-8 supports adaptive thinking only.  Extended thinking
    (``budget_tokens``) is **not** supported by the model and will cause an
    API error when submitted.  These tests verify that the recommended
    ``AnthropicFeatures`` configuration is correct and that the convenience
    factory produces the right payload for this model family.

    Source: https://platform.claude.com/docs/en/docs/about-claude/models/overview
    """

    def test_adaptive_config_is_correct_for_opus48(self):
        """AnthropicFeatures with adaptive thinking is the right config for opus-4-8."""
        features = AnthropicFeatures(adaptive_thinking_effort="medium")
        assert features.adaptive_thinking_effort == "medium"
        assert features.enable_extended_thinking is False
        assert features.thinking_budget_tokens is None

    def test_all_adaptive_effort_levels_accepted_for_opus48(self):
        """low / medium / high effort levels are all valid for adaptive mode on opus-4-8."""
        for effort in ("low", "medium", "high"):
            features = AnthropicFeatures(adaptive_thinking_effort=effort)
            assert features.adaptive_thinking_effort == effort
            assert features.enable_extended_thinking is False

    def test_default_features_safe_for_opus48(self):
        """Default AnthropicFeatures has no thinking enabled — safe for opus-4-8."""
        features = AnthropicFeatures()
        assert features.enable_extended_thinking is False
        assert features.thinking_budget_tokens is None
        assert features.adaptive_thinking_effort is None

    def test_extended_thinking_config_documents_incompatibility_with_opus48(self):
        """Extended thinking must NOT be combined with claude-opus-4-8.

        The dataclass accepts the flag (there is no model-level guard at
        the client layer); the caller is responsible for not passing
        enable_extended_thinking=True when targeting opus-4-8.  This test
        documents the boundary so that future guard implementations have a
        clear regression point.
        """
        correct = AnthropicFeatures(adaptive_thinking_effort="high")
        assert correct.enable_extended_thinking is False

        incompatible = AnthropicFeatures(
            enable_extended_thinking=True, thinking_budget_tokens=8000
        )
        assert incompatible.enable_extended_thinking is True

    @pytest.mark.asyncio
    async def test_create_anthropic_client_adaptive_for_opus48(self):
        """create_anthropic_client(enable_thinking='adaptive') is correct for opus-4-8."""
        client = await create_anthropic_client(
            api_key="test-key",
            enable_thinking="adaptive",
            adaptive_effort="medium",
        )
        assert client._features.adaptive_thinking_effort == "medium"
        assert client._features.enable_extended_thinking is False
        assert client._features.enable_interleaved_thinking is False

    @pytest.mark.asyncio
    async def test_opus48_adaptive_thinking_payload_shape(self):
        """Verify the thinking payload emitted for opus-4-8 adaptive mode."""
        features = AnthropicFeatures(adaptive_thinking_effort="high")
        client = AnthropicClient(api_key="test-key", features=features)

        mock_response = MagicMock()
        mock_response.content = []
        client._client.messages.create = AsyncMock(return_value=mock_response)

        await client.create_message(
            model="claude-opus-4-8",
            messages=[{"role": "user", "content": "Explain quantum entanglement"}],
            auto_retrieve_tools=False,
        )

        call_kwargs = client._client.messages.create.call_args.kwargs
        assert call_kwargs["model"] == "claude-opus-4-8"
        assert "thinking" in call_kwargs
        assert call_kwargs["thinking"]["type"] == "adaptive"
        assert call_kwargs["thinking"]["effort"] == "high"
        assert "budget_tokens" not in call_kwargs["thinking"]
