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
sys.modules['anthropic'] = mock_anthropic

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
