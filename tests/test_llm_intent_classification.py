"""
Tests for LLM-based intent classification.

Tests the new LLM-based intent classification feature that falls back
to using an LLM when keyword-based classification fails.
"""

from __future__ import annotations

import pytest

from agent_gantry.adapters.llm_client import LLMClient
from agent_gantry.core.router import TaskIntent, classify_intent
from agent_gantry.schema.config import LLMConfig


class MockLLMClient:
    """Mock LLM client for testing without real API calls."""

    def __init__(self, response: str = "data_query") -> None:
        self.response = response
        self.calls: list[dict[str, str | None]] = []

    async def classify_intent(
        self,
        query: str,
        conversation_summary: str | None = None,
        available_intents: list[str] | None = None,
    ) -> str:
        """Mock classify_intent that returns a fixed response."""
        self.calls.append({
            "query": query,
            "conversation_summary": conversation_summary,
        })
        return self.response

    async def health_check(self) -> bool:
        return True


@pytest.mark.asyncio
async def test_classify_intent_keyword_match():
    """Test that keyword-based classification works without LLM."""
    # Test data_query intent
    intent = await classify_intent("search for users")
    assert intent == TaskIntent.DATA_QUERY

    # Test data_mutation intent
    intent = await classify_intent("create a new user")
    assert intent == TaskIntent.DATA_MUTATION

    # Test communication intent
    intent = await classify_intent("send an email notification")
    assert intent == TaskIntent.COMMUNICATION


@pytest.mark.asyncio
async def test_classify_intent_llm_fallback():
    """Test that LLM is used when keywords don't match."""
    mock_llm = MockLLMClient(response="analysis")

    # Query with no obvious keywords
    intent = await classify_intent(
        query="What's the trend?",
        use_llm=True,
        llm_client=mock_llm,
    )

    # Should have called LLM
    assert len(mock_llm.calls) == 1
    assert intent == TaskIntent.ANALYSIS


@pytest.mark.asyncio
async def test_classify_intent_llm_not_used_when_keywords_match():
    """Test that LLM is not called when keywords match."""
    mock_llm = MockLLMClient(response="unknown")

    # Query with clear keywords
    intent = await classify_intent(
        query="get user data",
        use_llm=True,
        llm_client=mock_llm,
    )

    # Should NOT have called LLM (keywords matched)
    assert len(mock_llm.calls) == 0
    assert intent == TaskIntent.DATA_QUERY


@pytest.mark.asyncio
async def test_classify_intent_llm_disabled():
    """Test that LLM is not used when disabled."""
    mock_llm = MockLLMClient(response="data_query")

    # Query with no keywords, but LLM disabled
    intent = await classify_intent(
        query="What's happening?",
        use_llm=False,
        llm_client=mock_llm,
    )

    # Should NOT have called LLM
    assert len(mock_llm.calls) == 0
    assert intent == TaskIntent.UNKNOWN


@pytest.mark.asyncio
async def test_classify_intent_with_conversation_summary():
    """Test that conversation summary is passed to LLM."""
    mock_llm = MockLLMClient(response="file_operations")

    # Use a query with no keywords to ensure LLM is called
    intent = await classify_intent(
        query="What about this?",
        conversation_summary="Previous discussion context here",
        use_llm=True,
        llm_client=mock_llm,
    )

    # Verify conversation summary was passed
    assert len(mock_llm.calls) == 1
    assert mock_llm.calls[0]["conversation_summary"] == "Previous discussion context here"
    assert intent == TaskIntent.FILE_OPERATIONS


@pytest.mark.asyncio
async def test_classify_intent_llm_error_fallback():
    """Test that errors in LLM classification fall back to UNKNOWN."""
    class FailingLLMClient:
        async def classify_intent(self, **kwargs):
            raise Exception("API error")

    failing_llm = FailingLLMClient()

    intent = await classify_intent(
        query="What's the status?",
        use_llm=True,
        llm_client=failing_llm,
    )

    # Should fall back to UNKNOWN on error
    assert intent == TaskIntent.UNKNOWN


@pytest.mark.asyncio
async def test_llm_client_config():
    """Test LLM client configuration."""
    config = LLMConfig(
        provider="openai",
        model="gpt-4o-mini",
        max_tokens=100,
        temperature=0.0,
    )

    assert config.provider == "openai"
    assert config.model == "gpt-4o-mini"
    assert config.max_tokens == 100
    assert config.temperature == 0.0


@pytest.mark.asyncio
async def test_llm_client_initialization():
    """Test that LLMClient can be initialized with config."""
    # Skip if openai is not installed
    pytest.importorskip("openai")

    # This test only checks initialization, not actual API calls
    config = LLMConfig(
        provider="openai",
        model="gpt-4o-mini",
        api_key="test-key-not-real",
    )

    # Should not raise an error during initialization
    client = LLMClient(config)
    assert client is not None
    assert await client.health_check()


@pytest.mark.asyncio
async def test_classify_intent_invalid_llm_response():
    """Test handling of invalid LLM responses."""
    mock_llm = MockLLMClient(response="invalid_intent_name")

    intent = await classify_intent(
        query="What's this about?",
        use_llm=True,
        llm_client=mock_llm,
    )

    # Should fall back to UNKNOWN when LLM returns invalid intent
    assert intent == TaskIntent.UNKNOWN


@pytest.mark.asyncio
async def test_all_intent_types():
    """Test all intent types can be classified via keywords."""
    test_cases = [
        ("search users", TaskIntent.DATA_QUERY),
        ("get the list", TaskIntent.DATA_QUERY),
        ("create new record", TaskIntent.DATA_MUTATION),
        ("update the database", TaskIntent.DATA_MUTATION),
        ("delete this item", TaskIntent.DATA_MUTATION),
        ("analyze the data", TaskIntent.ANALYSIS),
        ("calculate the sum", TaskIntent.ANALYSIS),
        ("send email", TaskIntent.COMMUNICATION),
        ("notify the team", TaskIntent.COMMUNICATION),
        ("upload file", TaskIntent.FILE_OPERATIONS),
        ("export to csv", TaskIntent.FILE_OPERATIONS),
        ("handle support ticket", TaskIntent.CUSTOMER_SUPPORT),
        ("process refund", TaskIntent.CUSTOMER_SUPPORT),
        ("change user permissions", TaskIntent.ADMIN),
        ("admin panel settings", TaskIntent.ADMIN),
    ]

    for query, expected_intent in test_cases:
        intent = await classify_intent(query)
        assert intent == expected_intent, f"Query '{query}' should be {expected_intent}, got {intent}"
