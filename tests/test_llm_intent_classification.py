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
        self.calls.append(
            {
                "query": query,
                "conversation_summary": conversation_summary,
            }
        )
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
        model="gpt-5.4-mini",
        max_tokens=100,
        temperature=0.0,
    )

    assert config.provider == "openai"
    assert config.model == "gpt-5.4-mini"
    assert config.max_tokens == 100
    assert config.temperature == 0.0

    # Default model must not be a scheduled-for-shutdown ID (gpt-4o-mini
    # retires 2026-10-23); the default changed in the 2026-08-03 audit.
    assert LLMConfig().model == "gpt-5.4-mini"


@pytest.mark.asyncio
async def test_llm_client_initialization():
    """Test that LLMClient can be initialized with config."""
    # Skip if openai is not installed
    pytest.importorskip("openai")

    # This test only checks initialization, not actual API calls
    config = LLMConfig(
        provider="openai",
        model="gpt-5.4-mini",
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
        assert intent == expected_intent, (
            f"Query '{query}' should be {expected_intent}, got {intent}"
        )


@pytest.mark.asyncio
async def test_reasoning_model_request_shape():
    """Reasoning-family OpenAI models must get max_completion_tokens and no
    temperature — the legacy params fail the request and every classification
    silently degrades to the UNKNOWN fallback."""
    from types import SimpleNamespace

    from agent_gantry.adapters.llm_client import LLMClient

    captured: dict = {}

    class FakeCompletions:
        async def create(self, **kwargs):
            captured.update(kwargs)
            message = SimpleNamespace(content="data_query")
            return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    def make_client(model: str) -> LLMClient:
        # Bypass __init__ (it imports the openai SDK); the request-shape
        # branch under test only needs these attributes.
        client = LLMClient.__new__(LLMClient)
        client._config = LLMConfig(provider="openai", model=model, api_key="test-key-not-real")
        client._provider = "openai"
        client._model = model
        client._client = SimpleNamespace(
            chat=SimpleNamespace(completions=FakeCompletions())
        )
        return client

    # Reasoning family: max_completion_tokens with headroom, no temperature
    result = await make_client("gpt-5.4-mini").classify_intent("show me sales data")
    assert result == "data_query"
    assert captured["max_completion_tokens"] >= 100
    assert "max_tokens" not in captured
    assert "temperature" not in captured

    # Legacy family keeps the classic params
    captured.clear()
    await make_client("gpt-4o-mini").classify_intent("show me sales data")
    assert "max_tokens" in captured
    assert "temperature" in captured
    assert "max_completion_tokens" not in captured


@pytest.mark.asyncio
async def test_keywords_match_at_token_start_only():
    """A keyword inside another word is not a mention of it.

    Substring matching counted ``budget``/``target`` as ``get``, ``thread``
    as ``read`` and ``admin`` as ``dm``, and at 0.15 of the final score a
    spurious intent match outweighs the semantic gap between neighbouring
    tools.
    """
    assert await classify_intent("what is the budget target for this thread") == TaskIntent.UNKNOWN
    assert await classify_intent("open the admin dashboard") == TaskIntent.ADMIN
    # Inflections and snake_case names still count: the keyword starts the token.
    assert await classify_intent("uploading the exported files") == TaskIntent.FILE_OPERATIONS
    assert await classify_intent("created a new record") == TaskIntent.DATA_MUTATION
    assert await classify_intent("run send_email for the team") == TaskIntent.COMMUNICATION


def test_tool_text_patterns_share_the_token_start_rule():
    """The per-tool intent boost uses the same keyword rule as classification."""
    from agent_gantry.core.router import INTENT_TAG_PATTERNS

    communication = INTENT_TAG_PATTERNS[TaskIntent.COMMUNICATION]
    assert communication.search("admin_panel manage the administrators") is None
    assert communication.search("send_dm send a direct message") is not None

    data_query = INTENT_TAG_PATTERNS[TaskIntent.DATA_QUERY]
    assert data_query.search("set_budget_target") is None
    assert data_query.search("get_budget retrieve budget figures") is not None
