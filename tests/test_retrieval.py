"""
Tests for semantic retrieval and CLI behaviors.
"""

from __future__ import annotations

import pytest

from dotfetch import DotFetch
from dotfetch.cli.main import main
from dotfetch.schema.query import ConversationContext, ToolQuery


@pytest.mark.asyncio
async def test_retrieve_ranks_relevant_tool(sample_tools) -> None:
    """Ensure semantic retrieval surfaces the right tool."""
    fetcher = DotFetch()
    for tool in sample_tools:
        await fetcher.add_tool(tool)

    query = ToolQuery(context=ConversationContext(query="Please send an email to the customer"), score_threshold=0.0)
    result = await fetcher.retrieve(query)

    assert result.tools
    assert result.tools[0].tool.name == "send_email"
    assert len(result.tools) <= query.limit


@pytest.mark.asyncio
async def test_retrieval_latency(sample_tools) -> None:
    """Retrieval should stay within the latency budget."""
    fetcher = DotFetch()
    for tool in sample_tools:
        await fetcher.add_tool(tool)

    query = ToolQuery(context=ConversationContext(query="Generate a financial report"), limit=3)
    result = await fetcher.retrieve(query)

    # 2000ms budget accounts for sentence-transformers cold-start on slower CI
    # runners (macOS GitHub Actions regularly takes 300–400ms on first embed).
    # Subsequent warm calls are <50ms; this still catches severe regressions.
    assert result.total_time_ms < 2000


def test_cli_list_shows_demo_tools(capsys) -> None:
    """CLI list command should output demo tools."""
    exit_code = main(["list"])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "send_email" in captured.out


def test_cli_search_returns_match(capsys) -> None:
    """CLI search should surface relevant demo tool."""
    exit_code = main(["search", "send an urgent email"])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "send_email" in captured.out
