"""
Tests for semantic retrieval and CLI behaviors.
"""

from __future__ import annotations

import pytest

from agent_gantry import AgentGantry
from agent_gantry.cli.main import main
from agent_gantry.schema.query import ConversationContext, ToolQuery


@pytest.mark.asyncio
async def test_retrieve_ranks_relevant_tool(sample_tools) -> None:
    """Ensure semantic retrieval surfaces the right tool."""
    gantry = AgentGantry()
    for tool in sample_tools:
        await gantry.add_tool(tool)

    query = ToolQuery(context=ConversationContext(query="Please send an email to the customer"))
    result = await gantry.retrieve(query)

    assert result.tools
    assert result.tools[0].tool.name == "send_email"
    assert len(result.tools) <= query.limit


@pytest.mark.asyncio
async def test_retrieval_latency(sample_tools) -> None:
    """Retrieval should stay within the latency budget."""
    gantry = AgentGantry()
    for tool in sample_tools:
        await gantry.add_tool(tool)

    query = ToolQuery(context=ConversationContext(query="Generate a financial report"), limit=3)
    result = await gantry.retrieve(query)

    assert result.total_time_ms < 50


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
