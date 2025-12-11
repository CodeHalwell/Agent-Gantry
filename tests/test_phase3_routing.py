"""
Phase 3 routing behaviors: context awareness, intent, reranking, and MMR diversity.
"""

from __future__ import annotations

import pytest

from agent_gantry import AgentGantry
from agent_gantry.schema.query import ConversationContext, ToolQuery
from agent_gantry.schema.tool import ToolDefinition


class ForceReportReranker:
    """Deterministic reranker that prefers the reporting tool."""

    async def rerank(self, query: str, tools: list[tuple[ToolDefinition, float]], top_k: int):
        ordered = sorted(tools, key=lambda t: 0 if t[0].name == "generate_report" else 1)
        return ordered[:top_k]


@pytest.mark.asyncio
async def test_failed_tools_are_penalized(sample_tools) -> None:
    """Failed tools should be deprioritized in the final ranking."""
    gantry = AgentGantry()

    notify_customer = ToolDefinition(
        name="notify_customer",
        description="Notify a customer via email or SMS about an update.",
        parameters_schema={
            "type": "object",
            "properties": {
                "recipient": {"type": "string"},
                "channel": {"type": "string"},
                "message": {"type": "string"},
            },
            "required": ["recipient", "message"],
        },
        tags=["notification", "email", "sms"],
    )
    tools = sample_tools + [notify_customer]

    for tool in tools:
        await gantry.add_tool(tool)

    # Make send_email less attractive after a failure
    send_email = next(t for t in tools if t.name == "send_email")
    send_email.health.success_rate = 0.4
    send_email.cost.estimated_latency_ms = 9000

    context = ConversationContext(
        query="send a notification email to the customer",
        tools_failed=["send_email"],
    )
    result = await gantry.retrieve(
        ToolQuery(context=context, limit=1, diversity_factor=0.4),
    )

    assert result.tools
    assert result.tools[0].tool.name == "notify_customer"


@pytest.mark.asyncio
async def test_intent_matching_boosts_relevance(sample_tools) -> None:
    """Intent classification should lift the refund tool for customer support queries."""
    gantry = AgentGantry()
    for tool in sample_tools:
        await gantry.add_tool(tool)

    query = ToolQuery(context=ConversationContext(query="customer refund request"), limit=1)
    result = await gantry.retrieve(query)

    assert result.tools
    assert result.tools[0].tool.name == "process_refund"


@pytest.mark.asyncio
async def test_reranker_can_be_toggled(sample_tools) -> None:
    """Reranker should only apply when explicitly enabled."""
    gantry = AgentGantry(reranker=ForceReportReranker())
    for tool in sample_tools:
        await gantry.add_tool(tool)

    context = ConversationContext(query="send an email summary to finance")

    no_rerank = await gantry.retrieve(
        ToolQuery(context=context, limit=2, enable_reranking=False, score_threshold=0.0),
    )
    reranked = await gantry.retrieve(
        ToolQuery(context=context, limit=2, enable_reranking=True, score_threshold=0.0),
    )

    assert no_rerank.tools[0].tool.name == "send_email"
    assert reranked.tools[0].tool.name == "generate_report"


@pytest.mark.asyncio
async def test_mmr_promotes_diversity() -> None:
    """MMR should avoid returning near-duplicate tools when diversity is requested."""
    gantry = AgentGantry()

    email_customer = ToolDefinition(
        name="email_customer",
        description="Send an email update to the customer.",
        parameters_schema={
            "type": "object",
            "properties": {
                "to": {"type": "string"},
                "subject": {"type": "string"},
                "body": {"type": "string"},
            },
            "required": ["to", "subject", "body"],
        },
        tags=["email", "notification"],
    )
    email_admin = ToolDefinition(
        name="email_admin",
        description="Send an email update to the administrator.",
        parameters_schema=email_customer.parameters_schema,
        tags=["email", "notification"],
    )
    notify_customer = ToolDefinition(
        name="notify_customer",
        description="Send a notification or SMS to the customer.",
        parameters_schema=email_customer.parameters_schema,
        tags=["notification", "sms", "customer"],
    )

    for tool in [email_customer, email_admin, notify_customer]:
        await gantry.add_tool(tool)

    result = await gantry.retrieve(
        ToolQuery(
            context=ConversationContext(query="send an email or notification to the customer"),
            limit=2,
            diversity_factor=0.6,
        )
    )

    names = [tool.tool.name for tool in result.tools]
    assert "email_customer" in names
    assert "notify_customer" in names
