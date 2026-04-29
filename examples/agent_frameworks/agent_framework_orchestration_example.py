"""
Microsoft Agent Framework 1.2 — multi-agent orchestration with Agent-Gantry.

Demonstrates three production orchestration patterns, each powered by Gantry's
semantic tool routing:

1. **Sequential**: a research agent pipes its answer into a writer agent.
2. **Concurrent**: two analyst agents fan out on the same brief and their
   outputs are aggregated.
3. **Handoff**: a triage agent decides whether to keep the task or hand it
   off to a specialist.

Each participating agent uses a distinct, semantically selected subset of
the Gantry tool registry, so the LLM only sees the tools relevant to that
agent's role (i.e. the token-saving benefit compounds across participants).

AF 1.1.0 added ``GeminiChatClient`` (``from agent_framework.gemini import
GeminiChatClient``). Pass it as ``client`` to any bridge helper to run
agents on Gemini models. Set ``GOOGLE_API_KEY`` and swap ``OpenAIChatClient``
for ``GeminiChatClient`` to try it.

Run:

    pip install "agent-gantry[agent-frameworks]"
    export OPENAI_API_KEY=...
    python examples/agent_frameworks/agent_framework_orchestration_example.py
"""

from __future__ import annotations

import asyncio

from agent_framework.openai import OpenAIChatClient
from agent_framework.orchestrations import (
    ConcurrentBuilder,
    HandoffBuilder,
    SequentialBuilder,
)
from dotenv import load_dotenv

from agent_gantry import AgentGantry
from agent_gantry.integrations.agent_framework_bridge import GantryToolBridge

load_dotenv()


async def build_gantry() -> AgentGantry:
    """Register a realistic multi-role tool set for customer support."""
    gantry = AgentGantry()

    # Research role
    @gantry.register(tags=["research"])
    def search_knowledge_base(query: str) -> str:
        """Search the internal KB for support articles."""
        return f"3 articles matching '{query}'"

    @gantry.register(tags=["research"])
    def lookup_order(order_id: str) -> dict:
        """Look up order details by id."""
        return {"order_id": order_id, "status": "shipped", "carrier": "DHL"}

    # Billing role
    @gantry.register(tags=["billing"])
    def get_invoice(user_id: str) -> dict:
        """Get the latest invoice for a user."""
        return {"user_id": user_id, "amount_due": "$0.00"}

    # Writing role
    @gantry.register(tags=["writing"])
    def draft_email(subject: str, body: str) -> str:
        """Draft a customer-facing email response."""
        return f"DRAFT: {subject}\n\n{body}"

    await gantry.sync()
    return gantry


async def sequential_pipeline(
    bridge: GantryToolBridge, client: OpenAIChatClient, query: str
) -> None:
    """Research → Write pipeline with distinct Gantry tool slices per step."""
    researcher = await bridge.build_agent(
        client,
        query="research tools (search knowledge base, lookup order)",
        name="researcher",
        instructions="Gather facts using the research tools.",
        limit=3,
    )
    writer = await bridge.build_agent(
        client,
        query="compose a customer-facing email",
        name="writer",
        instructions="Draft a concise, empathetic customer email.",
        limit=2,
    )
    workflow = SequentialBuilder(participants=[researcher, writer]).build()
    print(f"\n[sequential] workflow built with {2} participants")
    result = await workflow.run(query)
    print(f"[sequential] result:\n{result}")


async def concurrent_analysts(
    bridge: GantryToolBridge, client: OpenAIChatClient, brief: str
) -> None:
    """Two analysts fan out on the same brief, outputs aggregated."""
    billing_analyst = await bridge.build_agent(
        client,
        query="billing and invoices",
        name="billing_analyst",
        instructions="Assess the financial situation.",
        limit=2,
    )
    ops_analyst = await bridge.build_agent(
        client,
        query="order lookup and knowledge base",
        name="ops_analyst",
        instructions="Assess the operational situation.",
        limit=2,
    )
    workflow = ConcurrentBuilder(
        participants=[billing_analyst, ops_analyst],
    ).build()
    print(f"\n[concurrent] workflow built with {2} participants")
    result = await workflow.run(brief)
    print(f"[concurrent] result:\n{result}")


async def handoff_triage(
    bridge: GantryToolBridge, client: OpenAIChatClient, query: str
) -> None:
    """Triage agent routes to either billing or research specialist."""
    triage = await bridge.build_agent(
        client,
        query="triage and route",
        name="triage",
        instructions=(
            "Decide whether this is a billing question or a research question. "
            "Hand off to the appropriate specialist."
        ),
        limit=4,
    )
    billing = await bridge.build_agent(
        client,
        query="billing and invoices",
        name="billing_specialist",
        instructions="Answer billing questions.",
        limit=2,
    )
    research = await bridge.build_agent(
        client,
        query="knowledge base and orders",
        name="research_specialist",
        instructions="Answer research questions.",
        limit=2,
    )

    # Handoff requires per-service-call history persistence on every agent.
    for a in (triage, billing, research):
        a.require_per_service_call_history_persistence = True

    workflow = (
        HandoffBuilder(name="support_desk")
        .participants([triage, billing, research])
        .with_start_agent(triage)
        .add_handoff(source=triage, targets=[billing, research])
        .build()
    )
    print("\n[handoff] workflow built")
    result = await workflow.run(query)
    print(f"[handoff] result:\n{result}")


async def main() -> None:
    gantry = await build_gantry()
    bridge = GantryToolBridge(gantry, score_threshold=0.1)
    client = OpenAIChatClient()

    await sequential_pipeline(
        bridge,
        client,
        "Order ORD-42 is delayed — look up the status and draft a polite "
        "apology email to the customer.",
    )
    await concurrent_analysts(
        bridge,
        client,
        "Review account u_123: billing health and recent order status.",
    )
    await handoff_triage(
        bridge, client, "My latest invoice looks wrong, can you check?"
    )


if __name__ == "__main__":
    asyncio.run(main())
