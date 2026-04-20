"""
Microsoft Agent Framework (1.0 GA) integration example.

Demonstrates how Agent-Gantry's semantic routing reduces token usage in
multi-agent systems by surfacing only the relevant tools per query.

Covers three construction patterns:

1. ``bridge.build_agent(client, query, ...)``
   Convenience one-liner using ``client.as_agent()`` — fine for single-agent flows.

2. ``bridge.as_agent(client, query, ...)``
   Constructs ``Agent(client, ...)`` directly; the preferred form when you need
   the result as a first-class ``Agent`` for ``WorkflowBuilder``.

3. ``bridge.build_workflow([specs], edges=[...])``
   Assembles a ``WorkflowAgent`` from multiple Gantry-equipped agents wired
   together with ``WorkflowBuilder`` edges (fan-out / handoff / chain patterns).
"""

import asyncio

from agent_framework import WorkflowAgent, WorkflowBuilder
from agent_framework.openai import OpenAIChatClient
from dotenv import load_dotenv

from agent_gantry import AgentGantry
from agent_gantry.core.security import SecurityPolicy
from agent_gantry.integrations.agent_framework_bridge import GantryToolBridge
from agent_gantry.integrations.agent_framework_middleware import (
    GantryApprovalMiddleware,
    GantryObservabilityMiddleware,
)
from agent_gantry.schema.tool import ToolCapability

load_dotenv()


async def main() -> str:
    # -----------------------------------------------------------------------
    # 1. Register tools with Agent-Gantry
    # -----------------------------------------------------------------------
    gantry = AgentGantry()

    @gantry.register
    def get_user_profile(user_id: str) -> dict[str, str]:
        """Fetch a user's profile from the CRM system including plan and region."""
        return {"user_id": user_id, "plan": "pro", "region": "us-east"}

    @gantry.register
    def get_billing_info(user_id: str) -> dict[str, str]:
        """Retrieve billing information for a customer account."""
        return {"user_id": user_id, "balance": "$0.00", "next_invoice": "2026-04-01"}

    @gantry.register
    def search_knowledge_base(query: str) -> str:
        """Search the internal knowledge base for support articles."""
        return f"Found 3 articles matching '{query}'"

    @gantry.register
    def list_open_tickets(user_id: str) -> list[str]:
        """List open support tickets for a user."""
        return ["TICKET-001", "TICKET-042"]

    # Destructive: DELETE_DATA capability → AF approval_mode="always_require"
    @gantry.register(capabilities=[ToolCapability.DELETE_DATA])
    def delete_user_account(user_id: str) -> str:
        """Delete a user account. Destructive; requires human approval."""
        return f"deleted:{user_id}"

    await gantry.sync()

    bridge = GantryToolBridge(gantry, score_threshold=0.1)
    client = OpenAIChatClient()

    policy = SecurityPolicy(
        require_confirmation=["delete_*", "refund_*"],
        max_requests_per_minute=60,
    )
    middleware = [
        GantryApprovalMiddleware(policy),
        GantryObservabilityMiddleware(gantry),
    ]

    # -----------------------------------------------------------------------
    # Pattern 1: build_agent — convenience one-liner (uses client.as_agent)
    # -----------------------------------------------------------------------
    print("=== Pattern 1: build_agent ===")
    agent1 = await bridge.build_agent(
        client,
        query="What plan is user abc123 on?",
        name="SupportAgent",
        instructions="You are a support assistant. Use tools to fetch customer data.",
        limit=2,
        middleware=middleware,
    )
    print(f"Built agent via build_agent: {agent1.name}")

    # -----------------------------------------------------------------------
    # Pattern 2: as_agent — direct Agent(client, ...) construction
    #
    # This is the preferred form for WorkflowBuilder because it hands back a
    # plain Agent object without wrapping it in an extra factory call.
    # -----------------------------------------------------------------------
    print("\n=== Pattern 2: as_agent (direct Agent construction) ===")
    billing_agent = await bridge.as_agent(
        client,
        query="billing invoices payments",
        name="BillingAgent",
        instructions="You handle billing, invoices, and payment questions.",
        limit=2,
        middleware=middleware,
    )
    support_agent = await bridge.as_agent(
        client,
        query="support tickets bugs technical",
        name="SupportAgent",
        instructions="You handle technical support tickets and bug reports.",
        limit=2,
        middleware=middleware,
    )

    # Use the agents directly with WorkflowBuilder — simple linear handoff
    handoff_workflow = (
        WorkflowBuilder(start_executor=billing_agent)
        .add_chain([billing_agent, support_agent])
        .build()
    )
    chained_agent = WorkflowAgent(handoff_workflow, name="BillingThenSupport")
    print(f"Built WorkflowAgent via add_chain: {chained_agent.name}")

    # -----------------------------------------------------------------------
    # Pattern 3: build_workflow — fan-out routing with conditional edges
    #
    # A triage agent inspects the query and routes to Billing or Support based
    # on a condition. This is the "Handoff / Magentic" pattern.
    # -----------------------------------------------------------------------
    print("\n=== Pattern 3: build_workflow (fan-out / handoff) ===")
    workflow_agent = await bridge.build_workflow(
        agent_specs=[
            dict(
                client=client,
                query="triage customer request classify routing",
                name="Triage",
                instructions="Classify the customer request and route it to the right team.",
                limit=2,
            ),
            dict(
                client=client,
                query="billing invoices payments",
                name="Billing",
                instructions="Handle billing, invoices, and payment questions.",
                limit=2,
                middleware=middleware,
            ),
            dict(
                client=client,
                query="support tickets bugs technical",
                name="Support",
                instructions="Handle technical support and bug reports.",
                limit=2,
                middleware=middleware,
            ),
        ],
        edges=[
            # Conditional handoff: route to Billing when billing keywords present
            ("Triage", "Billing", lambda ctx: any(
                kw in str(ctx).lower() for kw in ("invoice", "billing", "payment", "charge")
            )),
            # Default fallback edge to Support
            ("Triage", "Support"),
        ],
        workflow_name="CustomerServiceWorkflow",
    )
    print(f"Built fan-out WorkflowAgent: {workflow_agent.name}")

    # -----------------------------------------------------------------------
    # Run the workflow agent on a sample query
    # -----------------------------------------------------------------------
    print("\n--- Running fan-out WorkflowAgent ---")
    user_query = "My last invoice has an incorrect charge."
    response = await workflow_agent.run(user_query)
    print(f"Response: {response}")
    return str(response)


if __name__ == "__main__":
    asyncio.run(main())
