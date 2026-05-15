"""
Microsoft Agent Framework (1.4.0) integration example.

Demonstrates how Agent-Gantry's semantic routing reduces token usage in
multi-agent systems by surfacing only the relevant tools per query.

The ``agent-framework`` floor in ``pyproject.toml`` is ``>=1.4.0,<2.0.0``.
``GantryToolBridge`` works against any AF release in that range; the bridge
itself is API-stable across 1.3.x → 1.4.x and is unaffected by the 1.4.0
changes (MCP tool-call metadata, SkillFrontmatter extraction, A2A SDK v1.0
migration, and DevUI security hardening — all upstream-only APIs or
infrastructure that Gantry does not consume).

Covers three construction patterns:

1. ``bridge.build_agent(client, query, ...)``
   Convenience one-liner: constructs ``Agent(client, instructions, ...)`` for
   single-agent flows.

2. ``bridge.as_agent(client, query, ...)``
   Same as Pattern 1, preferred when you need a first-class ``Agent`` to feed
   directly into ``HandoffBuilder`` or ``SequentialBuilder``.

3. ``bridge.build_workflow([specs], edges=[...])``
   Assembles a ``WorkflowAgent`` from multiple Gantry-equipped agents wired
   together via ``AgentExecutor`` nodes and ``WorkflowBuilder`` edges.
"""

import asyncio

from agent_framework.openai import OpenAIChatClient
from agent_framework.orchestrations import SequentialBuilder
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
    # Pattern 1: build_agent — convenience one-liner
    #
    # Constructs Agent(client, instructions, ...) internally.
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
    # Returns a first-class Agent suitable for feeding into SequentialBuilder,
    # HandoffBuilder, or WorkflowBuilder.
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

    # Wire agents in sequence using SequentialBuilder (no AgentExecutor needed).
    sequential_workflow = SequentialBuilder(
        participants=[billing_agent, support_agent]
    ).build()
    print(f"Built sequential workflow with {type(sequential_workflow).__name__}")

    # -----------------------------------------------------------------------
    # Pattern 3: build_workflow — fan-out routing with WorkflowBuilder edges
    #
    # build_workflow() wraps each Agent in AgentExecutor automatically before
    # passing it to WorkflowBuilder. Use edges= for explicit routing topology.
    # For conditional hand-off routing use build_handoff_workflow() instead.
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
            (
                "Triage",
                "Billing",
                lambda ctx: any(
                    kw in str(ctx).lower()
                    for kw in ("invoice", "billing", "payment", "charge")
                ),
            ),
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
