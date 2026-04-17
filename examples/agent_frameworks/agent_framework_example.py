"""
Microsoft Agent Framework (1.0 GA) integration example.

Demonstrates how Agent-Gantry's semantic routing reduces token usage in
multi-agent systems by surfacing only the relevant tools per query.

This example showcases the three-part GA integration:

1. ``GantryToolBridge`` — wraps Gantry-registered tools as AF
   ``FunctionTool``s with ``approval_mode`` auto-derived from each tool's
   Gantry ``ToolCapability`` set (destructive tools → ``always_require``).
2. ``GantryApprovalMiddleware`` — routes tool-execution gates through
   Gantry's :class:`SecurityPolicy` so AF's approval events mirror
   Gantry's own rate-limits, domain allowlists, and confirmation patterns.
3. ``bridge.build_agent(...)`` — one-liner that combines semantic tool
   retrieval with AF agent construction.

In Agent Framework 1.0 GA, ``OpenAIChatClient`` replaces the old
``OpenAIResponsesClient``; the RC-era ``OpenAIChatClient`` is now
``OpenAIChatCompletionClient``. See the 1.0 upgrade guide.
"""

import asyncio

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
    # 1) Initialize Agent-Gantry and register a mix of safe + destructive tools.
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

    # Destructive tool: Gantry's DELETE_DATA capability auto-elevates this to
    # AF approval_mode="always_require" inside the bridge.
    @gantry.register(capabilities=[ToolCapability.DELETE_DATA])
    def delete_user_account(user_id: str) -> str:
        """Delete a user account. Destructive; requires human approval."""
        return f"deleted:{user_id}"

    await gantry.sync()

    # 2) Create the bridge. Default behaviour wraps Gantry tools as AF
    #    FunctionTool objects with the idiomatic @tool decorator applied.
    bridge = GantryToolBridge(gantry, score_threshold=0.1)

    # 3) Configure middleware: enforce Gantry's security policy on top of AF's
    #    native approval flow, plus record per-call observability.
    policy = SecurityPolicy(
        require_confirmation=["delete_*", "refund_*"],
        max_requests_per_minute=60,
    )
    middleware = [
        GantryApprovalMiddleware(policy),
        GantryObservabilityMiddleware(gantry),
    ]

    # 4) Build the agent in one call: semantic retrieval + AF construction.
    #    This is equivalent to calling bridge.get_tools(...) then
    #    client.as_agent(...). Tools for this query are dynamically selected
    #    by Gantry — the LLM only sees ~2 tool schemas instead of all 4.
    client = OpenAIChatClient()
    user_query = "What plan is user abc123 on?"

    agent = await bridge.build_agent(
        client,
        query=user_query,
        name="SupportAgent",
        instructions="You are a support assistant. Use the tools to fetch customer data.",
        limit=2,
        middleware=middleware,
    )

    print("--- Running Microsoft Agent Framework with Agent-Gantry ---")
    bound_tools = (agent.default_options or {}).get("tools") or []
    print(
        f"Selected {len(bound_tools)} tools (from {len(gantry.export_tools())} "
        f"registered) for: '{user_query}'"
    )

    response = await agent.run(user_query)
    print(f"\nAgent Response: {response}")
    return str(response)


if __name__ == "__main__":
    asyncio.run(main())
