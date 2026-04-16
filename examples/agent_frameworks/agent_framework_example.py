"""
Microsoft Agent Framework (1.0+) integration example.

Demonstrates how Agent-Gantry's semantic routing reduces token usage in
multi-agent systems by surfacing only the relevant tools per query.

Uses GantryToolBridge for seamless wrapping of Gantry tools as AF-compatible
Python callables, compatible with any AF chat client (OpenAI, Azure, Anthropic, etc.).
"""

import asyncio

from agent_framework.openai import OpenAIResponsesClient
from dotenv import load_dotenv

from agent_gantry import AgentGantry
from agent_gantry.integrations.agent_framework_bridge import GantryToolBridge

load_dotenv()


async def main() -> str:
    # 1) Initialize Agent-Gantry and register tools
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

    await gantry.sync()

    # 2) Create the bridge - this is the key integration point
    bridge = GantryToolBridge(gantry, score_threshold=0.1)

    # 3) Retrieve only relevant tools for this specific query
    #    This is where token savings happen: instead of sending ALL tools to the
    #    LLM, Gantry's semantic router selects only the relevant subset.
    user_query = "What plan is user abc123 on?"
    tools = await bridge.get_tools(user_query, limit=2)

    # 4) Create and run the Agent Framework agent with semantically selected tools
    client = OpenAIResponsesClient()
    agent = client.as_agent(
        name="SupportAgent",
        instructions="You are a support assistant. Use the tools to fetch customer data.",
        tools=tools,
    )

    print("--- Running Microsoft Agent Framework with Agent-Gantry ---")
    print(f"Selected {len(tools)} tools (from {3} registered) for: '{user_query}'")

    response = await agent.run(user_query)
    print(f"\nAgent Response: {response}")
    return str(response)


if __name__ == "__main__":
    asyncio.run(main())
