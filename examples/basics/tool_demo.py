import asyncio

from agent_gantry import AgentGantry
from agent_gantry.schema.execution import ToolCall


async def main():
    # 1. Initialize
    gantry = AgentGantry()

    try:
        # 2. Register a tool using the decorator
        @gantry.register(
            tags=["math"],
            examples=["how much tax do I pay on $100", "what's the sales tax on this order"],
        )
        def calculate_tax(amount: float) -> float:
            """Calculates US sales tax (8%) for a given amount."""
            return amount * 0.08

        # 3. Sync to index tools (required for semantic search)
        await gantry.sync()

        # 4. Retrieve relevant tools for a query
        # This returns OpenAI-compatible tool schemas you can pass to an LLM
        query = "How much tax do I pay on $100?"
        relevant_tools = await gantry.retrieve_tools(query, limit=1)
        print(f"Found tool: {relevant_tools}")

        # 5. Execute a tool
        result = await gantry.execute(
            ToolCall(tool_name="calculate_tax", arguments={"amount": 100.0})
        )
        print(f"Result: {result.result}")
    finally:
        await gantry.close()


if __name__ == "__main__":
    asyncio.run(main())
