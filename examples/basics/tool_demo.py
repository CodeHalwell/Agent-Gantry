import asyncio

from agent_gantry import AgentGantry
from agent_gantry.schema.execution import ToolCall


async def main():
    # 1. Initialize
    gantry = AgentGantry()

    # 2. Register a tool using the decorator
    @gantry.register(tags=["math"])
    def calculate_tax(amount: float) -> float:
        """Calculates US sales tax (8%) for a given amount."""
        return amount * 0.08

    # 3. Sync to index tools (required for semantic search)
    await gantry.sync()

    # 4. Retrieve relevant tools for a query
    # This returns OpenAI-compatible tool schemas you can pass to an LLM
    query = "How much tax do I pay on $100?"
    # Note: We lower the score_threshold because the default SimpleEmbedder
    # produces low-similarity scores. With OpenAI/Nomic, the default (0.5) works fine.
    relevant_tools = await gantry.retrieve_tools(query, limit=1, score_threshold=0.1)
    print(f"Found tool: {relevant_tools}")

    # 5. Execute a tool
    result = await gantry.execute(ToolCall(
        tool_name="calculate_tax",
        arguments={"amount": 100.0}
    ))
    print(f"Result: {result.result}")

if __name__ == "__main__":
    asyncio.run(main())
