"""
Hello world: register one tool, sync, retrieve it by meaning, execute it.

Runs offline with no API key.

Run: python examples/basics/tool_demo.py
"""

import asyncio

from dotfetch import DotFetch
from dotfetch.schema.execution import ToolCall


async def main():
    # 1. Initialize
    fetcher = DotFetch()

    try:
        # 2. Register a tool using the decorator
        @fetcher.register(
            tags=["math"],
            examples=["how much tax do I pay on $100", "what's the sales tax on this order"],
        )
        def calculate_tax(amount: float) -> float:
            """Calculates US sales tax (8%) for a given amount."""
            return amount * 0.08

        # 3. Sync to index tools. retrieve() would do this lazily on first use;
        #    syncing up front keeps the first query fast and the flow explicit.
        await fetcher.sync()

        # 4. Retrieve relevant tools for a query
        # This returns OpenAI-compatible tool schemas you can pass to an LLM
        query = "How much tax do I pay on $100?"
        relevant_tools = await fetcher.retrieve_tools(query, limit=1)
        print(f"Found tool: {relevant_tools}")

        # 5. Execute a tool
        result = await fetcher.execute(
            ToolCall(tool_name="calculate_tax", arguments={"amount": 100.0})
        )
        print(f"Result: {result.result}")
    finally:
        await fetcher.close()


if __name__ == "__main__":
    asyncio.run(main())
