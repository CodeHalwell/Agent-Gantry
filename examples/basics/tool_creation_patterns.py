"""
Five ways to register a tool: decorator, direct call, async function,
bound method, and renaming on registration.

Each pattern is executed afterwards to prove it works, then one is
retrieved by meaning. Runs offline with no API key.

Run: python examples/basics/tool_creation_patterns.py
"""

import asyncio

from dotfetch import DotFetch
from dotfetch.schema.execution import ToolCall


class MathService:
    """A sample service class."""

    def __init__(self, multiplier: float):
        self.multiplier = multiplier

    def multiply(self, value: float) -> float:
        """Multiplies a value by the configured multiplier."""
        return value * self.multiplier


async def main():
    print("=== dotfetch Tool Creation Patterns ===\n")

    fetcher = DotFetch()

    try:
        # --- Pattern 1: The Standard Decorator ---
        print("1. Registering via Decorator (@fetcher.register)")

        @fetcher.register(tags=["math", "basic"], examples=["add 5 and 3", "what's 12 plus 30"])
        def add(a: int, b: int) -> int:
            """Adds two integers together."""
            return a + b

        # --- Pattern 2: Direct Function Registration ---
        print("2. Registering via Function Call (fetcher.register(func))")

        def subtract(a: int, b: int) -> int:
            """Subtracts b from a."""
            return a - b

        # You can pass arguments just like the decorator
        fetcher.register(
            subtract,
            tags=["math", "basic"],
            examples=["take 4 away from 10", "what's 100 minus 37"],
        )

        # --- Pattern 3: Async Functions ---
        print("3. Registering Async Functions")

        @fetcher.register(
            tags=["async", "io"],
            examples=["fetch the page at example.com", "download the contents of this URL"],
        )
        async def fetch_data(url: str) -> str:
            """Simulates fetching data from a URL asynchronously."""
            await asyncio.sleep(0.1)  # Simulate I/O
            return f"Data from {url}"

        # --- Pattern 4: Class Methods (Bound Methods) ---
        print("4. Registering Class Methods")

        service = MathService(multiplier=10.0)

        # Register the bound method 'multiply' from the instance
        # Note: The 'self' parameter is handled automatically by the bound method
        fetcher.register(
            service.multiply,
            name="service_multiply",  # Good practice to give unique names to methods
            tags=["service", "math"],
            examples=[
                "multiply this number by the service multiplier",
                "scale 5 by the configured factor",
            ],
        )

        # --- Pattern 5: Renaming Tools ---
        print("5. Renaming Tools during Registration")

        def complex_internal_function_name_v2(x: int) -> int:
            """Returns the square of x."""
            return x * x

        # Expose it as 'square' to the LLM
        fetcher.register(
            complex_internal_function_name_v2,
            name="square",
            tags=["math"],
            examples=["what's 6 squared", "square the number 12"],
        )

        # --- Sync and Verify ---
        print("\nSyncing tools to registry...")
        await fetcher.sync()
        print(f"Total tools registered: {fetcher.tool_count}")

        # --- Test Execution ---
        print("\n--- Testing Executions ---")

        # Test Pattern 1
        res1 = await fetcher.execute(ToolCall(tool_name="add", arguments={"a": 5, "b": 3}))
        print(f"add(5, 3) = {res1.result}")

        # Test Pattern 2
        res2 = await fetcher.execute(ToolCall(tool_name="subtract", arguments={"a": 10, "b": 4}))
        print(f"subtract(10, 4) = {res2.result}")

        # Test Pattern 3
        res3 = await fetcher.execute(
            ToolCall(tool_name="fetch_data", arguments={"url": "example.com"})
        )
        print(f"fetch_data('example.com') = {res3.result}")

        # Test Pattern 4
        res4 = await fetcher.execute(
            ToolCall(tool_name="service_multiply", arguments={"value": 5.0})
        )
        print(f"service_multiply(5.0) [multiplier=10] = {res4.result}")

        # Test Pattern 5
        res5 = await fetcher.execute(ToolCall(tool_name="square", arguments={"x": 6}))
        print(f"square(6) = {res5.result}")

        # --- Retrieval Test ---
        print("\n--- Testing Retrieval ---")
        query = "I need to multiply a number by the service configuration"
        tools = await fetcher.retrieve_tools(query, limit=1)
        if tools:
            print(f"Query: '{query}'")
            print(f"Retrieved: {tools[0]['function']['name']}")
        else:
            print("No tools retrieved (check embedding model configuration)")
    finally:
        await fetcher.close()


if __name__ == "__main__":
    asyncio.run(main())
