import asyncio

from agent_gantry import AgentGantry
from agent_gantry.schema.execution import ToolCall


class MathService:
    """A sample service class."""

    def __init__(self, multiplier: float):
        self.multiplier = multiplier

    def multiply(self, value: float) -> float:
        """Multiplies a value by the configured multiplier."""
        return value * self.multiplier


async def main():
    print("=== Agent-Gantry Tool Creation Patterns ===\n")

    gantry = AgentGantry()

    # --- Pattern 1: The Standard Decorator ---
    print("1. Registering via Decorator (@gantry.register)")

    @gantry.register(tags=["math", "basic"])
    def add(a: int, b: int) -> int:
        """Adds two integers together."""
        return a + b

    # --- Pattern 2: Direct Function Registration ---
    print("2. Registering via Function Call (gantry.register(func))")

    def subtract(a: int, b: int) -> int:
        """Subtracts b from a."""
        return a - b

    # You can pass arguments just like the decorator
    gantry.register(subtract, tags=["math", "basic"])

    # --- Pattern 3: Async Functions ---
    print("3. Registering Async Functions")

    @gantry.register(tags=["async", "io"])
    async def fetch_data(url: str) -> str:
        """Simulates fetching data from a URL asynchronously."""
        await asyncio.sleep(0.1)  # Simulate I/O
        return f"Data from {url}"

    # --- Pattern 4: Class Methods (Bound Methods) ---
    print("4. Registering Class Methods")

    service = MathService(multiplier=10.0)

    # Register the bound method 'multiply' from the instance
    # Note: The 'self' parameter is handled automatically by the bound method
    gantry.register(
        service.multiply,
        name="service_multiply",  # Good practice to give unique names to methods
        tags=["service", "math"]
    )

    # --- Pattern 5: Renaming Tools ---
    print("5. Renaming Tools during Registration")

    def complex_internal_function_name_v2(x: int) -> int:
        """Returns the square of x."""
        return x * x

    # Expose it as 'square' to the LLM
    gantry.register(complex_internal_function_name_v2, name="square", tags=["math"])


    # --- Sync and Verify ---
    print("\nSyncing tools to registry...")
    await gantry.sync()
    print(f"Total tools registered: {gantry.tool_count}")

    # --- Test Execution ---
    print("\n--- Testing Executions ---")

    # Test Pattern 1
    res1 = await gantry.execute(ToolCall(tool_name="add", arguments={"a": 5, "b": 3}))
    print(f"add(5, 3) = {res1.result}")

    # Test Pattern 2
    res2 = await gantry.execute(ToolCall(tool_name="subtract", arguments={"a": 10, "b": 4}))
    print(f"subtract(10, 4) = {res2.result}")

    # Test Pattern 3
    res3 = await gantry.execute(ToolCall(tool_name="fetch_data", arguments={"url": "example.com"}))
    print(f"fetch_data('example.com') = {res3.result}")

    # Test Pattern 4
    res4 = await gantry.execute(ToolCall(tool_name="service_multiply", arguments={"value": 5.0}))
    print(f"service_multiply(5.0) [multiplier=10] = {res4.result}")

    # Test Pattern 5
    res5 = await gantry.execute(ToolCall(tool_name="square", arguments={"x": 6}))
    print(f"square(6) = {res5.result}")

    # --- Retrieval Test ---
    print("\n--- Testing Retrieval ---")
    query = "I need to multiply a number by the service configuration"
    tools = await gantry.retrieve_tools(query, limit=1)
    if tools:
        print(f"Query: '{query}'")
        print(f"Retrieved: {tools[0]['function']['name']}")
    else:
        print("No tools retrieved (check embedding model configuration)")

if __name__ == "__main__":
    asyncio.run(main())
