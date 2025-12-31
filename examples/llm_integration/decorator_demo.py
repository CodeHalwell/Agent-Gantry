"""
Semantic tools decorator demo.

Shows how the @with_semantic_tools decorator automatically retrieves and injects
relevant tools into LLM function calls based on the user's prompt.

This example demonstrates both the OLD and NEW patterns for using the decorator.
"""

import asyncio

from agent_gantry import AgentGantry, set_default_gantry, with_semantic_tools


# Mock LLM Client to simulate an API call
class MockLLMClient:
    async def create(self, prompt: str, tools: list | None = None) -> str:
        print(f"\n[MockLLM] Received Prompt: '{prompt}'")
        if tools:
            print(f"[MockLLM] Received {len(tools)} Tools:")
            for tool in tools:
                print(f"  - {tool['function']['name']}: {tool['function']['description']}")
            return "I have received the tools and would normally call one."
        else:
            print("[MockLLM] No tools received.")
            return "I have no tools to use."


async def main() -> None:
    gantry = AgentGantry()

    # 1. Register some tools
    @gantry.register
    def get_weather(city: str) -> str:
        """Get current weather for a city."""
        return f"Weather in {city}: Sunny"

    @gantry.register
    def get_stock_price(symbol: str) -> str:
        """Get current stock price."""
        return f"{symbol}: $150.00"

    @gantry.register
    def send_email(to: str, subject: str) -> str:
        """Send an email."""
        return "Email sent"

    await gantry.sync()

    client = MockLLMClient()

    print("=== Pattern 1: Using set_default_gantry() (RECOMMENDED) ===")
    # Set the default gantry instance for the current context
    # This allows using @with_semantic_tools without passing gantry explicitly
    set_default_gantry(gantry)

    # 2. Decorate a function with @with_semantic_tools
    # The decorator will:
    # - Extract the prompt from function arguments
    # - Retrieve semantically relevant tools
    # - Inject them into the 'tools' parameter
    #
    # Note: score_threshold=0.1 is used because the default SimpleEmbedder
    # uses deterministic hashing, which produces low similarity scores.
    # For production, use OpenAI or Nomic embeddings with the default threshold (0.5).
    @with_semantic_tools(limit=1, score_threshold=0.1)
    async def generate_response(prompt: str, tools: list | None = None) -> str:
        return await client.create(prompt, tools=tools)

    # 3. Call the decorated function
    print("\n--- Query 1: Weather ---")
    await generate_response("What is the weather in Tokyo?")

    print("\n--- Query 2: Stocks ---")
    await generate_response("What is the price of AAPL?")

    print("\n=== Pattern 2: Explicit Gantry (BACKWARDS COMPATIBLE) ===")
    # You can still pass gantry explicitly if you prefer
    @with_semantic_tools(gantry, limit=1, score_threshold=0.1)
    async def generate_response_explicit(prompt: str, tools: list | None = None) -> str:
        return await client.create(prompt, tools=tools)

    print("\n--- Query 3: Email ---")
    await generate_response_explicit("Send an email to the team")


if __name__ == "__main__":
    asyncio.run(main())
