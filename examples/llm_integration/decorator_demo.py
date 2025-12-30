import asyncio

from agent_gantry import AgentGantry
from agent_gantry.integrations.semantic_tools import with_semantic_tools


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

async def main():
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

    # 2. Decorate a function with @with_semantic_tools
    # This will automatically intercept the call, find relevant tools for the prompt,
    # and inject them into the 'tools' argument.
    # We set score_threshold=0.1 because we are using the default SimpleEmbedder (hashing)
    @with_semantic_tools(gantry, limit=1, score_threshold=0.1)
    async def generate_response(prompt: str, tools: list | None = None):
        return await client.create(prompt, tools=tools)

    # 3. Call the decorated function
    print("--- Query 1: Weather ---")
    await generate_response("What is the weather in Tokyo?")

    print("\n--- Query 2: Stocks ---")
    await generate_response("What is the price of AAPL?")

if __name__ == "__main__":
    asyncio.run(main())
