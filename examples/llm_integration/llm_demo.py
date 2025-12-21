import asyncio
import json
import os

from dotenv import load_dotenv

from agent_gantry import AgentGantry
from agent_gantry.schema.execution import ToolCall

load_dotenv()


# Try to import OpenAI, but we'll provide a mock fallback if no API key is present
try:
    from openai import AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

async def main():
    print("=== Agent-Gantry LLM Integration Demo ===\n")

    # 1. Initialize Gantry with Nomic Embeddings (for high-quality retrieval)
    print("1. Initializing AgentGantry with Nomic Embeddings...")
    try:
        from agent_gantry.adapters.embedders.nomic import NomicEmbedder
        embedder = NomicEmbedder(dimension=256)
        gantry = AgentGantry(embedder=embedder)
    except ImportError:
        print("Nomic dependencies missing. Falling back to SimpleEmbedder (less accurate).")
        gantry = AgentGantry()

    # 2. Register a diverse set of tools (The "Universe" of tools)
    print("2. Registering diverse toolset...")

    @gantry.register(tags=["math"])
    def add_numbers(a: float, b: float) -> float:
        """Add two numbers together."""
        return a + b

    @gantry.register(tags=["weather"])
    def get_weather(city: str) -> str:
        """Get the current weather for a city."""
        return f"The weather in {city} is sunny and 75°F."

    @gantry.register(tags=["finance"])
    def get_stock_price(ticker: str) -> str:
        """Get the current stock price for a ticker symbol."""
        return f"{ticker} is currently trading at $150.25"

    @gantry.register(tags=["system"])
    def get_system_time() -> str:
        """Get the current system time."""
        return "2023-10-27 10:00:00 UTC"

    @gantry.register(tags=["email"])
    def send_email(recipient: str, body: str) -> str:
        """Send an email."""
        return f"Sent email to {recipient}"

    # Sync to vector store
    await gantry.sync()
    print(f"   Registered {gantry.tool_count} tools in the registry.\n")

    # 3. Define the LLM interaction loop
    async def process_user_query(user_query: str):
        print(f"--- Processing Query: '{user_query}' ---")

        # A. RETRIEVAL: Get only the most relevant tools
        # We limit to 2 tools to demonstrate strict context filtering
        relevant_tools = await gantry.retrieve_tools(user_query, limit=2, score_threshold=0.4)

        print(f"   [Gantry] Context Reduction: {gantry.tool_count} total -> {len(relevant_tools)} relevant")
        for t in relevant_tools:
            print(f"   [Gantry] Selected: {t['function']['name']}")

        if not relevant_tools:
            print("   [Gantry] No relevant tools found. Asking LLM without tools.")

        # B. LLM CALL: Pass only the retrieved tools
        api_key = os.environ.get("OPENAI_API_KEY")

        if OPENAI_AVAILABLE and api_key:
            print("   [LLM] Calling OpenAI API...")
            client = AsyncOpenAI(api_key=api_key)

            messages = [{"role": "user", "content": user_query}]

            # Pass the filtered list of tools to the LLM
            response = await client.chat.completions.create(
                model="gpt-4o", # or gpt-3.5-turbo
                messages=messages,
                tools=relevant_tools if relevant_tools else None,
                tool_choice="auto" if relevant_tools else None
            )

            message = response.choices[0].message
            tool_calls = message.tool_calls

        else:
            print("   [LLM] Mocking LLM response (Set OPENAI_API_KEY to use real LLM)...")
            # Simple mock logic for demonstration purposes
            tool_calls = []
            if "50 + 20" in user_query and any(t['function']['name'] == 'add_numbers' for t in relevant_tools):
                # Mock a tool call object structure similar to OpenAI's
                class MockToolCall:
                    def __init__(self, name, args):
                        self.id = "call_123"
                        self.function = type('obj', (object,), {'name': name, 'arguments': json.dumps(args)})
                        self.type = 'function'

                tool_calls = [MockToolCall("add_numbers", {"a": 50, "b": 20})]
                print("   [LLM] Generated tool call: add_numbers(a=50, b=20)")
            elif "weather" in user_query.lower() and any(t['function']['name'] == 'get_weather' for t in relevant_tools):
                 class MockToolCall:
                    def __init__(self, name, args):
                        self.id = "call_456"
                        self.function = type('obj', (object,), {'name': name, 'arguments': json.dumps(args)})
                        self.type = 'function'
                 tool_calls = [MockToolCall("get_weather", {"city": "London"})]
                 print("   [LLM] Generated tool call: get_weather(city='London')")

        # C. EXECUTION: Execute the tool calls securely via Gantry
        if tool_calls:
            for tc in tool_calls:
                fn_name = tc.function.name
                fn_args = json.loads(tc.function.arguments)

                print(f"   [Gantry] Executing tool: {fn_name}")

                # Execute using AgentGantry's secure executor
                result = await gantry.execute(ToolCall(
                    tool_name=fn_name,
                    arguments=fn_args
                ))

                print(f"   [Result] {result.result}")
        else:
            print("   [Result] No tool calls made by LLM.")

        print("-" * 50 + "\n")

    # 4. Run scenarios
    await process_user_query("What is 50 + 20?")
    await process_user_query("What's the weather in London?")
    await process_user_query("Tell me a joke.") # Should retrieve no tools or irrelevant ones, LLM handles it

if __name__ == "__main__":
    asyncio.run(main())
