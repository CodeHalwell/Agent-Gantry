"""
OpenAI + Agent-Gantry integration demo.

Demonstrates three scenarios for using Agent-Gantry with OpenAI's chat completions API:
A. Dynamic tool retrieval (context window optimization)
B. Static tool list (for small toolsets)
C. Decorator-based automatic injection (recommended)
"""

import asyncio
import json
import os
from typing import Any

from dotenv import load_dotenv

from agent_gantry import AgentGantry, set_default_gantry, with_semantic_tools
from agent_gantry.schema.execution import ToolCall

# Load environment variables
load_dotenv()


async def main() -> None:
    print("=== Agent-Gantry + OpenAI Integration Demo ===\n")

    # 1. Check for API Key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("❌ Error: OPENAI_API_KEY not found in environment.")
        print("   Please set it in your .env file.")
        return

    # 2. Initialize Gantry
    # We use Nomic embeddings for better retrieval if available, else simple
    try:
        from agent_gantry.adapters.embedders.nomic import NomicEmbedder

        gantry = AgentGantry(embedder=NomicEmbedder())
        print("✅ Initialized with Nomic Embeddings")
    except ImportError:
        gantry = AgentGantry()
        print(
            "⚠️  Initialized with Simple Embeddings "
            "(Install 'agent-gantry[nomic]' for better results)"
        )

    # 3. Register Tools
    @gantry.register(tags=["weather"])
    def get_weather(location: str, unit: str = "celsius") -> str:
        """Get the current weather for a location."""
        return f"Weather in {location}: 22°{unit.upper()}, Sunny"

    @gantry.register(tags=["finance"])
    def get_stock_price(ticker: str) -> str:
        """Get the current stock price."""
        return f"{ticker.upper()}: $150.00"

    await gantry.sync()
    print(f"✅ Registered {gantry.tool_count} tools\n")

    # 4. Initialize OpenAI Client
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=api_key)

    # --- Scenario A: Dynamic Retrieval (The Gantry Way) ---
    print("--- Scenario A: Dynamic Retrieval (Context Window Optimization) ---")
    query = "What's the weather in Tokyo?"
    print(f"User Query: '{query}'")

    # Retrieve only relevant tools (OpenAI format by default)
    # Note: score_threshold=0.1 for SimpleEmbedder, use 0.5 (default) for Nomic/OpenAI
    score_threshold = 0.1 if isinstance(gantry._embedder, object) else 0.5
    tools = await gantry.retrieve_tools(query, limit=1, score_threshold=score_threshold)
    print(f"Gantry retrieved {len(tools)} tool(s): {[t['function']['name'] for t in tools]}")

    # Call LLM
    response = await client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": query}],
        tools=tools,
        tool_choice="auto",
    )

    tool_calls = response.choices[0].message.tool_calls
    if tool_calls:
        for tc in tool_calls:
            print(f"LLM decided to call: {tc.function.name}({tc.function.arguments})")

            # Execute securely via Gantry
            result = await gantry.execute(
                ToolCall(tool_name=tc.function.name, arguments=json.loads(tc.function.arguments))
            )
            print(f"Execution Result: {result.result}")
    else:
        print("LLM did not call any tools.")

    # --- Scenario B: Static Tool List (Small Toolsets) ---
    print("\n--- Scenario B: Static Tool List (For small toolsets) ---")
    # If you have < 10 tools, you might just want to pass them all
    all_tools = [t.to_openai_schema() for t in await gantry.list_tools()]
    print(f"Passing all {len(all_tools)} tools to LLM...")

    # (The rest of the LLM call is the same, just passing `tools=all_tools`)

    # --- Scenario C: Using the Decorator (Automatic Injection) - RECOMMENDED ---
    print("\n--- Scenario C: Using @with_semantic_tools Decorator (RECOMMENDED) ---")

    # Set default gantry for decorator usage
    set_default_gantry(gantry)

    # Wrap a function that calls the LLM. The decorator will:
    # 1. Extract the prompt from 'messages'
    # 2. Retrieve relevant tools
    # 3. Inject them into the 'tools' argument
    #
    # Note: score_threshold=0.1 is used here because we may be using SimpleEmbedder.
    # With Nomic or OpenAI embeddings, you can use the default (0.5).
    @with_semantic_tools(limit=1, score_threshold=0.1, dialect="openai")
    async def chat_with_tools(
        messages: list[dict[str, str]], tools: list[dict[str, Any]] | None = None
    ):
        print(f"   [Decorator] Injected {len(tools) if tools else 0} tools")
        return await client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=tools,
            tool_choice="auto" if tools else None,
        )

    query_c = "What is the stock price of AAPL?"
    print(f"User Query: '{query_c}'")

    response_c = await chat_with_tools(messages=[{"role": "user", "content": query_c}])

    tool_calls_c = response_c.choices[0].message.tool_calls
    if tool_calls_c:
        print(f"LLM decided to call: {tool_calls_c[0].function.name}")
    else:
        print("LLM did not call any tools.")


if __name__ == "__main__":
    asyncio.run(main())
