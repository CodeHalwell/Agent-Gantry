"""
OpenAI + Agent-Gantry integration demo.

Demonstrates three scenarios for using Agent-Gantry with OpenAI's APIs:
A. Responses API (primary - OpenAI's recommended interface for agentic apps)
B. Chat Completions API (also supported; used in legacy integrations)
C. Decorator-based automatic injection (recommended for new projects)
"""

import asyncio
import json
import os
from typing import Any

from dotenv import load_dotenv

from agent_gantry import AgentGantry, set_default_gantry, with_semantic_tools
from agent_gantry.adapters.embedders.simple import SimpleEmbedder
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

    # Note: score_threshold=0.1 for SimpleEmbedder, use 0.5 (default) for Nomic/OpenAI
    score_threshold = 0.1 if isinstance(gantry._embedder, SimpleEmbedder) else 0.5

    # --- Scenario A: Responses API (Primary - OpenAI's recommended direction) ---
    print("--- Scenario A: Responses API (OpenAI's recommended agentic interface) ---")
    query = "What's the weather in Tokyo?"
    print(f"User Query: '{query}'")

    # Retrieve tools in Responses API format (flat schema, no nested 'function' key)
    tools_responses = await gantry.retrieve_tools(
        query, limit=1, score_threshold=score_threshold, dialect="openai_responses"
    )
    print(f"Gantry retrieved {len(tools_responses)} tool(s): {[t['name'] for t in tools_responses]}")

    response = await client.responses.create(
        model="gpt-4.1",
        input=[{"role": "user", "content": query}],
        tools=tools_responses,
    )

    for item in response.output:
        if item.type == "function_call":
            fn_args = json.loads(item.arguments) if isinstance(item.arguments, str) else item.arguments
            print(f"LLM decided to call: {item.name}({item.arguments})")
            result = await gantry.execute(ToolCall(tool_name=item.name, arguments=fn_args))
            print(f"Execution Result: {result.result}")

    # --- Scenario B: Chat Completions API (also supported) ---
    print("\n--- Scenario B: Chat Completions API (still fully supported) ---")
    query_b = "What's the weather in London?"
    print(f"User Query: '{query_b}'")

    # Default dialect produces Chat Completions format (nested 'function' key)
    tools_chat = await gantry.retrieve_tools(query_b, limit=1, score_threshold=score_threshold)
    print(f"Gantry retrieved {len(tools_chat)} tool(s): {[t['function']['name'] for t in tools_chat]}")

    response_b = await client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": query_b}],
        tools=tools_chat,
        tool_choice="auto",
    )

    tool_calls = response_b.choices[0].message.tool_calls
    if tool_calls:
        for tc in tool_calls:
            print(f"LLM decided to call: {tc.function.name}({tc.function.arguments})")
            result = await gantry.execute(
                ToolCall(tool_name=tc.function.name, arguments=json.loads(tc.function.arguments))
            )
            print(f"Execution Result: {result.result}")
    else:
        print("LLM did not call any tools.")

    # --- Scenario C: Using the Decorator (RECOMMENDED) ---
    print("\n--- Scenario C: Using @with_semantic_tools Decorator (RECOMMENDED) ---")

    set_default_gantry(gantry)

    # dialect="openai_responses" injects tools in Responses API format
    @with_semantic_tools(limit=1, score_threshold=0.1, dialect="openai_responses")
    async def chat_with_tools(
        messages: list[dict[str, str]], tools: list[dict[str, Any]] | None = None
    ):
        print(f"   [Decorator] Injected {len(tools) if tools else 0} tools (Responses API format)")
        return await client.responses.create(
            model="gpt-4.1",
            input=messages,
            tools=tools,
        )

    query_c = "What is the stock price of AAPL?"
    print(f"User Query: '{query_c}'")

    response_c = await chat_with_tools(messages=[{"role": "user", "content": query_c}])

    for item in response_c.output:
        if item.type == "function_call":
            print(f"LLM decided to call: {item.name}")
        elif hasattr(response_c, "output_text") and response_c.output_text:
            print(f"Response: {response_c.output_text}")
            break


if __name__ == "__main__":
    asyncio.run(main())
