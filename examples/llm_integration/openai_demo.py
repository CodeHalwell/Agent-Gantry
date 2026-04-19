"""
OpenAI + Agent-Gantry integration demo.

Demonstrates four scenarios:
A. Responses API – recommended for agentic workloads (OpenAI's primary surface)
B. Dynamic retrieval via Chat Completions (still supported)
C. Static tool list (small toolsets)
D. Decorator-based automatic injection (recommended for wrappers)

OpenAI positioned the Responses API as the forward direction for agents and
published a sunset timeline for the Assistants API (August 2026). Scenario A
shows the current preferred pattern using client.responses.create().
"""

import asyncio
import json
import os
from typing import Any

from dotenv import load_dotenv

from agent_gantry import AgentGantry, set_default_gantry, with_semantic_tools
from agent_gantry.adapters.embedders.simple import SimpleEmbedder
from agent_gantry.schema.execution import ToolCall

load_dotenv()


async def main() -> None:
    print("=== Agent-Gantry + OpenAI Integration Demo ===\n")

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("❌ Error: OPENAI_API_KEY not found in environment.")
        print("   Please set it in your .env file.")
        return

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

    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=api_key)

    score_threshold = 0.1 if isinstance(gantry._embedder, SimpleEmbedder) else 0.5

    # --- Scenario A: Responses API (RECOMMENDED for agentic workloads) ---
    # The Responses API is OpenAI's primary surface for agents. It uses a flat
    # tool schema and returns output items rather than choices[].message.
    # Migrate from Assistants API before its August 2026 sunset.
    print("--- Scenario A: Responses API (recommended for agentic workloads) ---")
    query_a = "What's the weather in Tokyo?"
    print(f"User Query: '{query_a}'")

    # Retrieve tools in openai_responses dialect (flat schema)
    tools_responses = await gantry.retrieve_tools(
        query_a, limit=1, score_threshold=score_threshold, dialect="openai_responses"
    )
    print(f"Gantry retrieved {len(tools_responses)} tool(s): {[t['name'] for t in tools_responses]}")

    response_a = await client.responses.create(
        model="gpt-4.1",
        input=[{"role": "user", "content": query_a}],
        tools=tools_responses,
    )

    for item in response_a.output:
        if item.type == "function_call":
            print(f"LLM decided to call: {item.name}({item.arguments})")
            args = json.loads(item.arguments) if isinstance(item.arguments, str) else item.arguments
            result = await gantry.execute(ToolCall(tool_name=item.name, arguments=args))
            print(f"Execution Result: {result.result}")

            # Send tool result back to continue the conversation
            followup = await client.responses.create(
                model="gpt-4.1",
                input=[
                    {"role": "user", "content": query_a},
                    item.model_dump(),  # the function_call output item
                    {
                        "type": "function_call_output",
                        "call_id": item.call_id,
                        "output": str(result.result),
                    },
                ],
                tools=tools_responses,
            )
            for out in followup.output:
                if hasattr(out, "content"):
                    for c in out.content:
                        if hasattr(c, "text"):
                            print(f"Final answer: {c.text}")

    # --- Scenario B: Chat Completions (dynamic retrieval) ---
    print("\n--- Scenario B: Chat Completions API (dynamic retrieval) ---")
    query_b = "What's the weather in Tokyo?"
    print(f"User Query: '{query_b}'")

    tools_cc = await gantry.retrieve_tools(query_b, limit=1, score_threshold=score_threshold)
    print(f"Gantry retrieved {len(tools_cc)} tool(s): {[t['function']['name'] for t in tools_cc]}")

    response_b = await client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": query_b}],
        tools=tools_cc,
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

    # --- Scenario C: Static Tool List (small toolsets) ---
    print("\n--- Scenario C: Static Tool List (for small toolsets) ---")
    all_tools = [t.to_dialect("openai") for t in await gantry.list_tools()]
    print(f"Passing all {len(all_tools)} tools to LLM...")

    # --- Scenario D: Decorator-based automatic injection (RECOMMENDED for wrappers) ---
    print("\n--- Scenario D: @with_semantic_tools Decorator (RECOMMENDED) ---")

    set_default_gantry(gantry)

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

    query_d = "What is the stock price of AAPL?"
    print(f"User Query: '{query_d}'")

    response_d = await chat_with_tools(messages=[{"role": "user", "content": query_d}])

    tool_calls_d = response_d.choices[0].message.tool_calls
    if tool_calls_d:
        print(f"LLM decided to call: {tool_calls_d[0].function.name}")
    else:
        print("LLM did not call any tools.")


if __name__ == "__main__":
    asyncio.run(main())
