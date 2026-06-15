"""
Mistral AI + Agent-Gantry integration demo.

The official `mistralai` SDK was quarantined on PyPI on 2026-05-12 and is no
longer installable from the package index. Mistral's chat endpoint is
OpenAI-compatible, so this demo uses the `openai` Python SDK with
``base_url="https://api.mistral.ai/v1"``. All function-calling schemas produced
by Gantry's ``dialect="openai"`` or default adapter work without modification.

Install:
    pip install "agent-gantry[openai]"
"""

import asyncio
import json
import os
from typing import Any

from dotenv import load_dotenv

from agent_gantry import AgentGantry, set_default_gantry, with_semantic_tools
from agent_gantry.schema.execution import ToolCall

load_dotenv()


async def main():
    print("=== Agent-Gantry + Mistral AI Integration Demo ===\n")

    api_key = os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        print("❌ Error: MISTRAL_API_KEY not found in environment.")
        print("   Please set it in your .env file.")
        return

    gantry = AgentGantry()
    set_default_gantry(gantry)

    @gantry.register(tags=["translation"])
    def translate_text(text: str, target_lang: str) -> str:
        """Translate text to a target language."""
        return f"Translated '{text}' to {target_lang}: [Translated Text]"

    await gantry.sync()
    print(f"✅ Registered {gantry.tool_count} tools\n")

    # Mistral is OpenAI-compatible: point AsyncOpenAI at the Mistral base URL.
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=api_key, base_url="https://api.mistral.ai/v1")

    # --- Scenario: Dynamic Retrieval ---
    print("--- Scenario: Dynamic Retrieval ---")
    query = "Translate 'Hello World' to French"
    print(f"User Query: '{query}'")

    # Gantry's default dialect produces OpenAI-compatible function schemas,
    # which Mistral's endpoint accepts without modification.
    tools = await gantry.retrieve_tools(query, limit=1, score_threshold=0.1)
    print(f"Gantry retrieved {len(tools)} tool(s)")

    response = await client.chat.completions.create(
        model="mistral-large-latest",
        messages=[{"role": "user", "content": query}],
        tools=tools,
        tool_choice="auto",
    )

    tool_calls = response.choices[0].message.tool_calls
    if tool_calls:
        for tc in tool_calls:
            print(f"Mistral decided to call: {tc.function.name}({tc.function.arguments})")
            result = await gantry.execute(
                ToolCall(
                    tool_name=tc.function.name,
                    arguments=json.loads(tc.function.arguments)
                    if isinstance(tc.function.arguments, str)
                    else tc.function.arguments,
                )
            )
            print(f"Execution Result: {result.result}")

    # --- Scenario: Using @with_semantic_tools Decorator ---
    print("\n--- Scenario: Using @with_semantic_tools Decorator (RECOMMENDED) ---")

    @with_semantic_tools(limit=1, score_threshold=0.1, prompt_param="user_query")
    async def chat_with_mistral(user_query: str, tools: list[dict[str, Any]] = None):
        print(f"Decorator injected {len(tools) if tools else 0} tools")
        response = await client.chat.completions.create(
            model="mistral-large-latest",
            messages=[{"role": "user", "content": user_query}],
            tools=tools,
            tool_choice="auto",
        )
        if response.choices[0].message.tool_calls:
            tc = response.choices[0].message.tool_calls[0]
            print(f"Mistral (via decorator) called: {tc.function.name}")
            return tc.function.name
        return "No tool called"

    await chat_with_mistral("Translate 'Good Morning' to Spanish")

    # --- Scenario: Typed MistralAdapter (provider-specific convenience) ---
    # MistralAdapter(gantry).tools(query, limit=n) bakes in dialect="mistral",
    # which yields the same OpenAI-style function schema Mistral's endpoint
    # accepts — so it drops straight into the OpenAI-compatible client above.
    print("\n--- Scenario: Typed MistralAdapter ---")
    from agent_gantry.mistral import MistralAdapter

    query_adapter = "Translate 'Thank you' to German"
    print(f"User Query: '{query_adapter}'")

    tools_adapter = await MistralAdapter(gantry).tools(query_adapter, limit=1, score_threshold=0.1)
    print(f"Gantry retrieved {len(tools_adapter)} tool(s)")

    response_adapter = await client.chat.completions.create(
        model="mistral-large-latest",
        messages=[{"role": "user", "content": query_adapter}],
        tools=tools_adapter,
        tool_choice="auto",
    )
    if response_adapter.choices[0].message.tool_calls:
        tc = response_adapter.choices[0].message.tool_calls[0]
        print(f"Mistral (via adapter) called: {tc.function.name}")


if __name__ == "__main__":
    asyncio.run(main())
