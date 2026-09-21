"""
Project Demo - Massive Toolset with Semantic Routing

This demo shows how Agent-Gantry handles a 91-tool catalogue efficiently by:
1. Using persistent storage (LanceDB) - embeddings computed once, stored on disk
2. Semantic routing - only relevant tools sent to LLM, not all 91
3. Lazy imports - heavy dependencies loaded only when tool is executed

Run (from the repo root):
    python -m examples.project_demo.main_persistent

The first run builds the LanceDB cache under tools/.tool_cache/ (once); later
runs load it. `python -m examples.project_demo.tools.tools_persistent --sync`
rebuilds it explicitly. Selection runs without a key; set OPENAI_API_KEY for
the model call. Needs `agent-gantry[example-tools]` for the tools' lazy imports.
"""

import asyncio
import json
import os

from dotenv import load_dotenv
from openai import AsyncOpenAI

from agent_gantry.integrations.semantic_tools import with_semantic_tools
from agent_gantry.schema.execution import ToolCall

# Import the persistent tools module
from examples.project_demo.tools.tools_persistent import tools as gantry

load_dotenv()

# Constructed lazily: at module scope this raises on import when no key is
# set, which makes the file unimportable rather than merely unrunnable.
_client: AsyncOpenAI | None = None


def client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI()
    return _client


@with_semantic_tools(
    gantry,
    limit=3,  # Only 3 most relevant tools sent to LLM
    # score_threshold left at 0.0: an absolute cosine cutoff of 0.6 returns
    # nothing at all for most embedders once the query gets longer.
    score_threshold=0.0,
    dialect="openai_responses",
)
async def generate_response(prompt: str, tools: list | None = None):
    """LLM call that gets semantic tools injected."""
    first = await client().responses.create(
        model="gpt-5.4-mini",
        input=prompt,
        tools=tools,
        tool_choice="auto",
    )

    # Parse output items from Responses API
    output_items = first.output
    tool_calls = [item for item in output_items if item.type == "function_call"]
    text_items = [item for item in output_items if item.type == "message"]
    natural_text = text_items[0].content[0].text if text_items and text_items[0].content else ""

    tool_results = []

    if tool_calls:
        # Execute each tool call and collect results
        function_call_outputs = []
        for tc in tool_calls:
            result = await gantry.execute(
                ToolCall(tool_name=tc.name, arguments=json.loads(tc.arguments))
            )
            tool_results.append(result)
            function_call_outputs.append(
                {
                    "type": "function_call_output",
                    "call_id": tc.call_id,
                    "output": json.dumps(result.result),
                }
            )

        # Follow up with tool results
        follow_up = await client().responses.create(
            model="gpt-5.4-mini",
            input=function_call_outputs,
            previous_response_id=first.id,
        )

        # Extract text from follow-up response
        follow_up_text_items = [item for item in follow_up.output if item.type == "message"]
        if follow_up_text_items and follow_up_text_items[0].content:
            final_text = follow_up_text_items[0].content[0].text
        else:
            final_text = natural_text
        return final_text, tool_calls, tool_results

    return natural_text, tool_calls, tool_results


async def main() -> None:
    # The persistence half is the point of this demo and needs no key, so it
    # runs first: build or reuse the on-disk vector store, then retrieve from
    # it. Only the model call below is gated.
    from examples.project_demo.tools.tools_persistent import check_sync_status

    try:
        # Builds the store on first use (list_tools() syncs before listing), so
        # there is no separate "needs sync" step: the first run is the slow one.
        status = await check_sync_status()
        if status.get("built_now"):
            print("📦 First run: embedded the tools and created the vector database.")
            print("   (This only happens once - later runs load it from disk)")
            print()

        print(f"✓ {status['stored']} tools loaded from persistent storage")
        print(f"  Database: {status['db_path']}")
        print()

        user_query = "I have a dataset [12.5, 14.2, 11.8, 13.9, 15.1]. Can you calculate the mean and standard deviation, and also generate a random secure password for me?"
        print(f"User Query: '{user_query}'")
        print()

        selected = await gantry.retrieve_tools(user_query, limit=3)
        print(f"Gantry selected {len(selected)} of {status['stored']} tools:")
        for tool in selected:
            # Default dialect is OpenAI chat-completions shape, so the name
            # sits under "function" rather than at the top level.
            print(f"  • {tool['function']['name']}")
        print()

        if not os.getenv("OPENAI_API_KEY"):
            print("Set OPENAI_API_KEY to run the model call itself.")
            print("Everything above works without one, and the vector store")
            print("it built persists for the next run.")
            return

        final_text, tool_calls, tool_results = await generate_response(user_query)

        if final_text:
            print(f"LLM response: {final_text}")
            print()

        if tool_calls:
            print("Tools called:")
            for tc, result in zip(tool_calls, tool_results):
                print(f"  • {tc.name}({tc.arguments})")
                print(f"    Result: {result.result}")
        else:
            print("LLM did not call any tools.")
    finally:
        await gantry.close()


if __name__ == "__main__":
    asyncio.run(main())
