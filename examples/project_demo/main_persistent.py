"""
Project Demo - Massive Toolset with Semantic Routing

This demo shows how Agent-Gantry handles 100+ tools efficiently by:
1. Using persistent storage (LanceDB) - embeddings computed once, stored on disk
2. Semantic routing - only relevant tools sent to LLM, not all 100+
3. Lazy imports - heavy dependencies loaded only when tool is executed

FIRST RUN:
    python -m examples.project_demo.tools.tools_persistent --sync

THEN RUN THIS:
    python -m examples.project_demo.main_persistent
"""

import asyncio
import json

from dotenv import load_dotenv
from openai import AsyncOpenAI

from agent_gantry.integrations.semantic_tools import with_semantic_tools
from agent_gantry.schema.execution import ToolCall

# Import the persistent tools module
from examples.project_demo.tools.tools_persistent import tools as gantry

load_dotenv()

client = AsyncOpenAI()


@with_semantic_tools(
    gantry,
    limit=3,  # Only 3 most relevant tools sent to LLM
    score_threshold=0.6,
    dialect="openai_responses",
    auto_sync=False,  # Don't re-sync on every call - tools are persisted!
)
async def generate_response(prompt: str, tools: list | None = None):
    """LLM call that gets semantic tools injected."""
    print(tools)
    first = await client.responses.create(
        model="gpt-4o-mini",
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
            function_call_outputs.append({
                "type": "function_call_output",
                "call_id": tc.call_id,
                "output": json.dumps(result.result),
            })

        # Follow up with tool results
        follow_up = await client.responses.create(
            model="gpt-4o-mini",
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
    # Check if tools are synced, if not, sync them automatically
    from examples.project_demo.tools.tools_persistent import check_sync_status, sync_tools

    status = await check_sync_status()
    if status.get("needs_sync"):
        print("📦 Tools not yet persisted. Creating vector database...")
        print("   (This only happens once - subsequent runs will be instant)")
        print()
        await sync_tools()
        print()
        # Refresh status after sync
        status = await check_sync_status()

    print(f"✓ {status['stored']} tools loaded from persistent storage")
    print(f"  Database: {status['db_path']}")
    print()

    user_query = "I have a dataset [12.5, 14.2, 11.8, 13.9, 15.1]. Can you calculate the mean and standard deviation, and also generate a random secure password for me?"
    print(f"User Query: '{user_query}'")
    print()

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


if __name__ == "__main__":
    asyncio.run(main())
