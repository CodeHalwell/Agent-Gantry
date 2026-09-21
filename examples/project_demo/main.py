"""
Project Demo - a 300-tool catalogue routed through the OpenAI Responses API.

Imports the 300 tools registered in ``tools/tools.py`` (chemistry, maths,
files, networking, ...; needs ``agent-gantry[example-tools]`` for rdkit, pint,
pubchempy, sympy and requests), syncs them, and shows Gantry narrowing the
catalogue to three tools for one query before any model is called.

Run (from the repo root):
    python -m examples.project_demo.main

Selection runs without a key; set OPENAI_API_KEY to run the model call and
tool execution. See ``main_persistent.py`` for the same flow with an on-disk
LanceDB store.
"""

import asyncio
import json
import os

from dotenv import load_dotenv
from openai import AsyncOpenAI

from agent_gantry.integrations.semantic_tools import with_semantic_tools
from agent_gantry.schema.execution import ToolCall

# Import the tools module which creates and configures the gantry instance
from examples.project_demo.tools.tools import tools as gantry

load_dotenv()

# Constructed lazily: at module scope this raises on import when no key is
# set, which makes the file unimportable rather than merely unrunnable.
_client: AsyncOpenAI | None = None


def client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI()
    return _client


@with_semantic_tools(gantry, limit=3, score_threshold=0.0, dialect="openai_responses")
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
    user_query = "I have a dataset [12.5, 14.2, 11.8, 13.9, 15.1]. Can you calculate the mean and standard deviation, and also generate a random secure password for me?"
    print(f"User Query: '{user_query}'")

    try:
        # The retrieval half needs no key, so do it first and show it. This is
        # the part worth seeing: a 300-tool catalogue narrowed to three before
        # a single token is spent.
        selected = await gantry.retrieve_tools(user_query, limit=3)
        total = len(await gantry.list_tools())
        print(f"\nCatalogue: {total} tools. Gantry selected {len(selected)} for this query:")
        for tool in selected:
            # Default dialect is OpenAI chat-completions shape, so the name
            # sits under "function" rather than at the top level.
            print(f"  - {tool['function']['name']}")

        if not os.getenv("OPENAI_API_KEY"):
            print("\nSet OPENAI_API_KEY to run the model call itself.")
            print("Everything above works without one.")
            return

        final_text, tool_calls, tool_results = await generate_response(user_query)

        if final_text:
            print(f"LLM response: {final_text}")

        if tool_calls:
            for tc, result in zip(tool_calls, tool_results):
                print(f"LLM decided to call: {tc.name}({tc.arguments})")
                print(f"Execution Result: {result.result}")
        else:
            print("LLM did not call any tools.")
    finally:
        await gantry.close()


if __name__ == "__main__":
    asyncio.run(main())
