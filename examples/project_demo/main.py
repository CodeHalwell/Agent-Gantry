import asyncio
import json

from dotenv import load_dotenv
from openai import AsyncOpenAI

from agent_gantry.integrations.decorator import with_semantic_tools
from agent_gantry.schema.execution import ToolCall

# Import the tools module which creates and configures the gantry instance
from examples.project_demo.tools.tools import tools as gantry

load_dotenv()

client = AsyncOpenAI()

@with_semantic_tools(gantry, limit=1, score_threshold=0.1)
async def generate_response(prompt: str, tools: list | None = None):
    """LLM call that gets semantic tools injected."""

    messages: list[dict[str, object]] = [{"role": "user", "content": prompt}]

    first = await client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        tools=tools,
        tool_choice="auto" if tools else None,
    )

    first_msg = first.choices[0].message
    tool_calls = first_msg.tool_calls or []
    natural_text = first_msg.content or ""

    tool_results = []

    if tool_calls:
        messages.append(
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": tc.type,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in tool_calls
                ],
            }
        )

        for tc in tool_calls:
            result = await gantry.execute(
                ToolCall(tool_name=tc.function.name, arguments=json.loads(tc.function.arguments))
            )
            tool_results.append(result)

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(result.result),
                }
            )

        follow_up = await client.chat.completions.create(model="gpt-4o", messages=messages)
        final_text = follow_up.choices[0].message.content or natural_text
        return final_text, tool_calls, tool_results

    return natural_text, tool_calls, tool_results


async def main() -> None:
    user_query = "What is the molecular weight of caffeine?"
    print(f"User Query: '{user_query}'")

    final_text, tool_calls, tool_results = await generate_response(user_query)

    if final_text:
        print(f"LLM response: {final_text}")

    if tool_calls:
        for tc, result in zip(tool_calls, tool_results):
            print(f"LLM decided to call: {tc.function.name}({tc.function.arguments})")
            print(f"Execution Result: {result.result}")
    else:
        print("LLM did not call any tools.")


if __name__ == "__main__":
    asyncio.run(main())
