import asyncio
import json
import os
from types import SimpleNamespace
from typing import Any, Sequence

from dotenv import load_dotenv

from agent_gantry import AgentGantry
from agent_gantry.schema.execution import ToolCall

load_dotenv()

try:
    from openai import AsyncOpenAI

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


class MockToolCall:
    def __init__(self, name: str, args: dict[str, Any]):
        self.id = f"call_{name}"
        self.type = "function"
        self.function = SimpleNamespace(name=name, arguments=json.dumps(args))


def _mock_tool_calls(text: str, tool_schemas: Sequence[dict[str, Any]]) -> list[MockToolCall]:
    names = {schema["function"]["name"] for schema in tool_schemas}
    calls: list[MockToolCall] = []
    if "order" in text and "get_order_status" in names:
        calls.append(MockToolCall("get_order_status", {"order_id": "12345"}))
    if "upgrade" in text and "upgrade_shipping" in names:
        calls.append(
            MockToolCall("upgrade_shipping", {"order_id": "12345", "speed": "express"})
        )
    return calls


async def _ensure_tools(gantry: AgentGantry) -> None:
    @gantry.register(tags=["orders"])
    def get_order_status(order_id: str) -> dict[str, str]:
        """Look up the latest status for an order."""
        return {"order_id": order_id, "status": "shipped", "carrier": "DHL"}

    @gantry.register(tags=["orders"])
    def upgrade_shipping(order_id: str, speed: str) -> dict[str, str]:
        """Upgrade shipping speed for an order."""
        return {"order_id": order_id, "speed": speed, "status": "upgraded"}

    await gantry.sync()


async def run_turn(
    *,
    gantry: AgentGantry,
    client: AsyncOpenAI | None,
    messages: list[dict[str, Any]],
    user_content: str,
) -> None:
    messages.append({"role": "user", "content": user_content})
    tools = await gantry.retrieve_tools(user_content, limit=2, score_threshold=0.2)

    tool_calls: list[Any] = []
    if client is not None:
        resp = await client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )
        message = resp.choices[0].message
        messages.append(
            {
                "role": "assistant",
                "content": message.content,
                "tool_calls": message.tool_calls,
            }
        )
        tool_calls = message.tool_calls or []
    else:
        tool_calls = _mock_tool_calls(user_content, tools)
        serialized_tool_calls = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            }
            for tc in tool_calls
        ]
        messages.append(
            {
                "role": "assistant",
                "content": "Let me check that.",
                "tool_calls": serialized_tool_calls,
            }
        )

    for tc in tool_calls:
        args = json.loads(tc.function.arguments)
        result = await gantry.execute(ToolCall(tool_name=tc.function.name, arguments=args))
        messages.append(
            {
                "role": "tool",
                "tool_call_id": tc.id,
                "content": json.dumps(result.result),
            }
        )

    if tool_calls:
        if client is not None:
            follow_up = await client.chat.completions.create(model="gpt-4o", messages=messages)
            final = follow_up.choices[0].message.content or ""
        else:
            final = (
                "Updated the order and confirmed shipping."
                if any(tc.function.name == "upgrade_shipping" for tc in tool_calls)
                else "Order status: shipped via DHL."
            )
        messages.append({"role": "assistant", "content": final})
        print(f"Assistant: {final}")
    else:
        print("Assistant: No tools were invoked this turn.")


async def main() -> None:
    gantry = AgentGantry()
    await _ensure_tools(gantry)

    api_key = os.environ.get("OPENAI_API_KEY")
    client = AsyncOpenAI(api_key=api_key) if OPENAI_AVAILABLE and api_key else None

    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": "You are a helpful support agent. Use tools when available.",
        }
    ]

    print("=== Multi-turn LLM + Gantry demo ===")
    await run_turn(
        gantry=gantry,
        client=client,
        messages=messages,
        user_content="My order 12345 seems delayed. Can you check the status?",
    )
    await run_turn(
        gantry=gantry,
        client=client,
        messages=messages,
        user_content="Please upgrade that order to express shipping.",
    )


if __name__ == "__main__":
    asyncio.run(main())
