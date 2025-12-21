import asyncio
from typing import Annotated

from dotenv import load_dotenv
from agent_framework import ChatAgent
from agent_framework.openai import OpenAIChatClient
from pydantic import Field

from agent_gantry import AgentGantry
from agent_gantry.schema.execution import ToolCall

load_dotenv()


async def main() -> str:
    # 1) Initialize Agent-Gantry and register tools
    gantry = AgentGantry()

    @gantry.register
    def get_user_profile(user_id: str) -> dict[str, str]:
        """Fetch a user's profile from the CRM."""
        return {"user_id": user_id, "plan": "pro", "region": "us-east"}

    await gantry.sync()

    # 2) Retrieve relevant tools for this query (lower threshold for SimpleEmbedder demos)
    user_query = "What plan is user abc123 on?"
    tools = await gantry.retrieve_tools(user_query, limit=1, score_threshold=0.1)

    # 3) Wrap Gantry tools for Microsoft Agent Framework
    def make_tool_wrapper(tool_name: str, gantry_instance: AgentGantry):
        """Factory function to properly bind tool name to wrapper."""
        async def tool_wrapper(
            user_id: Annotated[str, Field(description="The user ID to look up.")]
        ) -> str:
            result = await gantry_instance.execute(
                ToolCall(tool_name=tool_name, arguments={"user_id": user_id})
            )
            return str(result.result) if result.status == "success" else str(result.error)

        # Set wrapper metadata for better debuggability and framework compatibility.
        tool_wrapper.__name__ = tool_name
        tool_wrapper.__doc__ = (
            f"Tool wrapper for Agent-Gantry tool '{tool_name}'. "
            "Invokes the underlying tool with a user_id argument."
        )
        return tool_wrapper

    agent_tools = []
    for schema in tools:
        name = schema["function"]["name"]

        if name == "get_user_profile":
            agent_tools.append(make_tool_wrapper(name, gantry))

    # 4) Create and run the Agent Framework ChatAgent
    chat_agent = ChatAgent(
        chat_client=OpenAIChatClient(model_id="gpt-4o"),
        instructions=(
            "You are a support assistant. Use the tools to fetch customer data."
        ),
        tools=agent_tools,
    )

    print("--- Running Microsoft Agent Framework with Agent-Gantry ---")
    response = await chat_agent.run(user_query)
    print(f"\nAgent Response: {response}")
    return str(response)


if __name__ == "__main__":
    asyncio.run(main())
