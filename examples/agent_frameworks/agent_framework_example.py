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
    agent_tools = []
    for schema in tools:
        name = schema["function"]["name"]

        if name == "get_user_profile":
            async def get_user_profile_tool(
                user_id: Annotated[str, Field(description="The user ID to look up.")]
            ) -> str:
                result = await gantry.execute(
                    ToolCall(tool_name="get_user_profile", arguments={"user_id": user_id})
                )
                return str(result.result) if result.status == "success" else str(result.error)

            agent_tools.append(get_user_profile_tool)

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
