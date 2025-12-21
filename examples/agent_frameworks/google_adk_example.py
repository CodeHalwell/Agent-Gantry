import asyncio

from dotenv import load_dotenv
from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import FunctionTool
from google.genai import types

from agent_gantry import AgentGantry
from agent_gantry.schema.execution import ToolCall

load_dotenv()

APP_NAME = "gantry_adk_app"
USER_ID = "demo_user"
SESSION_ID = "demo_session"


async def run_query(query: str) -> str:
    # 1) Initialize Gantry and register tools
    gantry = AgentGantry()

    @gantry.register
    def get_order_status(order_id: str) -> dict[str, str]:
        """Look up the current status for an order ID."""
        return {"order_id": order_id, "status": "shipped", "carrier": "DHL"}

    await gantry.sync()

    # 2) Retrieve relevant tools for this query
    retrieved_tools = await gantry.retrieve_tools(query, limit=1, score_threshold=0.1)

    # 3) Wrap Gantry tools as ADK FunctionTools
    adk_tools: list[FunctionTool] = []
    for schema in retrieved_tools:
        name = schema["function"]["name"]

        if name == "get_order_status":
            async def get_order_status_tool(order_id: str) -> str:
                result = await gantry.execute(
                    ToolCall(tool_name="get_order_status", arguments={"order_id": order_id})
                )
                return str(result.result) if result.status == "success" else str(result.error)

            adk_tools.append(FunctionTool(func=get_order_status_tool))

    # 4) Build ADK agent and runner
    adk_agent = Agent(
        model="gemini-2.0-flash",
        name="gantry_adk_agent",
        instruction="You are a helpful agent that can look up order statuses via tools.",
        tools=adk_tools,
    )

    session_service = InMemorySessionService()
    await session_service.create_session(app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID)
    runner = Runner(agent=adk_agent, app_name=APP_NAME, session_service=session_service)

    # 5) Execute the query and return the final response
    events = runner.run_async(
        user_id=USER_ID,
        session_id=SESSION_ID,
        new_message=types.Content(role="user", parts=[types.Part(text=query)]),
    )

    final_text = ""
    async for event in events:
        if event.is_final_response():
            final_text = event.content.parts[0].text
            break

    return final_text


async def main() -> None:
    user_query = "What's the status of order 12345?"
    print("--- Running Google ADK Agent with Agent-Gantry ---")
    result = await run_query(user_query)
    print(f"\nAgent Response: {result}")


if __name__ == "__main__":
    asyncio.run(main())
