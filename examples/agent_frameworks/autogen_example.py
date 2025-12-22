import asyncio
from dotenv import load_dotenv
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient

from agent_gantry import AgentGantry
from agent_gantry.schema.execution import ToolCall

load_dotenv()

async def main():
    # 1. Initialize Agent-Gantry
    gantry = AgentGantry()

    @gantry.register
    def get_system_load():
        """Get the current system CPU load."""
        return "CPU Load: 15%"

    await gantry.sync()

    # 2. Retrieve tools from Gantry
    user_query = "Check the system load and report back."
    # Lowering threshold for SimpleEmbedder compatibility in this example
    retrieved_tools = await gantry.retrieve_tools(user_query, limit=1, score_threshold=0.1)

    # 3. Wrap Gantry tools for AutoGen (AG2)
    def make_autogen_tool(tool_name: str, gantry_instance: AgentGantry):
        """Factory function to properly bind tool name to AutoGen tool wrapper."""
        async def tool_wrapper() -> str:
            result = await gantry_instance.execute(ToolCall(tool_name=tool_name, arguments={}))
            return str(result.result) if result.status == "success" else result.error
        tool_wrapper.__name__ = tool_name
        tool_wrapper.__doc__ = f"Get the current {tool_name.replace('_', ' ')}."
        return tool_wrapper

    autogen_tools = []
    for ts in retrieved_tools:
        name = ts["function"]["name"]
        
        if name == "get_system_load":
            autogen_tools.append(make_autogen_tool(name, gantry))

    # 4. Setup AutoGen Agent
    model_client = OpenAIChatCompletionClient(model="gpt-4o")
    
    assistant = AssistantAgent(
        name="assistant",
        model_client=model_client,
        tools=autogen_tools,
        system_message="You are a helpful assistant. Use tools to answer questions."
    )

    # 5. Run Conversation
    print("--- Running AutoGen (AG2) Agent with Agent-Gantry ---")
    await Console(assistant.run_stream(task=user_query))

if __name__ == "__main__":
    asyncio.run(main())
