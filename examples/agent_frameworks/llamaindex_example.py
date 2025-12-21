import asyncio
from dotenv import load_dotenv
from llama_index.llms.openai import OpenAI
from llama_index.core.tools import FunctionTool

from agent_gantry import AgentGantry
from agent_gantry.schema.execution import ToolCall

load_dotenv()

async def main():
    # 1. Initialize Agent-Gantry
    gantry = AgentGantry()

    @gantry.register
    def get_user_preferences(user_id: str):
        """Get preferences for a specific user."""
        return {"user_id": user_id, "theme": "dark", "notifications": True}

    await gantry.sync()

    # 2. Retrieve tools from Gantry
    user_query = "What are the preferences for user 'dev_123'?"
    # Lowering threshold for SimpleEmbedder compatibility in this example
    retrieved_tools = await gantry.retrieve_tools(user_query, limit=1, score_threshold=0.1)

    # 3. Convert Gantry tools to LlamaIndex tools
    def make_llama_tool(tool_name: str, tool_desc: str, gantry_instance: AgentGantry):
        """Factory function to properly bind tool name to LlamaIndex tool wrapper."""
        async def tool_wrapper(user_id: str):
            result = await gantry_instance.execute(ToolCall(tool_name=tool_name, arguments={"user_id": user_id}))
            return str(result.result) if result.status == "success" else result.error
        tool_wrapper.__doc__ = tool_desc
        return FunctionTool.from_defaults(async_fn=tool_wrapper)

    llama_tools = []
    for ts in retrieved_tools:
        name = ts["function"]["name"]
        desc = ts["function"]["description"]
        
        if name == "get_user_preferences":
            llama_tools.append(make_llama_tool(name, desc, gantry))

    # 4. Setup LlamaIndex Agent
    from llama_index.core.agent.workflow import ReActAgent
    llm = OpenAI(model="gpt-4o")
    agent = ReActAgent(tools=llama_tools, llm=llm)

    # 5. Run Agent
    print("--- Running LlamaIndex Agent with Agent-Gantry ---")
    response = await agent.run(user_msg=user_query)
    print(f"\nFinal Response: {response}")

if __name__ == "__main__":
    asyncio.run(main())
