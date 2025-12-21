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
    llama_tools = []
    for ts in retrieved_tools:
        name = ts["function"]["name"]
        
        if name == "get_user_preferences":
            async def get_user_preferences(user_id: str):
                """Get preferences for a specific user."""
                result = await gantry.execute(ToolCall(tool_name="get_user_preferences", arguments={"user_id": user_id}))
                return str(result.result) if result.status == "success" else result.error
            
            llama_tools.append(FunctionTool.from_defaults(async_fn=get_user_preferences))

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
