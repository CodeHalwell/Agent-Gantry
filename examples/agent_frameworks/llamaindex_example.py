import asyncio

from dotenv import load_dotenv
from llama_index.llms.openai import OpenAI

from agent_gantry import AgentGantry
from agent_gantry.llamaindex import for_llamaindex

load_dotenv()


async def main():
    # 1. Initialize Agent-Gantry
    gantry = AgentGantry()

    @gantry.register
    def get_user_preferences(user_id: str):
        """Get preferences for a specific user."""
        return {"user_id": user_id, "theme": "dark", "notifications": True}

    await gantry.sync()

    # 2. Select relevant tools and get them as native LlamaIndex FunctionTools
    #    in one call (retrieval + conversion + execution wiring).
    user_query = "What are the preferences for user 'dev_123'?"
    # Lowering threshold for SimpleEmbedder compatibility in this example
    llama_tools = await for_llamaindex(gantry, user_query, limit=1, score_threshold=0.1)

    # 4. Setup LlamaIndex Agent
    from llama_index.core.agent.workflow import ReActAgent

    llm = OpenAI(model="gpt-5.5")
    agent = ReActAgent(tools=llama_tools, llm=llm)

    # 5. Run Agent
    print("--- Running LlamaIndex Agent with Agent-Gantry ---")
    response = await agent.run(user_msg=user_query)
    print(f"\nFinal Response: {response}")


if __name__ == "__main__":
    asyncio.run(main())
