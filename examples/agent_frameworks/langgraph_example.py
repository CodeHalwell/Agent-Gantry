import asyncio

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from agent_gantry import AgentGantry
from agent_gantry.langgraph import LangGraphAdapter

load_dotenv()


async def main():
    # 1. Initialize Agent-Gantry
    gantry = AgentGantry()

    @gantry.register(tags=["gantry", "work"])
    def search_docs(query: str):
        """Search internal documentation about how Agent-Gantry works."""
        return f"Found results for '{query}': Agent-Gantry is a tool orchestrator."

    await gantry.sync()

    # 2. Setup LLM and Tools
    llm = ChatOpenAI(model="gpt-5.5")

    # Use Gantry to select tools and get them as native LangChain tools (which
    # LangGraph consumes) in one call — retrieval + conversion + execution.
    user_query = "How does Agent-Gantry work?"
    # Lowering threshold for SimpleEmbedder compatibility in this example
    gantry_tools = await LangGraphAdapter(gantry).select(
        user_query, limit=2, score_threshold=0.1
    )
    print(f"Gantry retrieved {len(gantry_tools)} tools.")

    # 3. Build the Agent using LangGraph's create_react_agent (LangGraph 1.x)
    agent = create_react_agent(llm, tools=gantry_tools)

    # 4. Run the Agent
    print("--- Running LangGraph Agent with Gantry-sourced tools ---")
    inputs = {"messages": [HumanMessage(content=user_query)]}

    result = await agent.ainvoke(inputs)

    print(f"\nFinal Response: {result['messages'][-1].content}")


if __name__ == "__main__":
    asyncio.run(main())
