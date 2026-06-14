"""
LangChain + LangGraph Agent-Gantry integration example.

Uses LangGraph's create_react_agent (the recommended approach since LangChain
dropped its own agent executor abstractions in favour of LangGraph in 1.x).
"""

import asyncio

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from agent_gantry import AgentGantry
from agent_gantry.langchain import for_langchain

load_dotenv()


async def main():
    # 1. Initialize Agent-Gantry
    gantry = AgentGantry()

    @gantry.register(tags=["weather"])
    def get_weather(location: str):
        """Get the current weather in a given location."""
        return f"The weather in {location} is sunny and 25°C."

    @gantry.register(tags=["finance"])
    def get_stock_price(symbol: str):
        """Get the current stock price for a symbol."""
        return f"The stock price for {symbol} is $150.00."

    await gantry.sync()

    # 2. Define the user query
    user_query = "What's the weather in London and the stock price for AAPL?"

    # 3. Use Agent-Gantry to select relevant tools and get them as native
    #    LangChain StructuredTools in one call (retrieval + conversion +
    #    execution wiring). LangGraph's create_react_agent consumes these.
    # Lowering threshold for SimpleEmbedder compatibility in this example
    langchain_tools = await for_langchain(gantry, user_query, limit=2, score_threshold=0.1)
    print(f"Gantry retrieved {len(langchain_tools)} tools.")

    # 5. Build and run a LangGraph ReAct agent
    # create_react_agent is the LangGraph-native replacement for the old
    # LangChain AgentExecutor pattern.
    # temperature is not supported on gpt-5.5 (reasoning model); omit it.
    llm = ChatOpenAI(model="gpt-5.5")
    agent = create_react_agent(llm, tools=langchain_tools)

    print("\n--- Running LangGraph ReAct Agent with Gantry-sourced tools ---")
    result = await agent.ainvoke({"messages": [HumanMessage(content=user_query)]})

    print(f"\nFinal Response: {result['messages'][-1].content}")


if __name__ == "__main__":
    asyncio.run(main())
