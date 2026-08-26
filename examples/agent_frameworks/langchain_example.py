"""
LangChain + LangGraph Agent-Gantry integration example.

Uses ``langchain.agents.create_agent`` (the recommended agent constructor
since LangChain 1.0; the older ``langgraph.prebuilt.create_react_agent`` is
deprecated and removed outright in LangGraph 2.0).

Static tier shown here: Gantry selects the relevant slice once and the agent
is built with that fixed tool list. For per-turn re-selection (tools re-chosen
on every model turn), see ``LangGraphAdapter.areact_agent`` in
``langgraph_example.py`` — LangChain itself fixes tools at construction, so
the deeper tier lives one layer up in LangGraph.
"""

import asyncio

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from agent_gantry import AgentGantry
from agent_gantry.langchain import LangChainAdapter

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
    #    execution wiring). create_agent consumes these directly.
    # Lowering threshold for SimpleEmbedder compatibility in this example
    langchain_tools = await LangChainAdapter(gantry).select(
        user_query, limit=2, score_threshold=0.1
    )
    print(f"Gantry retrieved {len(langchain_tools)} tools.")

    # 4. Build and run the agent with the Gantry-selected slice.
    # temperature is not supported on gpt-5.5 (reasoning model); omit it.
    llm = ChatOpenAI(model="gpt-5.5")
    agent = create_agent(model=llm, tools=langchain_tools)

    print("\n--- Running LangChain agent with Gantry-sourced tools ---")
    result = await agent.ainvoke({"messages": [HumanMessage(content=user_query)]})

    print(f"\nFinal Response: {result['messages'][-1].content}")


if __name__ == "__main__":
    asyncio.run(main())
