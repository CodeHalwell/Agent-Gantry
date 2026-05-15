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
from agent_gantry.schema.execution import ToolCall

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

    # 3. Use Agent-Gantry to retrieve only relevant tools
    # Lowering threshold for SimpleEmbedder compatibility in this example
    retrieved_tools = await gantry.retrieve_tools(user_query, limit=2, score_threshold=0.1)
    print(f"Gantry retrieved {len(retrieved_tools)} tools.")

    # 4. Wrap Gantry tools as LangChain tools for use in LangGraph
    from langchain_core.tools import tool

    def make_langchain_tool(tool_name: str, tool_desc: str, gantry_instance: AgentGantry):
        """Factory that binds the captured tool_name into a LangChain tool."""

        @tool
        async def tool_wrapper(**kwargs):
            result = await gantry_instance.execute(ToolCall(tool_name=tool_name, arguments=kwargs))
            return result.result if result.status == "success" else result.error

        tool_wrapper.__name__ = tool_name
        tool_wrapper.__doc__ = tool_desc
        return tool_wrapper

    langchain_tools = [
        make_langchain_tool(
            ts["function"]["name"],
            ts["function"]["description"],
            gantry,
        )
        for ts in retrieved_tools
    ]

    # 5. Build and run a LangGraph ReAct agent
    # create_react_agent is the LangGraph-native replacement for the old
    # LangChain AgentExecutor pattern.
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    agent = create_react_agent(llm, tools=langchain_tools)

    print("\n--- Running LangGraph ReAct Agent with Gantry-sourced tools ---")
    result = await agent.ainvoke({"messages": [HumanMessage(content=user_query)]})

    print(f"\nFinal Response: {result['messages'][-1].content}")


if __name__ == "__main__":
    asyncio.run(main())
