import asyncio
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage

from agent_gantry import AgentGantry
from agent_gantry.schema.execution import ToolCall

load_dotenv()

async def main():
    # 1. Initialize Agent-Gantry
    gantry = AgentGantry()

    # 2. Register some tools
    @gantry.register(tags=["weather"])
    def get_weather(location: str):
        """Get the current weather in a given location."""
        return f"The weather in {location} is sunny and 25°C."

    @gantry.register(tags=["finance"])
    def get_stock_price(symbol: str):
        """Get the current stock price for a symbol."""
        return f"The stock price for {symbol} is $150.00."

    await gantry.sync()

    # 3. Define the user query
    user_query = "What's the weather in London and the stock price for AAPL?"

    # 4. Use Agent-Gantry to retrieve only relevant tools
    # This reduces the prompt size by only sending what's needed
    # Lowering threshold for SimpleEmbedder compatibility in this example
    retrieved_tools = await gantry.retrieve_tools(user_query, limit=2, score_threshold=0.1)
    
    print(f"Gantry retrieved {len(retrieved_tools)} tools.")

    # 5. Convert Gantry tools to LangChain tools
    # We wrap the Gantry execution so LangChain can call it
    def make_langchain_tool(tool_name: str, tool_desc: str, param_name: str, gantry_instance: AgentGantry):
        """Factory function to properly bind tool name to LangChain tool wrapper."""
        @tool
        async def tool_wrapper(**kwargs):
            result = await gantry_instance.execute(ToolCall(tool_name=tool_name, arguments=kwargs))
            return result.result if result.status == "success" else result.error
        tool_wrapper.__name__ = tool_name
        tool_wrapper.__doc__ = tool_desc
        return tool_wrapper

    langchain_tools = []
    for tool_schema in retrieved_tools:
        name = tool_schema["function"]["name"]
        desc = tool_schema["function"]["description"]
        
        if name == "get_weather":
            langchain_tools.append(make_langchain_tool(name, desc, "location", gantry))
        elif name == "get_stock_price":
            langchain_tools.append(make_langchain_tool(name, desc, "symbol", gantry))

    # 6. Setup LangChain Agent
    # In the latest LangChain, create_agent is the preferred way to build agents
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    
    agent = create_agent(llm, tools=langchain_tools)

    # 7. Run the agent
    print("\n--- Running LangChain Agent ---")
    # The new agent.invoke pattern uses a messages list
    response = await agent.ainvoke({
        "messages": [HumanMessage(content=user_query)]
    })
    
    # Extract the final message content
    final_message = response["messages"][-1]
    print(f"\nFinal Response: {final_message.content}")

if __name__ == "__main__":
    asyncio.run(main())
