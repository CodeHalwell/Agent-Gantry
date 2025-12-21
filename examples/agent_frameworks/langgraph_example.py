import asyncio
import os
from typing import Annotated, TypedDict, Union
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage
from langchain.agents import create_agent
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode

from agent_gantry import AgentGantry
from agent_gantry.integrations.framework_adapters import fetch_framework_tools
from agent_gantry.schema.execution import ToolCall

load_dotenv()

# Define the state for our graph
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], lambda x, y: x + y]

async def main():
    # 1. Initialize Agent-Gantry
    gantry = AgentGantry()

    @gantry.register(tags=["gantry", "work"])
    def search_docs(query: str):
        """Search internal documentation about how Agent-Gantry works."""
        return f"Found results for '{query}': Agent-Gantry is a tool orchestrator."

    await gantry.sync()

    # 2. Setup LLM and Tools
    llm = ChatOpenAI(model="gpt-4o")
    
    # Use Gantry to fetch tools for the specific query
    user_query = "How does Agent-Gantry work?"
    # Lowering threshold for SimpleEmbedder compatibility in this example
    tools_schema = await fetch_framework_tools(gantry, user_query, framework="langgraph", score_threshold=0.1)
    print(f"Gantry retrieved {len(tools_schema)} tools.")
    
    # Wrap Gantry execution for LangGraph
    from langchain.tools import tool
    
    gantry_tools = []
    for ts in tools_schema:
        name = ts["function"]["name"]
        
        if name == "search_docs":
            @tool
            async def search_docs(query: str):
                """Search internal documentation about how Agent-Gantry works."""
                result = await gantry.execute(ToolCall(tool_name="search_docs", arguments={"query": query}))
                return result.result if result.status == "success" else result.error
            gantry_tools.append(search_docs)

    # 3. Build the Agent using the new create_agent pattern
    # This returns a compiled graph that handles tool calling
    agent = create_agent(llm, tools=gantry_tools)

    # 4. Run the Agent
    print(f"--- Running LangGraph Agent with Gantry-sourced tools ---")
    inputs = {"messages": [HumanMessage(content=user_query)]}
    
    # The agent created by create_agent is already a compiled graph
    result = await agent.ainvoke(inputs)
    
    print(f"\nFinal Response: {result['messages'][-1].content}")

if __name__ == "__main__":
    asyncio.run(main())
