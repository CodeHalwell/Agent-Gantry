import asyncio
from typing import Annotated, TypedDict
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage
from langchain.agents import create_agent

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
    
    def make_langgraph_tool(tool_name: str, tool_desc: str, gantry_instance: AgentGantry):
        """Factory function to properly bind tool name to wrapper."""
        @tool
        async def tool_wrapper(**kwargs):
            result = await gantry_instance.execute(ToolCall(tool_name=tool_name, arguments=kwargs))
            return result.result if result.status == "success" else result.error
        tool_wrapper.__name__ = tool_name
        tool_wrapper.__doc__ = tool_desc
        return tool_wrapper
    
    gantry_tools = []
    for ts in tools_schema:
        name = ts["function"]["name"]
        desc = ts["function"]["description"]
        
        if name == "search_docs":
            gantry_tools.append(make_langgraph_tool(name, desc, gantry))

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
