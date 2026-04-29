"""
LangChain + LangGraph Agent-Gantry integration example.

Uses ``GantryToolBridge`` to wrap Gantry tools as native LangChain
``StructuredTool`` instances. The bridge enforces an optional
``SecurityPolicy`` and approval callback before each tool runs and routes
every invocation through ``gantry.execute`` so retries, circuit breakers,
and telemetry all flow through uniformly.

The wrapped tools are usable in either LangChain (``langchain.agents.create_agent``)
or LangGraph (``langgraph.prebuilt.create_react_agent``) — the same
``StructuredTool`` works in both because LangGraph reuses LangChain's tool
abstraction.
"""

import asyncio

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from agent_gantry import AgentGantry
from agent_gantry.core.security import SecurityPolicy
from agent_gantry.integrations.langchain_bridge import GantryToolBridge

load_dotenv()


async def main() -> None:
    # 1. Initialise Agent-Gantry and register tools.
    gantry = AgentGantry()

    @gantry.register(tags=["weather"])
    def get_weather(location: str) -> str:
        """Get the current weather in a given location."""
        return f"The weather in {location} is sunny and 25C."

    @gantry.register(tags=["finance"])
    def get_stock_price(symbol: str) -> str:
        """Get the current stock price for a symbol."""
        return f"The stock price for {symbol} is $150.00."

    await gantry.sync()

    # 2. Build a GantryToolBridge with a security policy. Any tool whose
    # name matches ``require_confirmation`` will block execution unless the
    # supplied ``approval_callback`` returns True.
    policy = SecurityPolicy(require_confirmation=["delete_*", "refund_*"])

    async def approve(tool_def, arguments) -> bool:
        # Stand-in for an actual human-in-the-loop UI / Slack ping. Returning
        # ``True`` allows the call; ``False`` denies it.
        print(f"[approval] {tool_def.name}({arguments}) -> auto-approve")
        return True

    bridge = GantryToolBridge(
        gantry,
        security_policy=policy,
        approval_callback=approve,
    )

    # 3. Define the user query and let the bridge retrieve only the relevant
    # tools as native LangChain ``StructuredTool`` instances.
    user_query = "What's the weather in London and the stock price for AAPL?"
    tools = await bridge.get_tools(user_query, limit=2, score_threshold=0.1)
    print(f"Bridge surfaced {len(tools)} tools: {[t.name for t in tools]}")

    # 4. Run a LangGraph ReAct agent backed by the Gantry-selected tools.
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    agent = create_react_agent(llm, tools=tools)

    print("\n--- LangGraph ReAct agent with Gantry-sourced tools ---")
    result = await agent.ainvoke({"messages": [HumanMessage(content=user_query)]})
    print(f"\nFinal response: {result['messages'][-1].content}")


if __name__ == "__main__":
    asyncio.run(main())
