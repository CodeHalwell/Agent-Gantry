"""
LangChain + Agent-Gantry: give the agent a slice of your catalogue, not all of it.

Uses ``langchain.agents.create_agent`` (the recommended constructor since
LangChain 1.0; the older ``langgraph.prebuilt.create_react_agent`` is deprecated
and removed in LangGraph 2.0).

This is the **static tier**: Gantry picks the relevant tools once and the agent
is built with that fixed list. LangChain fixes its tools at construction, so
per-turn re-selection lives one layer up — see ``langgraph_example.py`` for
``LangGraphAdapter.areact_agent``, which re-chooses tools on every model turn.

Everything except the final model call runs **without an API key**, so you can
see what Gantry selects before spending anything.

Run with::

    pip install agent-gantry langchain langchain-openai
    export OPENAI_API_KEY=...        # only needed for the last step
    python examples/agent_frameworks/langchain_example.py
"""

from __future__ import annotations

import asyncio
import os

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from agent_gantry import AgentGantry
from agent_gantry.langchain import LangChainAdapter

USER_QUERY = "What's the weather in London?"


async def main() -> None:
    gantry = AgentGantry()
    try:
        # `examples=[...]` is the highest-value thing you can add to a tool.
        # It is what the router embeds and what a selector reads, and on our own
        # benchmark it moved the default embedder from 1/5 to 5/5 correct — more
        # than switching to a larger embedding model did. Write the phrases a
        # user would actually type, not a restatement of the description.
        @gantry.register(
            tags=["weather"],
            examples=["what's the weather in London", "is it raining in Leeds"],
        )
        def get_weather(location: str) -> str:
            """Get the current weather in a given location."""
            return f"The weather in {location} is sunny and 25C."

        @gantry.register(
            tags=["finance"],
            examples=["what is AAPL trading at", "get me the stock price for Tesco"],
        )
        def get_stock_price(symbol: str) -> str:
            """Get the current stock price for a ticker symbol."""
            return f"The stock price for {symbol} is $150.00."

        @gantry.register(
            tags=["comms"],
            examples=["email the team", "send Priya a note about the report"],
        )
        def send_email(to: str, subject: str, body: str) -> str:
            """Send an email message to a named recipient."""
            return f"Sent '{subject}' to {to}."

        await gantry.sync()

        # One call does retrieval, conversion and execution wiring: you get back
        # native LangChain StructuredTools that `create_agent` consumes directly.
        # Note there is no `score_threshold` here — it defaults to 0.0, and
        # raising it is a silent-drop trap: a long query dilutes absolute
        # similarity, so a non-zero cutoff can quietly return nothing at all.
        tools = await LangChainAdapter(gantry).select(USER_QUERY, limit=1)

        print(f"Catalogue: 3 tools. Gantry selected {len(tools)} for this query:")
        for tool in tools:
            print(f"  - {tool.name}")
        print("\nThe other two were never converted, so their schemas never reach")
        print("the prompt. Raise `limit` to hand the agent a wider slice.\n")

        if not os.getenv("OPENAI_API_KEY"):
            print("Set OPENAI_API_KEY to run the agent itself.")
            print("Everything above works without one.")
            return

        llm = ChatOpenAI(model="gpt-5.5")
        agent = create_agent(model=llm, tools=tools)

        print("--- running the agent with the Gantry-selected slice ---")
        result = await agent.ainvoke({"messages": [HumanMessage(content=USER_QUERY)]})
        print(f"\n{result['messages'][-1].content}")
    finally:
        # Releases the vector store and any embedder resources. Examples leak
        # without this; long-running services leak worse.
        await gantry.close()


if __name__ == "__main__":
    asyncio.run(main())
