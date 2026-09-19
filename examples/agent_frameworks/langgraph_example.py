"""
LangGraph + Agent-Gantry: tools re-chosen on every model turn.

This is the **dynamic tier**, and it is what LangGraph gives you that plain
LangChain does not. ``LangGraphAdapter.areact_agent`` installs middleware that
re-runs selection before each model call, so a conversation that starts about
the weather and turns into a refund request gets different tools on turn two —
without you rebuilding the agent.

Compare ``langchain_example.py``, where the tool list is fixed at construction.

The agent build and the selection both run **without an API key**; only the
model call needs one.

Run with::

    pip install agent-gantry langchain langgraph langchain-openai
    export OPENAI_API_KEY=...        # only needed for the last step
    python examples/agent_frameworks/langgraph_example.py
"""

from __future__ import annotations

import asyncio
import os

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from agent_gantry import AgentGantry
from agent_gantry.langgraph import LangGraphAdapter

# Two turns that want different tools. That contrast is the whole demo.
TURNS = [
    "How does Agent-Gantry route tools?",
    "Actually, refund order 4471 and email the customer to confirm.",
]


async def main() -> None:
    gantry = AgentGantry()
    try:
        # `examples=[...]` carries more routing weight than anything else you
        # can add here — it is the text the router embeds. Use the phrasing a
        # user would type.
        @gantry.register(
            tags=["docs"],
            examples=["how does the router work", "search the internal docs"],
        )
        def search_docs(query: str) -> str:
            """Search internal documentation about how Agent-Gantry works."""
            return f"Docs for '{query}': Gantry retrieves the top-k relevant tools."

        @gantry.register(
            tags=["billing"],
            examples=["refund this order", "give the customer their money back"],
        )
        def refund_order(order_id: str) -> str:
            """Issue a refund against a customer order."""
            return f"Refunded order {order_id}."

        @gantry.register(
            tags=["comms"],
            examples=["email the customer", "let them know by email"],
        )
        def send_email(to: str, body: str) -> str:
            """Send an email to a customer."""
            return f"Emailed {to}."

        @gantry.register(
            tags=["analytics"],
            examples=["how many signups last month", "query the warehouse"],
        )
        def run_query(sql: str) -> str:
            """Run a read-only SQL query against the analytics warehouse."""
            return "42"

        await gantry.sync()

        adapter = LangGraphAdapter(gantry)

        # What each turn would get, before any model is involved. This is the
        # selection the middleware performs internally on every model call.
        print("Catalogue: 4 tools. What each turn selects:\n")
        for turn in TURNS:
            chosen = await adapter.select(turn, limit=2)
            names = ", ".join(t.name for t in chosen)
            print(f"  {turn[:52]:<52} -> {names}")
        print(
            "\nDifferent turns, different tools — that is the point of the\n"
            "dynamic tier. A statically-built agent would carry all four\n"
            "schemas through both turns.\n"
            "\nNote turn one also pulled in `send_email`: top-k always returns\n"
            "k, so with limit=2 the runner-up comes along whether or not it is\n"
            "wanted. Lower `limit`, or configure a selector, if you want the\n"
            "catalogue to be able to answer 'none of these'.\n"
        )

        if not os.getenv("OPENAI_API_KEY"):
            print("Set OPENAI_API_KEY to run the agent itself.")
            print("Everything above works without one.")
            return

        # areact_agent installs the selection middleware. `limit` bounds how
        # many tools reach the model on any given turn.
        agent = await adapter.areact_agent(ChatOpenAI(model="gpt-5.5"), limit=2)

        print("--- running the agent; tools are re-selected per turn ---")
        result = await agent.ainvoke({"messages": [HumanMessage(content=TURNS[0])]})
        print(f"\n{result['messages'][-1].content}")
    finally:
        await gantry.close()


if __name__ == "__main__":
    asyncio.run(main())
