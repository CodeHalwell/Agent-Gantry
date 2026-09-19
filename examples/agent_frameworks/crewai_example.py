"""
CrewAI + Agent-Gantry: each agent in the crew gets only the tools it needs.

CrewAI crews often share one big tool list across every agent, which means the
researcher carries the refund tool and the support agent carries the SQL tool.
Gantry lets you select per agent from the same catalogue, so each one's prompt
stays small and it cannot reach for something outside its job.

Selection runs **without an API key**; only the crew kickoff needs one.

Run with::

    pip install agent-gantry crewai langchain-openai
    export OPENAI_API_KEY=...        # only needed for the kickoff
    python examples/agent_frameworks/crewai_example.py
"""

from __future__ import annotations

import asyncio
import os

from crewai import Agent, Crew, Process, Task
from langchain_openai import ChatOpenAI

from agent_gantry import AgentGantry
from agent_gantry.crewai import CrewAIAdapter

RESEARCH_BRIEF = "Look up the customer john@example.com and their recent orders"
SUPPORT_BRIEF = "Refund the customer's last order and email them to confirm"


async def main() -> None:
    gantry = AgentGantry()
    try:
        # `examples=[...]` is the strongest routing signal a tool can carry —
        # more than the description, and more than a bigger embedding model.
        @gantry.register(
            tags=["crm"],
            examples=["look up this customer", "who is john@example.com"],
        )
        def get_customer_info(email: str) -> dict:
            """Retrieve customer details from the CRM."""
            return {"name": "John Doe", "tier": "Gold", "email": email}

        @gantry.register(
            tags=["crm"],
            examples=["what did they order", "show recent orders"],
        )
        def list_orders(email: str) -> list[str]:
            """List a customer's recent orders."""
            return ["order-4471 (Gold subscription, 79.00)"]

        @gantry.register(
            tags=["billing"],
            examples=["refund this order", "give them their money back"],
        )
        def refund_order(order_id: str) -> str:
            """Issue a refund against a customer order."""
            return f"Refunded {order_id}."

        @gantry.register(
            tags=["comms"],
            examples=["email the customer", "confirm by email"],
        )
        def send_email(to: str, body: str) -> str:
            """Send an email to a customer."""
            return f"Emailed {to}."

        await gantry.sync()

        adapter = CrewAIAdapter(gantry)

        # Two agents, one catalogue, different slices. No score_threshold: it
        # defaults to 0.0, and raising it silently drops tools on longer queries.
        research_tools = await adapter.select(RESEARCH_BRIEF, limit=2)
        support_tools = await adapter.select(SUPPORT_BRIEF, limit=2)

        print("Catalogue: 4 tools. What each crew member gets:\n")
        print(f"  researcher -> {[t.name for t in research_tools]}")
        print(f"  support    -> {[t.name for t in support_tools]}")
        print(
            "\nThe researcher never sees `refund_order`, so it cannot call it —\n"
            "selection doubles as a blast radius limit, not just a token saving.\n"
        )

        if not os.getenv("OPENAI_API_KEY"):
            print("Set OPENAI_API_KEY to run the crew itself.")
            print("Everything above works without one.")
            return

        llm = ChatOpenAI(model="gpt-5.5")
        researcher = Agent(
            role="Customer Success Researcher",
            goal="Find and summarise customer information",
            backstory="You are an expert in CRM systems and customer data.",
            tools=research_tools,
            llm=llm,
            verbose=True,
        )
        support = Agent(
            role="Customer Support Operative",
            goal="Resolve the customer's issue and confirm what was done",
            backstory="You handle refunds and customer correspondence.",
            tools=support_tools,
            llm=llm,
            verbose=True,
        )
        research_task = Task(
            description=RESEARCH_BRIEF,
            expected_output="A summary of the customer's profile and tier.",
            agent=researcher,
        )
        # Both agents run, each holding only its own slice. This is the point
        # of the example: `support` can refund because `refund_order` was
        # selected for its brief, and `researcher` cannot, because it was not
        # selected for that one.
        support_task = Task(
            description=SUPPORT_BRIEF,
            expected_output="Confirmation of the refund and the email sent.",
            agent=support,
            context=[research_task],
        )
        crew = Crew(
            agents=[researcher, support],
            tasks=[research_task, support_task],
            process=Process.sequential,
        )

        print("--- running the crew ---")
        result = await crew.kickoff_async()
        print(f"\n{result}")
    finally:
        await gantry.close()


if __name__ == "__main__":
    asyncio.run(main())
