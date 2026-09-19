"""
Google ADK + Agent-Gantry: native FunctionTools from a routed catalogue.

``GoogleADKAdapter.select`` returns real ``google.adk`` FunctionTools, already
wired to execute through ``gantry.execute`` — so retries, timeouts, circuit
breakers and the security policy still apply to every call the agent makes.

Selection runs **without an API key**; only the agent run needs one
(``GOOGLE_API_KEY`` or ``GEMINI_API_KEY``).

Run with::

    pip install agent-gantry google-adk
    export GOOGLE_API_KEY=...        # only needed for the agent run
    python examples/agent_frameworks/google_adk_example.py
"""

from __future__ import annotations

import asyncio
import os

from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from agent_gantry import AgentGantry
from agent_gantry.google_adk import GoogleADKAdapter

APP_NAME = "gantry_adk_app"
USER_ID = "demo_user"
SESSION_ID = "demo_session"

USER_QUERY = "What's the status of order 12345?"


async def build_gantry() -> AgentGantry:
    gantry = AgentGantry()

    # `examples=[...]` is the text the router actually embeds — the single
    # highest-value field on a tool definition.
    @gantry.register(
        tags=["orders"],
        examples=["where is my order", "has order 12345 shipped"],
    )
    def get_order_status(order_id: str) -> dict[str, str]:
        """Look up the current status for an order ID."""
        return {"order_id": order_id, "status": "shipped", "carrier": "DHL"}

    @gantry.register(
        tags=["orders"],
        examples=["cancel my order", "stop that order going out"],
    )
    def cancel_order(order_id: str) -> str:
        """Cancel an order that has not yet dispatched."""
        return f"Cancelled {order_id}."

    @gantry.register(
        tags=["billing"],
        examples=["refund me", "I want my money back"],
    )
    def refund_order(order_id: str) -> str:
        """Issue a refund against an order."""
        return f"Refunded {order_id}."

    await gantry.sync()
    return gantry


async def run_query(query: str, gantry: AgentGantry | None = None) -> str:
    """Select tools for ``query``, build an ADK agent, and run it.

    Kept as its own function so the agent path stays callable — and testable —
    independently of ``main``'s credential check.
    """
    own_gantry = gantry is None
    gantry = gantry or await build_gantry()
    try:
        # No score_threshold: it defaults to 0.0, and raising it is a
        # silent-drop trap on longer queries.
        adk_tools = await GoogleADKAdapter(gantry).select(query, limit=1)

        adk_agent = Agent(
            model="gemini-2.5-flash",
            name="order_status_agent",
            instruction="You are a helpful agent that looks up order status via tools.",
            tools=adk_tools,
        )

        session_service = InMemorySessionService()
        await session_service.create_session(
            app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID
        )
        runner = Runner(
            agent=adk_agent, app_name=APP_NAME, session_service=session_service
        )

        events = runner.run_async(
            user_id=USER_ID,
            session_id=SESSION_ID,
            new_message=types.Content(role="user", parts=[types.Part(text=query)]),
        )

        final_text = ""
        async for event in events:
            if not event.is_final_response():
                continue
            # ADK emits control events with `content=None`, and a content event
            # can carry zero parts or a part whose `text` is None (a function
            # call, say). Reaching straight for `.parts[0].text` crashes on all
            # three, so walk the parts and take the first that has text.
            content = getattr(event, "content", None)
            parts = getattr(content, "parts", None) or []
            final_text = next(
                (p.text for p in parts if getattr(p, "text", None)),
                "",
            )
            if final_text:
                break

        return final_text
    finally:
        if own_gantry:
            await gantry.close()


async def main() -> None:
    gantry = await build_gantry()
    try:
        selected = await GoogleADKAdapter(gantry).select(USER_QUERY, limit=1)
        print(f"Catalogue: 3 tools. Gantry selected {len(selected)} for this query:")
        for tool in selected:
            print(f"  - {tool.name}")
        print("\nThe agent is built with only that slice, so it cannot call")
        print("`refund_order` for a status question.\n")

        if not (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")):
            print("Set GOOGLE_API_KEY (or GEMINI_API_KEY) to run the agent itself.")
            print("Everything above works without one.")
            return

        print("--- running the ADK agent ---")
        text = await run_query(USER_QUERY, gantry)
        print(f"\n{text or '(the model returned no text)'}")
    finally:
        await gantry.close()


if __name__ == "__main__":
    asyncio.run(main())
