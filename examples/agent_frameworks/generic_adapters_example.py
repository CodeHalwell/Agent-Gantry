"""
No adapter for your framework? Use the framework-neutral ``GantryToolset``.

Every ``<Framework>Adapter`` in this directory is a thin layer over the same
core: ``GantryToolset.select(query, limit=...)`` runs retrieval and returns
``ToolSpec`` handles. Each spec carries what any framework's tool constructor
wants — ``name``, ``description`` and a JSON-Schema ``parameters`` dict — plus
``ainvoke(**kwargs)`` / ``invoke(**kwargs)`` that execute through
``gantry.execute`` (retries, timeouts, circuit breakers, security policy).
Hand those three fields to your framework's tool type and point its callback
at ``spec.ainvoke``; that is all a dedicated adapter does.

This file needs no framework installed and **no API key**.

Run with::

    pip install agent-gantry
    python examples/agent_frameworks/generic_adapters_example.py
"""

from __future__ import annotations

import asyncio
import json

from agent_gantry import AgentGantry
from agent_gantry.integrations.frameworks import GantryToolset

USER_QUERY = "What is the market data for MSFT?"


async def main() -> None:
    gantry = AgentGantry()
    try:
        # `examples=[...]` is the text the router embeds — the single
        # highest-value field on a tool definition.
        @gantry.register(
            tags=["finance"],
            examples=["what's AAPL trading at right now", "get me live prices for TSLA"],
        )
        def get_market_data(ticker: str) -> dict:
            """Get real-time market data for a ticker."""
            return {"ticker": ticker, "price": 250.45, "volume": "1.2M"}

        @gantry.register(
            tags=["finance"],
            examples=["convert 100 dollars to euros", "how much is 50 GBP in USD"],
        )
        def convert_currency(amount: float, frm: str, to: str) -> str:
            """Convert an amount of money from one currency to another."""
            return f"{amount} {frm} = {amount * 1.1:.2f} {to}"

        @gantry.register(
            tags=["comms"],
            examples=["email the team", "send Priya a note about the report"],
        )
        def send_email(to: str, body: str) -> str:
            """Send an email message to a named recipient."""
            return f"Emailed {to}."

        await gantry.sync()

        # No score_threshold: it defaults to 0.0, and raising it is a
        # silent-drop trap on longer queries.
        specs = await GantryToolset(gantry).select(USER_QUERY, limit=1)

        print(f"Catalogue: 3 tools. Gantry selected {len(specs)} for this query:")
        for spec in specs:
            print(f"  - {spec.name}: {spec.description}")

        # Everything a framework's tool constructor needs is on the spec.
        spec = specs[0]
        print("\nJSON-Schema parameters to hand to your framework:")
        print(json.dumps(spec.parameters, indent=2))

        # The callback you wire in is `spec.ainvoke` (or `spec.invoke` from
        # sync code). It runs through gantry.execute like every adapter does.
        print("\nInvoking it through Gantry:")
        print(f"  {spec.name}(ticker='MSFT') -> {await spec.ainvoke(ticker='MSFT')}")
    finally:
        await gantry.close()


if __name__ == "__main__":
    asyncio.run(main())
