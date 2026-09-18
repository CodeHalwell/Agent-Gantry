"""
Choosing MCP servers and Agent Skills without embeddings.

``JevSelector`` covers all three of Gantry's catalogues, not just tools. This
demo drives the two that the tool-focused demos do not show:

* ``retrieve_mcp_servers()`` — which of your connected MCP servers a prompt
  actually needs, so you connect to one instead of all of them.
* ``retrieve_skills()`` — which Agent Skills to inject into the system prompt.

Both take the same path: the catalogue goes to the model, one yes/no question
per entry, and only the entries that clear the threshold come back. Neither
needs an embedder, a vector store or a sync.

Run with::

    pip install agent-gantry[jev,mcp]
    export TYPESAFE_API_KEY=...
    python examples/protocols/jev_mcp_selection_demo.py

Without a key it still runs: selection fails open and the semantic path
answers instead, which is the behaviour you want in production.
"""

from __future__ import annotations

import asyncio
import os

from agent_gantry import AgentGantry, JevSelector, Skill

SERVERS = [
    ("filesystem", "Reads, writes and lists files on the local disk.",
     ["open a file", "what is in this folder"]),
    ("postgres", "Runs SQL queries against the production Postgres database.",
     ["how many users signed up", "query the orders table"]),
    ("slack", "Posts messages to channels and reads conversations in Slack.",
     ["tell the team", "post to #eng"]),
    ("github", "Reads and writes issues, pull requests and repository contents.",
     ["open a PR", "comment on issue 42"]),
    ("stripe", "Looks up customers, charges and refunds in Stripe.",
     ["refund this order", "find the customer's last payment"]),
]

SKILLS = [
    ("refund_policy", "How to decide whether a customer qualifies for a refund."),
    ("oncall_runbook", "What to do when the pager goes off for the payments service."),
    ("code_review", "How this team reviews pull requests and what blocks a merge."),
    ("expense_claims", "How to submit and approve an expense claim."),
]

QUERIES = [
    "this customer is asking for their money back",
    "the payments service is paging, what do I do?",
    "review the open pull request on the billing repo",
    "what's the weather in Manchester?",
]


async def main() -> None:
    if not os.getenv("TYPESAFE_API_KEY"):
        print("TYPESAFE_API_KEY is not set — selection will fail open to semantic search.\n")

    gantry = AgentGantry(selector=JevSelector(threshold=0.4))
    if gantry._mcp_registry is None:
        print("MCP support is not installed: pip install agent-gantry[mcp]")
        await gantry.close()
        return

    try:
        # Registering a server records what it is; it does not connect. That
        # matters here — the point is to decide which ones are worth
        # connecting to before paying for any of them.
        for name, description, examples in SERVERS:
            gantry.register_mcp_server(
                name,
                ["echo", name],  # a real server would be e.g. ["npx", "-y", "@mcp/server-fs"]
                description=description,
                examples=examples,
            )

        for name, description in SKILLS:
            await gantry.add_skill(
                Skill(name=name, description=description, content=f"The body of {name}.")
            )

        print("MCP servers")
        for query in QUERIES:
            servers = await gantry.retrieve_mcp_servers(query, limit=2)
            names = [server.name for server in servers] or ["(none relevant)"]
            print(f"  {query[:48]:<48} -> {names}")

        print("\nAgent Skills")
        for query in QUERIES:
            found = await gantry.retrieve_skills(query, limit=2)
            names = [f"{r.skill.name} {r.score:.2f}" for r in found] or ["(none relevant)"]
            print(f"  {query[:48]:<48} -> {names}")

        # Skills are meant to be injected verbatim, so this is the form you
        # would actually append to a system prompt.
        prompt = await gantry.retrieve_skills_as_prompt(QUERIES[0], limit=1)
        print(f"\nSystem-prompt fragment for {QUERIES[0]!r}:")
        print("  " + (prompt.strip().replace("\n", "\n  ") if prompt else "(nothing to inject)"))
    finally:
        await gantry.close()


if __name__ == "__main__":
    asyncio.run(main())
