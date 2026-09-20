"""
Selecting tools, skills and MCP servers with TypeSafe's Jev, instead of embeddings.

Two ways to use the same model:

1. ``JevSelector`` *replaces* semantic matching. No embedder, no vector store,
   no sync — the catalogue is put to the model and it answers which entries are
   relevant. Suits catalogues of tens to a few hundred entries, because every
   candidate is sent on every query.
2. ``JevReranker`` *refines* semantic matching, reordering the vector-search
   shortlist. Works at any catalogue size, because only the shortlist is sent.

Run with::

    pip install agent-gantry[jev]
    export TYPESAFE_API_KEY=...
    python examples/routing/jev_selection_demo.py

Without a key the script still runs: selection fails open, and you will see the
semantic path answer instead — which is the point of the fallback.
"""

from __future__ import annotations

import asyncio
import os

from agent_gantry import AgentGantry, JevReranker, JevSelector, Skill


async def build_gantry(
    selector: JevSelector | None = None,
    reranker: JevReranker | None = None,
) -> AgentGantry:
    """A small catalogue spanning two namespaces, plus one skill."""
    gantry = AgentGantry(selector=selector, reranker=reranker)
    try:

        @gantry.register(
            tags=["math"], namespace="numbers", examples=["add 3 and 4", "what is 10 plus 5"]
        )
        async def add_numbers(a: float, b: float) -> float:
            """Add two numbers together and return the sum."""
            return a + b

        @gantry.register(
            tags=["math"], namespace="numbers", examples=["divide 10 by 2", "what is 12 over 4"]
        )
        async def divide_numbers(a: float, b: float) -> float:
            """Divide the first number by the second and return the quotient."""
            return a / b

        @gantry.register(
            tags=["email"],
            namespace="comms",
            examples=["email Bob about the meeting", "send a note to the team"],
        )
        async def send_email(to: str, subject: str, body: str) -> str:
            """Send an email message to a named recipient."""
            return f"sent to {to}"

        @gantry.register(
            tags=["email"], namespace="comms", examples=["show my messages", "any new mail"]
        )
        async def list_inbox(limit: int = 10) -> list[str]:
            """List the most recent messages sitting in the inbox."""
            return ["a message"][:limit]

        await gantry.add_skill(
            Skill(
                name="refund_policy",
                description="How to decide whether a customer qualifies for a refund.",
                content="Refunds are allowed within 30 days of purchase.",
            )
        )
    except BaseException:
        await gantry.close()
        raise
    return gantry


async def demo_selection() -> None:
    """Choose tools with no embeddings involved."""
    print("\n=== selector: no embeddings, no vector search ===")
    gantry = await build_gantry(selector=JevSelector(threshold=0.3))
    try:
        for query in (
            "what's 12 divided by 4?",
            "tell Priya the report is ready",
            # Negation is where a decision model pulls ahead of cosine
            # similarity: an embedding of this sits right next to the email
            # tools it is explicitly ruling out.
            "show me my messages, but do not send anything",
        ):
            result = await gantry.retrieve(
                _query(query, limit=2),
            )
            names = [scored.tool.name for scored in result.tools]
            timing = (
                f"{result.selection_time_ms:.0f}ms selecting"
                if result.selection_time_ms is not None
                else f"{result.total_time_ms:.0f}ms (fell back to semantic search)"
            )
            print(f"  {query!r}\n    -> {names}  [{timing}]")

        skills = await gantry.retrieve_skills("can this customer get their money back?", limit=1)
        print(f"  skill -> {[s.skill.name for s in skills]}")
    finally:
        await gantry.close()


async def demo_reranking() -> None:
    """Keep semantic search, but reorder its shortlist with the same model."""
    print("\n=== reranker: semantic search first, Jev second ===")
    gantry = await build_gantry(reranker=JevReranker())
    try:
        await gantry.sync()
        tools = await gantry.retrieve_tools("what's 12 divided by 4?", limit=2)
        print(f"  -> {[t['function']['name'] for t in tools]}")
    finally:
        await gantry.close()


def _query(text: str, limit: int):
    from agent_gantry import ConversationContext, ToolQuery

    return ToolQuery(context=ConversationContext(query=text), limit=limit, score_threshold=0.0)


async def main() -> None:
    if not os.getenv("TYPESAFE_API_KEY"):
        print("TYPESAFE_API_KEY is not set — selection will fail open to semantic search.")
    await demo_selection()
    await demo_reranking()


if __name__ == "__main__":
    asyncio.run(main())
