"""
``GantryContextProvider`` — the AF-native way to plug Gantry into an Agent.

This example shows the *idiomatic* Microsoft Agent Framework integration:
attach Agent-Gantry as an :class:`agent_framework.ContextProvider` and let
AF call ``before_run`` on every ``agent.run(...)``. The provider extracts
the latest user message, retrieves the top-k semantically relevant tools
from Gantry, and injects them into the per-invocation
``SessionContext.tools``.

Why this matters:

* **Per-turn dynamic tool selection** — no need to pre-bake a tool list
  before constructing the agent. Each turn picks fresh tools.
* **Zero interference with AF Skills** — :class:`SkillsProvider` and
  :class:`GantryContextProvider` are sibling :class:`ContextProvider`
  instances that AF merges by ``source_id``. Both providers contribute
  tools and instructions; nobody overwrites anybody.
* **Workflow-builder compatible** — ``WorkflowBuilder``,
  ``SequentialBuilder``, ``HandoffBuilder``, ``AgentExecutor`` and
  ``WorkflowAgent`` all dispatch through ``agent.run()``, which fires the
  provider pipeline. Drop a Gantry-backed agent into any orchestration
  primitive without further changes.

The example walks through three patterns:

1. **Bare Agent** with ``GantryContextProvider`` — minimum-viable wiring.
2. **Coexistence** with :class:`SkillsProvider` — both providers attached,
   each with its own ``source_id``.
3. **Workflow** — the same provider-equipped agent dropped into
   ``SequentialBuilder``.
"""

import asyncio

from agent_framework import Agent, Skill, SkillsProvider
from agent_framework.openai import OpenAIChatClient
from agent_framework.orchestrations import SequentialBuilder
from dotenv import load_dotenv

from agent_gantry import AgentGantry, GantryContextProvider


def build_gantry() -> AgentGantry:
    """Create a Gantry instance with a few representative tools."""
    gantry = AgentGantry()

    @gantry.register
    def get_weather(city: str) -> str:
        """Get the current weather for a city."""
        return f"Weather in {city}: Sunny, 22C"

    @gantry.register
    def book_flight(origin: str, destination: str) -> str:
        """Book a flight between two cities."""
        return f"Booked flight: {origin} -> {destination}"

    @gantry.register
    def lookup_user(user_id: str) -> dict:
        """Look up a user profile from the CRM."""
        return {"id": user_id, "plan": "pro"}

    @gantry.register
    def issue_refund(user_id: str, amount: float) -> str:
        """Issue a refund to a user."""
        return f"Refunded ${amount:.2f} to {user_id}"

    return gantry


async def example_bare_agent(gantry: AgentGantry) -> None:
    """Pattern 1: minimum-viable Agent + GantryContextProvider.

    Each ``agent.run(query)`` triggers semantic retrieval against
    ``query`` and injects the top-k matching tools — *without* the caller
    pre-computing a tools list.
    """
    agent = Agent(
        OpenAIChatClient(),
        "You are a helpful concierge. Use tools to fulfil requests.",
        context_providers=[
            GantryContextProvider(gantry, top_k=3, score_threshold=0.0),
        ],
    )

    result = await agent.run("What's the weather in Tokyo?")
    print("[bare] →", result.text)


async def example_with_af_skills(gantry: AgentGantry) -> None:
    """Pattern 2: GantryContextProvider alongside AF SkillsProvider.

    Both providers are first-class ``ContextProvider`` instances, attached
    to the same agent with distinct ``source_id`` values. AF merges their
    contributions cleanly: SkillsProvider adds skill tools and prompt
    instructions, Gantry adds the dynamically retrieved tool set.

    The ``skills=True`` flag on the Gantry provider additionally pins the
    union of tools bound to *Gantry's* registered skills (separate from
    AF skills) on every run.
    """
    summarise_skill = Skill(
        name="summarise",
        description="Summarise the conversation so far.",
        content=(
            "When the user asks for a summary, produce a 3-bullet recap of "
            "the conversation."
        ),
    )

    agent = Agent(
        OpenAIChatClient(),
        "You are a customer-service assistant.",
        context_providers=[
            GantryContextProvider(
                gantry,
                top_k=3,
                score_threshold=0.0,
                skills=True,           # always include Gantry-skill-bound tools
                always_include=["lookup_user"],  # pinned utility tool
            ),
            SkillsProvider(skills=[summarise_skill]),
        ],
    )

    result = await agent.run("I want a refund of $42 on order ABC.")
    print("[skills] →", result.text)


async def example_workflow(gantry: AgentGantry) -> None:
    """Pattern 3: provider-equipped agents inside a SequentialBuilder.

    ``SequentialBuilder`` calls each agent's ``run`` in turn, which fires
    the provider pipeline on every step. The same Gantry instance can
    feed multiple agents — each one gets tools tailored to its role
    because each retrieval is driven by the previous step's output.
    """
    client = OpenAIChatClient()

    triage = Agent(
        client,
        "You triage customer requests.",
        name="Triage",
        context_providers=[GantryContextProvider(gantry, top_k=2, score_threshold=0.0)],
    )
    resolver = Agent(
        client,
        "You resolve the customer's issue using the available tools.",
        name="Resolver",
        context_providers=[GantryContextProvider(gantry, top_k=3, score_threshold=0.0)],
    )

    workflow = SequentialBuilder(participants=[triage, resolver]).build()
    result = await workflow.run("My flight LHR->NRT was cancelled, please rebook.")
    print("[workflow] →", result)


async def main() -> None:
    load_dotenv()
    gantry = build_gantry()
    await gantry.sync()

    await example_bare_agent(gantry)
    await example_with_af_skills(gantry)
    await example_workflow(gantry)


if __name__ == "__main__":
    asyncio.run(main())
