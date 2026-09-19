"""
Microsoft Agent Framework Harness example (AF 1.7.0+).

Demonstrates ``create_harness_agent`` — a factory that builds a pre-configured
``Agent`` with batteries included, automatically wiring up function invocation,
conversation-history persistence, context-window compaction, task tracking
(TodoProvider), agent-mode management, and OpenTelemetry observability.

.. warning::
    ``create_harness_agent`` is **experimental** in AF 1.7.0.  It is decorated
    with ``@experimental(feature_id=ExperimentalFeature.HARNESS)`` in the
    agent-framework package.  The API surface may change in future AF releases.
    Gantry's integration follows AF's experimental lifecycle; pin to
    ``agent-framework>=1.7.0,<2.0.0`` and review the AF changelog before
    upgrading.
    Source: https://github.com/microsoft/agent-framework/releases (AF 1.7.0,
    released 2026-05-28).

Integration pattern with Gantry::

    bridge = GantryToolBridge(gantry)
    tools = await bridge.get_tools("research papers summarise", limit=5)

    agent = create_harness_agent(
        client,
        max_context_window_tokens=128_000,
        max_output_tokens=8_192,
        tools=tools,
        name="ResearchAgent",
        agent_instructions="...",
    )
    result = await agent.run("Summarise the latest research on LLMs.")

The harness agent automatically chains these context providers:

- ``InMemoryHistoryProvider``  — conversation history persists across turns.
- ``CompactionProvider``       — trims context when approaching the token limit.
- ``TodoProvider``             — multi-step task tracking via a to-do list tool.
- ``AgentModeProvider``        — switches the agent between research/action/summarise modes.
- Built-in web-search tool     — injected when the client supports it.

Gantry tools are passed via ``tools=`` and merged with the built-in harness tools.

Requires: pip install "agent-gantry[agent-frameworks]"
"""

from __future__ import annotations

import asyncio
import os

from dotenv import load_dotenv

from agent_gantry import AgentGantry
from agent_gantry.integrations.agent_framework_bridge import GantryToolBridge

load_dotenv()


async def main() -> None:
    # ------------------------------------------------------------------
    # 1. Register domain tools with Agent-Gantry
    # ------------------------------------------------------------------
    gantry = AgentGantry()

    @gantry.register(
        examples=[
            "find articles about rate limiting",
            "what does the knowledge base say about retries",
        ],
    )
    def search_knowledge_base(topic: str) -> str:
        """Search the internal knowledge base for articles and technical reports."""
        return f"Found 5 articles matching '{topic}'"

    @gantry.register(
        examples=[
            "recent papers on protein folding",
            "what's new in reinforcement learning",
        ],
    )
    def get_latest_papers(field: str, max_results: int = 5) -> list[str]:
        """Retrieve the titles of the latest research papers in a scientific field."""
        return [f"{field}_paper_{i}.pdf" for i in range(1, max_results + 1)]

    @gantry.register(
        examples=[
            "summarise this report",
            "give me the key findings from that PDF",
        ],
    )
    def summarise_document(document_path: str) -> str:
        """Summarise a document and extract key findings."""
        return f"Summary of {document_path}: [key findings extracted]"

    @gantry.register(
        examples=[
            "write up a report on the findings",
            "put together a structured research report",
        ],
    )
    def create_report(title: str, sections: list[str]) -> str:
        """Create a structured research report with the specified sections."""
        return f"Report '{title}' created with {len(sections)} sections."

    await gantry.sync()

    # ------------------------------------------------------------------
    # 2. Retrieve semantically relevant tools for this task
    # ------------------------------------------------------------------
    bridge = GantryToolBridge(gantry)
    tools = await bridge.get_tools(
        "research papers summarise literature review write report",
        limit=5,
    )
    print(f"Gantry selected {len(tools)} tool(s) for the harness agent")

    # ------------------------------------------------------------------
    # 3. Build the harness agent
    #
    # create_harness_agent constructs Agent(client, instructions, ...) and
    # automatically attaches:
    #
    #   InMemoryHistoryProvider  – conversation history across turns
    #   CompactionProvider       – context-window compression
    #   TodoProvider             – step-by-step task tracking
    #   AgentModeProvider        – research / action / summarise mode
    #   Built-in web search      – added when client supports it
    #
    # Gantry tools are injected via tools=; they blend with the harness
    # built-ins.  Destructive Gantry tools automatically carry
    # approval_mode="always_require" via GantryToolBridge's capability
    # mapping, so the harness approval gate is still respected.
    # ------------------------------------------------------------------
    # `create_harness_agent` is experimental and only exists in AF >= 1.7.0.
    # The repo's `agent-frameworks` extra used to resolve AF to its 1.5.0
    # floor, which put this example permanently out of reach; that is fixed
    # from 0.17.0, where the floor is 1.19.0. The guard stays for anyone on an
    # older AF installed standalone or on a pre-0.17.0 lockfile — say so
    # plainly instead of surfacing a bare ImportError.
    try:
        from agent_framework import create_harness_agent
    except ImportError:
        import importlib.metadata as _md

        try:
            installed = _md.version("agent-framework")
        except _md.PackageNotFoundError:
            installed = "not installed"
        print(
            f"This example needs `create_harness_agent`, added in agent-framework "
            f"1.7.0; you have {installed}."
        )
        print(
            "agent-gantry 0.17.0+ requires agent-framework>=1.19.0, so syncing "
            "the extra is usually enough:"
        )
        print("    pip install -U 'agent-gantry[agent-frameworks]'")
        print("    # or, standalone:  pip install 'agent-framework>=1.7.0'")
        await gantry.close()
        return

    from agent_framework.openai import OpenAIChatClient

    if not os.getenv("OPENAI_API_KEY"):
        print("Set OPENAI_API_KEY to run the harness agent.")
        await gantry.close()
        return

    client = OpenAIChatClient()

    agent = create_harness_agent(
        client,
        max_context_window_tokens=128_000,
        max_output_tokens=8_192,
        name="GantryResearchAgent",
        description="A research assistant that searches and synthesises information.",
        agent_instructions=(
            "You are a research assistant. "
            "Use the available tools to find, summarise, and synthesise information. "
            "When asked, create a structured final report."
        ),
        tools=tools,
    )
    print(
        f"Harness agent '{agent.name}' constructed "
        f"({len(tools)} Gantry tool(s) + harness built-ins)"
    )

    # ------------------------------------------------------------------
    # 4. Run the agent — history is managed by the harness automatically
    # ------------------------------------------------------------------
    print("\n--- Turn 1 ---")
    response = await agent.run(
        "Summarise the latest research on large language models."
    )
    print(f"Response: {response}")

    print("\n--- Turn 2 (history is retained from turn 1) ---")
    response2 = await agent.run(
        "Based on what you found, create a structured report with "
        "sections: Introduction, Key Findings, and Future Directions."
    )
    print(f"Response: {response2}")


if __name__ == "__main__":
    asyncio.run(main())
