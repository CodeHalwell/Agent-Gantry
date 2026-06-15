"""Universal native-tool adapters: select once, export to any framework.

Agent-Gantry selects a small, relevant slice of tools from a large registry.
The ``agent_gantry.integrations.frameworks`` adapters turn that slice into the
*native* tool object of whichever framework you use, so the framework can
introspect and call it — while every invocation still routes through
``gantry.execute`` (retries, timeouts, circuit breakers, security policy).

This example runs offline (no LLM). It shows:

1. The framework-neutral core: ``GantryToolset.select`` → ``ToolSpec`` → invoke.
2. The per-framework adapters: ``LangChainAdapter`` / ``CrewAIAdapter`` / … whose
   ``.select(query, limit=...)`` builds native objects when the framework is
   installed (skipped cleanly here if it is not).

Run::

    python examples/frameworks/universal_adapters_example.py
"""

from __future__ import annotations

import asyncio

from agent_gantry import AgentGantry
from agent_gantry.adapters.embedders.simple import SimpleEmbedder
from agent_gantry.integrations.frameworks import GantryToolset


def build_gantry() -> AgentGantry:
    gantry = AgentGantry(embedder=SimpleEmbedder(dimension=128))

    @gantry.register(tags=["email"])
    def send_email(to: str, subject: str = "", body: str = "") -> str:
        """Compose and send an email message to a recipient."""
        return f"email sent to {to}"

    @gantry.register(tags=["weather"])
    def get_weather(city: str) -> str:
        """Get the current weather for a city."""
        return f"It is 21C and sunny in {city}."

    @gantry.register(tags=["finance"])
    def convert_currency(amount: float, frm: str, to: str) -> str:
        """Convert an amount of money from one currency to another."""
        return f"{amount} {frm} = {amount * 1.1:.2f} {to}"

    return gantry


async def main() -> None:
    gantry = build_gantry()
    await gantry.sync()

    # 1) Framework-neutral core ------------------------------------------- #
    toolset = GantryToolset(gantry)
    specs = await toolset.select("email the report to finance", limit=2)
    print("Selected (framework-neutral) ToolSpecs:")
    for s in specs:
        print(f"  - {s.name}: {s.description!r}  (score={s.score:.3f})")

    # ToolSpec is callable directly — async or sync — through Gantry.
    top = specs[0]
    print("\nInvoke the top spec directly:")
    print("  ainvoke:", await top.ainvoke(to="finance@acme.com"))
    print("  invoke :", top.invoke(to="finance@acme.com"))  # safe inside the loop

    # 2) Export to each framework ----------------------------------------- #
    # One ``<Framework>Adapter`` class per framework; ``adapter.select(query,
    # limit=...)`` selects and builds the native tool objects in one async call.
    from agent_gantry.integrations import frameworks as F

    adapters = {
        "langchain": F.LangChainAdapter,
        "langgraph": F.LangGraphAdapter,
        "llamaindex": F.LlamaIndexAdapter,
        "crewai": F.CrewAIAdapter,
        "pydantic_ai": F.PydanticAIAdapter,
        "openai_agents": F.OpenAIAgentsAdapter,
        "smolagents": F.SmolagentsAdapter,
        "haystack": F.HaystackAdapter,
        "agno": F.AgnoAdapter,
        "autogen": F.AutoGenAdapter,
    }

    print("\nExport the selection to each framework's native tool object:")
    for name, adapter_cls in adapters.items():
        try:
            native = await adapter_cls(gantry).select("email the report to finance", limit=2)
            kind = type(native[0]).__name__ if native else "—"
            print(f"  [built] {name:<14} -> {len(native)} x {kind}")
        except ImportError:
            print(f"  [skip ] {name:<14} -> not installed (pip install it to use)")

    print(
        "\nEvery built tool calls back through gantry.execute, so retries, "
        "timeouts,\ncircuit breakers and the security policy all still apply."
    )


if __name__ == "__main__":
    asyncio.run(main())
