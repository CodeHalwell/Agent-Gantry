"""Reverse direction: register existing framework-native tools INTO Gantry.

``universal_adapters_example.py`` (next to this file) shows the well-trodden
direction — select Gantry tools, export them as a framework's native object.
This example is the other half: you already have tools built with LangChain /
CrewAI / LlamaIndex, and you want them in Gantry's registry so they gain
semantic routing, the security policy, retries, and telemetry — and become
re-exportable to *any other* framework via the existing export adapters.
"Register once, use anywhere" only holds if tools can enter Gantry from
outside it, not just leave it.

Each framework is optional and imported lazily: if a framework isn't
installed, its section prints ``[skip]`` and the example moves on — nothing
here requires all three to be present. Install one or more to see it flip to
``[ok]``::

    pip install langchain-core crewai llama-index-core

Run::

    python examples/frameworks/importers_example.py
"""

from __future__ import annotations

import asyncio
import os

# CrewAI ships opt-out telemetry that can block for ~30s on a firewalled
# network; disable it before crewai is ever imported.
os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")

from agent_gantry import AgentGantry
from agent_gantry.adapters.embedders.simple import SimpleEmbedder
from agent_gantry.integrations.importers import (
    register_crewai_tools,
    register_langchain_tools,
    register_llamaindex_tools,
)
from agent_gantry.schema.execution import ExecutionStatus, ToolCall


def build_gantry() -> AgentGantry:
    gantry = AgentGantry(embedder=SimpleEmbedder(dimension=128))

    @gantry.register(tags=["notes"])
    def save_note(text: str) -> str:
        """Save a short text note for later reference."""
        return f"saved: {text}"

    return gantry


async def import_from_langchain(gantry: AgentGantry) -> None:
    try:
        from langchain_core.tools import tool as lc_tool
    except ImportError:
        print("  [skip] langchain-core not installed (pip install langchain-core)")
        return

    @lc_tool
    def get_weather(city: str) -> str:
        """Get the current weather for a city."""
        return f"Sunny and 21C in {city}"

    count = await register_langchain_tools(gantry, [get_weather], tags=["weather"])
    print(f"  [ok]   registered {count} LangChain tool(s): get_weather")


async def import_from_crewai(gantry: AgentGantry) -> None:
    try:
        from crewai.tools import BaseTool as CrewAIBaseTool
    except ImportError:
        print("  [skip] crewai not installed (pip install crewai)")
        return

    from pydantic import BaseModel, Field

    class ConvertArgs(BaseModel):
        amount: float = Field(description="Amount to convert")
        frm: str = Field(description="Source currency code")
        to: str = Field(description="Target currency code")

    class ConvertCurrencyTool(CrewAIBaseTool):
        name: str = "Convert Currency"
        description: str = "Convert an amount of money from one currency to another."
        args_schema: type[BaseModel] = ConvertArgs

        def _run(self, amount: float, frm: str, to: str) -> str:
            return f"{amount} {frm} = {amount * 1.1:.2f} {to}"

    count = await register_crewai_tools(gantry, [ConvertCurrencyTool()], tags=["finance"])
    print(f"  [ok]   registered {count} CrewAI tool(s): convert_currency")


async def import_from_llamaindex(gantry: AgentGantry) -> None:
    try:
        from llama_index.core.tools import FunctionTool
    except ImportError:
        print("  [skip] llama-index-core not installed (pip install llama-index-core)")
        return

    def search_flights(origin: str, destination: str) -> str:
        """Search for available flights between two airports."""
        return f"3 flights found from {origin} to {destination}"

    native = FunctionTool.from_defaults(
        fn=search_flights,
        name="search_flights",
        description="Search for available flights between two airports.",
    )
    count = await register_llamaindex_tools(gantry, [native], tags=["travel"])
    print(f"  [ok]   registered {count} LlamaIndex tool(s): search_flights")


async def main() -> None:
    gantry = build_gantry()

    print("Importing native tools from each framework (skips cleanly if not installed):")
    await import_from_langchain(gantry)
    await import_from_crewai(gantry)
    await import_from_llamaindex(gantry)

    await gantry.sync()
    print(f"\nGantry now knows {gantry.tool_count} tool(s) total (native + imported).")

    # Every imported tool is a first-class registry citizen: retrievable via
    # semantic search, transcodable to any provider dialect, and executable
    # through the normal gantry.execute() path (security policy, retries,
    # circuit breakers, telemetry) exactly like a @gantry.register-ed tool.
    print("\nRetrieve + execute an imported tool through the normal gantry.execute() path:")
    for query, tool_name, args in [
        ("what's the weather like", "get_weather", {"city": "Paris"}),
        (
            "convert money between currencies",
            "convert_currency",
            {"amount": 100, "frm": "USD", "to": "EUR"},
        ),
        ("find me a flight", "search_flights", {"origin": "LHR", "destination": "JFK"}),
    ]:
        # limit=10 (not the usual small top-k) so this stays deterministic
        # under SimpleEmbedder's coarse hash-based similarity -- a real
        # embedder would rank the right tool near the top with a much
        # smaller limit. See examples/frameworks/README.md.
        found = await gantry.retrieve_tools(query, limit=10)
        names = [t["function"]["name"] for t in found]
        if tool_name not in names:
            continue  # that framework wasn't installed, so it was never imported
        result = await gantry.execute(ToolCall(tool_name=tool_name, arguments=args))
        status = "OK" if result.status == ExecutionStatus.SUCCESS else result.status.value
        print(f"  {tool_name:<18} status={status:<8} result={result.result!r}")

    print(
        "\nEvery imported tool also re-exports to any OTHER framework via the "
        "existing adapters (agent_gantry.integrations.frameworks) -- import "
        "once, use anywhere."
    )


if __name__ == "__main__":
    asyncio.run(main())
