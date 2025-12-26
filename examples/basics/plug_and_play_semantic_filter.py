"""
Plug-and-play demo: load tools from a module and semantically filter them with
one decorator.

This shows how an existing LLM call can adopt Agent-Gantry without rewriting
tool code—just import the module and wrap the call.
"""

import asyncio
from typing import Any

from agent_gantry import AgentGantry, set_default_gantry, with_semantic_tools


async def mock_llm(prompt: str, *, tools: list[dict[str, Any]] | None = None) -> str:
    """Simulate an LLM call that receives injected tools."""
    tool_names = [t["function"]["name"] for t in tools or []]
    print(f"\n[LLM] Prompt: {prompt}")
    print(f"[LLM] Received tools: {tool_names if tool_names else 'none'}")
    return "LLM response (mock)"


async def main() -> None:
    # Drop-in: load tools that live in a separate module
    gantry = await AgentGantry.from_modules(["examples.basics.toolpack"])
    set_default_gantry(gantry)

    # Add a single decorator to inject only the relevant tools per prompt
    @with_semantic_tools(limit=2, score_threshold=0.0)
    async def chat(prompt: str, *, tools: list[dict[str, Any]] | None = None):
        return await mock_llm(prompt, tools=tools)

    # Each call receives only the tools that match the request
    await chat("Convert 10 kilometers to miles.")
    await chat("What's the current UTC time?")
    await chat("Draft a quick email to Maria about a $45 invoice.")


if __name__ == "__main__":
    asyncio.run(main())
