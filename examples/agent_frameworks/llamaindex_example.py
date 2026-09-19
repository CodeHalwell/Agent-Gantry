"""
LlamaIndex + Agent-Gantry: native FunctionTools for a ReActAgent.

``LlamaIndexAdapter.select`` runs retrieval, converts the result to real
``llama_index`` FunctionTools, and wires each one back through
``gantry.execute`` — so the agent's calls still pass through retries, timeouts,
circuit breakers and the security policy.

Selection runs **without an API key**; only the agent run needs one.

Run with::

    pip install agent-gantry llama-index llama-index-llms-openai
    export OPENAI_API_KEY=...        # only needed for the agent run
    python examples/agent_frameworks/llamaindex_example.py
"""

from __future__ import annotations

import asyncio
import os

from llama_index.core.agent.workflow import ReActAgent
from llama_index.llms.openai import OpenAI

from agent_gantry import AgentGantry
from agent_gantry.llamaindex import LlamaIndexAdapter

USER_QUERY = "What are the preferences for user dev_123?"


async def main() -> None:
    gantry = AgentGantry()
    try:
        # `examples=[...]` is what the router embeds, and it moves retrieval
        # accuracy more than switching to a larger embedding model does.
        @gantry.register(
            tags=["users"],
            examples=["what are their settings", "show me this user's preferences"],
        )
        def get_user_preferences(user_id: str) -> dict:
            """Get preferences for a specific user."""
            return {"user_id": user_id, "theme": "dark", "notifications": True}

        @gantry.register(
            tags=["users"],
            examples=["change their theme", "update notification settings"],
        )
        def update_user_preferences(user_id: str, key: str, value: str) -> str:
            """Update a single preference for a user."""
            return f"Set {key}={value} for {user_id}."

        @gantry.register(
            tags=["search"],
            examples=["find documents about", "search the knowledge base"],
        )
        def search_documents(query: str) -> str:
            """Search the document index for matching passages."""
            return f"3 passages matched '{query}'."

        await gantry.sync()

        # No score_threshold: it defaults to 0.0. Raising it is a silent-drop
        # trap — longer queries dilute absolute similarity.
        tools = await LlamaIndexAdapter(gantry).select(USER_QUERY, limit=1)

        print(f"Catalogue: 3 tools. Gantry selected {len(tools)} for this query:")
        for tool in tools:
            print(f"  - {tool.metadata.name}")
        print("\nThe read-only lookup was chosen over the writer, so this agent")
        print("cannot mutate preferences while answering a question about them.\n")

        if not os.getenv("OPENAI_API_KEY"):
            print("Set OPENAI_API_KEY to run the agent itself.")
            print("Everything above works without one.")
            return

        agent = ReActAgent(tools=tools, llm=OpenAI(model="gpt-5.5"))

        print("--- running the LlamaIndex ReActAgent ---")
        response = await agent.run(user_msg=USER_QUERY)
        print(f"\n{response}")
    finally:
        await gantry.close()


if __name__ == "__main__":
    asyncio.run(main())
