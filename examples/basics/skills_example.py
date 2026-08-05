"""
Semantic skill selection: procedural memory alongside tools.

Skills are registered once, retrieved by meaning per prompt, and injected
into the system prompt as contextual guidance — they are never executed.
Works out of the box with the default in-memory store (LanceDB also
supports skills for persistence).

Run: python examples/basics/skills_example.py
"""

from __future__ import annotations

import asyncio

from agent_gantry import AgentGantry, Skill, SkillCategory


async def main() -> None:
    gantry = AgentGantry()

    await gantry.add_skills(
        [
            Skill(
                name="api_pagination",
                description="How to implement cursor-based pagination for API endpoints",
                content=(
                    "Use cursor-based pagination rather than offset/limit: return an "
                    "opaque cursor with each page and accept it on the next request. "
                    "Offsets skew under concurrent writes; cursors do not."
                ),
                category=SkillCategory.HOW_TO,
                tags=["api", "pagination", "rest"],
                related_tools=["fetch_page"],
            ),
            Skill(
                name="retry_backoff",
                description="Pattern for retrying flaky network calls with exponential backoff",
                content=(
                    "Retry transient failures with exponential backoff plus jitter "
                    "(e.g. 1s, 2s, 4s, 8s). Never retry non-idempotent operations "
                    "without a dedupe key."
                ),
                category=SkillCategory.PATTERN,
                tags=["network", "retry", "resilience"],
            ),
        ]
    )

    # Retrieve the most relevant skills for a prompt...
    results = await gantry.retrieve_skills("my HTTP requests keep timing out", limit=1)
    for result in results:
        print(f"[{result.score:.2f}] {result.skill.qualified_name}")

    # ...or get them pre-formatted for system-prompt injection
    prompt_block = await gantry.retrieve_skills_as_prompt(
        "my HTTP requests keep timing out", limit=1
    )
    print()
    print(prompt_block)


if __name__ == "__main__":
    asyncio.run(main())
