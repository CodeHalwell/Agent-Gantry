"""Real-package tests for ``GantryLiveHaystackToolInvoker.build()``.

Guards the per-call builder across the haystack 2.x -> 3.0 split: haystack
3.0 removed ``ToolInvoker`` (the ``Agent`` component owns tool execution),
so ``build()`` returns a ``ToolInvoker`` on 2.x and, on 3.x, either a
per-call ``Agent`` (when the builder was given a ``chat_generator``) or a
clear error. The stubbed suites never exercised ``build()`` against the real
package, which is how the 3.0 break slipped past them.
"""

from __future__ import annotations

import pytest

pytest.importorskip("haystack")

from agent_gantry import AgentGantry
from agent_gantry.adapters.embedders.simple import SimpleEmbedder
from agent_gantry.haystack import HaystackAdapter

try:
    from haystack.components.tools import ToolInvoker  # noqa: F401

    HAS_TOOL_INVOKER = True
except ImportError:
    HAS_TOOL_INVOKER = False


@pytest.fixture
async def gantry():
    g = AgentGantry(embedder=SimpleEmbedder(dimension=64))

    @g.register(tags=["email"])
    def send_email(to: str, body: str = "") -> str:
        "Send an email message to a recipient."
        return f"sent:{to}"

    await g.sync()
    return g


@pytest.mark.skipif(not HAS_TOOL_INVOKER, reason="haystack >= 3.0 removed ToolInvoker")
async def test_build_returns_tool_invoker_on_haystack2(gantry):
    builder = HaystackAdapter(gantry).tool_invoker_builder(limit=1, score_threshold=0.0)
    invoker = await builder.build("send an email")
    assert type(invoker).__name__ == "ToolInvoker"
    assert [t.name for t in invoker.tools] == ["send_email"]


@pytest.mark.skipif(HAS_TOOL_INVOKER, reason="haystack 2.x still has ToolInvoker")
async def test_build_without_chat_generator_raises_clear_error_on_haystack3(gantry):
    builder = HaystackAdapter(gantry).tool_invoker_builder(limit=1, score_threshold=0.0)
    with pytest.raises(RuntimeError, match="removed ToolInvoker"):
        await builder.build("send an email")


@pytest.mark.skipif(HAS_TOOL_INVOKER, reason="haystack 2.x still has ToolInvoker")
async def test_build_constructs_agent_with_chat_generator_on_haystack3(gantry):
    class _ToolFriendlyGenerator:
        """Minimal chat generator: haystack's Agent requires run() to accept tools."""

        def run(self, messages, tools=None, **kwargs):
            return {"replies": []}

    builder = HaystackAdapter(gantry).tool_invoker_builder(
        limit=1, score_threshold=0.0, chat_generator=_ToolFriendlyGenerator()
    )
    agent = await builder.build("send an email")
    assert type(agent).__name__ == "Agent"
    assert [t.name for t in agent.tools] == ["send_email"]
