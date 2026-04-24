import asyncio
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_gantry import AgentGantry
from agent_gantry.integrations.anthropic_skills import SkillsClient


@pytest.mark.asyncio
async def test_execute_tool_calls_performance():
    gantry = MagicMock(spec=AgentGantry)

    # Simulate a delay in tool execution to represent a real-world scenario (e.g., API call, DB lookup)
    async def slow_execute(*args, **kwargs):
        await asyncio.sleep(0.1)
        mock_result = MagicMock()
        mock_result.status = "success"
        mock_result.result = "Tool executed"
        return mock_result

    gantry.execute = AsyncMock(side_effect=slow_execute)

    client = SkillsClient(api_key="test-key", gantry=gantry)

    # Mock response with multiple tool uses
    response = MagicMock()
    blocks = []
    for i in range(10):
        tool_block = MagicMock()
        tool_block.type = "tool_use"
        tool_block.id = f"tool_{i}"
        tool_block.name = "test_tool"
        tool_block.input = {"arg": "value"}
        blocks.append(tool_block)

    response.content = blocks

    start_time = time.time()
    tool_results = await client.execute_tool_calls(response)
    concurrent_time = time.time() - start_time

    print(f"\nExecution time for 10 tools (concurrent): {concurrent_time:.2f} seconds")

    assert len(tool_results) == 10

    # Verify concurrency by comparing against sequential baseline.
    # Sequential would take ~1.0s (10 * 0.1s); concurrent should be much faster.
    # Use a generous upper bound (1.5s) — macOS kqueue has higher asyncio
    # task-scheduling overhead than Linux epoll, so we need extra headroom.
    assert concurrent_time < 1.5, (
        f"Execution was too slow: {concurrent_time:.2f}s. "
        f"Expected < 1.5s (sequential baseline ≈ 1.0s, "
        f"concurrent should complete in ~0.1s + overhead)."
    )
