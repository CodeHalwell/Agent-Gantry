import asyncio
import time
from unittest.mock import MagicMock, AsyncMock

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
    end_time = time.time()

    execution_time = end_time - start_time
    print(f"\nExecution time for 10 tools: {execution_time:.2f} seconds")

    assert len(tool_results) == 10

    # If done sequentially, it should take ~1.0 seconds
    # If done concurrently with gather, it should take ~0.1 seconds
    assert execution_time < 0.2, f"Execution was too slow: {execution_time:.2f}s. Expected < 0.2s"
