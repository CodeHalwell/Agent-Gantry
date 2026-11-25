import asyncio
import time
from functools import partial
from unittest.mock import AsyncMock, MagicMock

import pytest

from dotfetch import DotFetch
from dotfetch.integrations.anthropic_skills import SkillsClient


@pytest.mark.asyncio
async def test_execute_tool_calls_performance():
    fetcher = MagicMock(spec=DotFetch)

    # Simulate a delay in tool execution (0.5s makes scheduling overhead negligible
    # relative to task duration, keeping the speedup ratio stable on slow runners).
    async def slow_execute(*args, **kwargs):
        await asyncio.sleep(0.5)
        mock_result = MagicMock()
        mock_result.status = "success"
        mock_result.result = "Tool executed"
        return mock_result

    fetcher.execute = AsyncMock(side_effect=slow_execute)
    # The concurrency now lives in DotFetch.execute_tool_calls, which the
    # spec'd mock would otherwise stub out; bind the real method so this
    # still measures actual fan-out.
    fetcher.execute_tool_calls = partial(DotFetch.execute_tool_calls, fetcher)

    client = SkillsClient(api_key="test-key", fetcher=fetcher)

    # Build a mock response with 10 tool-use blocks
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

    # Measure sequential baseline: call fetcher.execute 10 times in series
    seq_start = time.time()
    for block in blocks:
        await fetcher.execute(block.name, block.input)
    sequential_time = time.time() - seq_start

    # Reset the mock call count so concurrent run starts fresh
    fetcher.execute.reset_mock()

    # Measure concurrent execution via execute_tool_calls
    conc_start = time.time()
    tool_results = await client.execute_tool_calls(response)
    concurrent_time = time.time() - conc_start

    print(f"\nSequential time  (10 tools): {sequential_time:.2f}s")
    print(f"Concurrent time  (10 tools): {concurrent_time:.2f}s")
    print(f"Speedup: {sequential_time / concurrent_time:.1f}x")

    assert len(tool_results) == 10

    # Concurrent execution must be at least 5x faster than sequential.
    # This is hardware-agnostic: if the runner is slow, sequential is slow too,
    # so the ratio remains meaningful regardless of CI runner speed.
    speedup = sequential_time / concurrent_time
    assert speedup >= 5.0, (
        f"Concurrent ({concurrent_time:.2f}s) only {speedup:.1f}x faster than "
        f"sequential ({sequential_time:.2f}s). Expected >=5x speedup — "
        f"execute_tool_calls should use asyncio.gather for parallel execution."
    )
