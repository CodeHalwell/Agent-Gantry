"""Omitted settings inherit configuration; explicit call settings override it."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock

import pytest

from agent_gantry import AgentGantry, ToolCall
from agent_gantry.schema.config import AgentGantryConfig, ExecutionConfig
from agent_gantry.schema.execution import ExecutionStatus

PROVIDERS = [
    "openai",
    "openai_responses",
    "anthropic",
    "gemini",
    "mistral",
    "groq",
    "agent_framework",
]


def _build_call(construction: str, **values: Any) -> ToolCall:
    if construction == "construct":
        return ToolCall.model_construct(**values)
    if construction == "copy":
        return ToolCall(tool_name="original", arguments={}).model_copy(update=values)
    return ToolCall(**values)


@pytest.mark.parametrize("construction", ["validated", "construct", "copy"])
@pytest.mark.parametrize("explicit", [False, True])
async def test_zero_retries_is_distinct_from_an_omitted_retry_count(
    explicit, construction, monkeypatch
):
    config = AgentGantryConfig(execution=ExecutionConfig(max_retries=1))
    attempts = []

    async with AgentGantry(config=config) as gantry:

        @gantry.register
        async def flaky() -> str:
            """Fail the first attempt, then return successfully."""
            attempts.append(1)
            if len(attempts) == 1:
                raise RuntimeError("try again")
            return "done"

        monkeypatch.setattr("agent_gantry.core.executor.asyncio.sleep", AsyncMock())
        call = _build_call(
            construction,
            tool_name="flaky",
            arguments={},
            **({"retry_count": 0} if explicit else {}),
        )
        result = await gantry.execute(call)
        assert len(attempts) == (1 if explicit else 2)
        assert result.status == (ExecutionStatus.FAILURE if explicit else ExecutionStatus.SUCCESS)


@pytest.mark.parametrize("construction", ["validated", "construct", "copy"])
@pytest.mark.parametrize("explicit", [False, True])
async def test_timeout_configuration_applies_unless_the_call_overrides_it(explicit, construction):
    config = AgentGantryConfig(execution=ExecutionConfig(default_timeout_ms=100, max_retries=0))
    async with AgentGantry(config=config) as gantry:

        @gantry.register
        async def slow() -> str:
            """Finish after the configured default timeout."""
            await asyncio.sleep(0.2)
            return "done"

        call = _build_call(
            construction,
            tool_name="slow",
            arguments={},
            **({"timeout_ms": 2000} if explicit else {}),
        )
        result = await gantry.execute(call)
        assert result.status == (ExecutionStatus.SUCCESS if explicit else ExecutionStatus.TIMEOUT)


async def test_sync_framework_bridge_uses_the_configured_timeout():
    config = AgentGantryConfig(execution=ExecutionConfig(default_timeout_ms=100, max_retries=0))
    async with AgentGantry(config=config) as gantry:

        @gantry.register
        async def slow() -> str:
            """Wait until execution cancels this tool."""
            await asyncio.Event().wait()
            return "unreachable"

        from agent_gantry.integrations.frameworks.base import GantryToolset, ToolExecutionError

        spec = (await GantryToolset(gantry).select("wait until cancelled", limit=1))[0]
        with pytest.raises(ToolExecutionError) as exc:
            await asyncio.to_thread(spec.invoke)
        assert exc.value.status == "timeout"


@pytest.mark.parametrize("dialect", PROVIDERS)
async def test_provider_call_conversion_inherits_retry_configuration(dialect):
    from agent_gantry.adapters.tool_spec.base import ToolCallPayload
    from agent_gantry.adapters.tool_spec.registry import get_adapter

    config = AgentGantryConfig(execution=ExecutionConfig(max_retries=1))
    attempts = []
    async with AgentGantry(config=config) as gantry:

        @gantry.register
        async def flaky() -> str:
            """Fail once and then return a value."""
            attempts.append(1)
            if len(attempts) == 1:
                raise RuntimeError("try again")
            return "done"

        call = get_adapter(dialect).to_tool_call(ToolCallPayload(tool_name="flaky"))
        result = await gantry.execute(call)
        assert result.status == ExecutionStatus.SUCCESS
        assert len(attempts) == 2


@pytest.mark.parametrize("dialect", PROVIDERS)
@pytest.mark.parametrize("override", [False, True])
async def test_round_trip_retry_configuration_and_explicit_zero(dialect, override):
    from agent_gantry.adapters.tool_spec.base import ToolCallPayload

    config = AgentGantryConfig(execution=ExecutionConfig(max_retries=1))
    attempts = []
    async with AgentGantry(config=config) as gantry:

        @gantry.register
        async def flaky() -> str:
            """Fail once and then return a value."""
            attempts.append(1)
            if len(attempts) == 1:
                raise RuntimeError("try again")
            return "done"

        replies = await gantry.execute_tool_calls(
            [ToolCallPayload(tool_name="flaky")],
            dialect=dialect,
            **({"retry_count": 0} if override else {}),
        )
        assert len(attempts) == (1 if override else 2)
        assert ("done" in str(replies)) == (not override)


@pytest.mark.parametrize("override", [False, True])
async def test_round_trip_inherits_timeout_unless_overridden(override):
    from agent_gantry.adapters.tool_spec.base import ToolCallPayload

    config = AgentGantryConfig(execution=ExecutionConfig(default_timeout_ms=100, max_retries=0))
    async with AgentGantry(config=config) as gantry:

        @gantry.register
        async def slow() -> str:
            """Return after the configured timeout."""
            await asyncio.sleep(0.2)
            return "done"

        replies = await gantry.execute_tool_calls(
            [ToolCallPayload(tool_name="slow")],
            **({"timeout_ms": 2000} if override else {}),
        )
        assert ("done" in str(replies)) == override
