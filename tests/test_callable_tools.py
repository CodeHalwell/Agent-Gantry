"""Callable objects, partials, and decorated async tools share execution semantics."""

from __future__ import annotations

import asyncio
import functools
import threading
from datetime import date

import pytest
from pydantic import BaseModel

from agent_gantry import AgentGantry, ToolCall
from agent_gantry.schema.config import AgentGantryConfig, ExecutionConfig
from agent_gantry.schema.execution import ExecutionStatus
from agent_gantry.schema.introspection import build_parameters_schema


class Payload(BaseModel):
    count: int


class AsyncCallable:
    async def __call__(self, payload: Payload, when: date) -> str:
        """Read a typed payload on the requested date."""
        await asyncio.sleep(0)
        return f"{payload.count}:{when.isoformat()}"


def test_callable_instance_schema_preserves_types():
    schema = build_parameters_schema(AsyncCallable())
    assert schema["properties"]["payload"]["type"] == "object"
    assert schema["properties"]["payload"]["properties"]["count"]["type"] == "integer"
    assert schema["properties"]["when"]["format"] == "date"


def test_partial_schema_preserves_unbound_parameter_types():
    def multiply(scale: int, value: int) -> int:
        return scale * value

    schema = build_parameters_schema(functools.partial(multiply, 2))
    assert schema["properties"] == {"value": {"type": "integer"}}
    assert schema["required"] == ["value"]


async def test_async_callable_is_awaited_and_arguments_are_reconstructed():
    async with AgentGantry() as gantry:
        gantry.register(AsyncCallable(), name="read_payload")
        result = await gantry.execute(
            ToolCall(
                tool_name="read_payload",
                arguments={"payload": {"count": 3}, "when": "2026-10-16"},
            )
        )
        assert result.status == ExecutionStatus.SUCCESS
        assert result.result == "3:2026-10-16"


async def test_partial_reconstructs_model_arguments():
    def summarize(prefix: str, payload: Payload) -> str:
        return f"{prefix}:{payload.count}"

    async with AgentGantry() as gantry:
        gantry.register(functools.partial(summarize, "report"), name="summarize_payload")
        result = await gantry.execute(
            ToolCall(tool_name="summarize_payload", arguments={"payload": {"count": 4}})
        )
        assert result.status == ExecutionStatus.SUCCESS
        assert result.result == "report:4"


async def test_sync_handler_returning_awaitable_is_awaited():
    loop_thread = threading.get_ident()
    sync_threads = []

    def wrapper(value: int):
        """Wrap an async calculation with a synchronous decorator."""
        sync_threads.append(threading.get_ident())

        async def calculate():
            assert threading.get_ident() == loop_thread
            await asyncio.sleep(0)
            return value * 2

        return calculate()

    async with AgentGantry() as gantry:
        gantry.register(wrapper)
        result = await gantry.execute(ToolCall(tool_name="wrapper", arguments={"value": 3}))
        assert result.status == ExecutionStatus.SUCCESS
        assert result.result == 6
        assert sync_threads and sync_threads[0] != loop_thread


@pytest.mark.parametrize("async_object", [False, True], ids=["sync_factory", "async_callable"])
async def test_awaitable_failure_is_reported_instead_of_success(async_object):
    def fail_later():
        """Fail after the synchronous handler returns."""

        async def fail():
            raise ValueError("failure inside awaitable")

        return fail()

    class FailingCallable:
        async def __call__(self):
            return await fail_later()

    config = AgentGantryConfig(execution=ExecutionConfig(max_retries=0))
    async with AgentGantry(config=config) as gantry:
        gantry.register(FailingCallable() if async_object else fail_later, name="fail_later")
        result = await gantry.execute(ToolCall(tool_name="fail_later", arguments={}))
        assert result.status == ExecutionStatus.FAILURE
        assert result.error_type == "ValueError"
        assert "failure inside awaitable" in result.error


@pytest.mark.parametrize("async_object", [False, True], ids=["sync_factory", "async_callable"])
async def test_timeout_covers_returned_awaitable_and_cancels_it(async_object):
    cancelled = asyncio.Event()

    def wait_later():
        """Return an awaitable which waits until cancelled."""

        async def wait():
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        return wait()

    class WaitingCallable:
        async def __call__(self):
            return await wait_later()

    config = AgentGantryConfig(execution=ExecutionConfig(max_retries=0))
    async with AgentGantry(config=config) as gantry:
        gantry.register(WaitingCallable() if async_object else wait_later, name="wait_later")
        result = await gantry.execute(
            ToolCall(tool_name="wait_later", arguments={}, timeout_ms=100)
        )
        assert result.status == ExecutionStatus.TIMEOUT
        assert cancelled.is_set()


async def test_cancellation_reaches_async_callable():
    started = asyncio.Event()
    cancelled = asyncio.Event()

    class WaitingCallable:
        async def __call__(self):
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

    async with AgentGantry() as gantry:
        gantry.register(WaitingCallable(), name="wait_for_cancel")
        task = asyncio.create_task(
            gantry.execute(ToolCall(tool_name="wait_for_cancel", arguments={}))
        )
        await asyncio.wait_for(started.wait(), timeout=2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert cancelled.is_set()
