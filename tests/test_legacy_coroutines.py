"""Legacy async decorators must retain loop affinity and response accounting."""

from __future__ import annotations

import asyncio
import inspect
import threading
from unittest.mock import AsyncMock

from dotfetch import ToolCall
from dotfetch.integrations.semantic_tools import SemanticToolSelector
from dotfetch.schema.execution import ExecutionStatus


def _mark_legacy(function):
    # The marker protocol used by asgiref before inspect.markcoroutinefunction
    # became available. Exercise it even when tests run on a newer Python.
    function._is_coroutine = asyncio.coroutines._is_coroutine
    return function


async def test_legacy_async_generation_keeps_retrieval_and_usage_on_the_calling_loop(
    fetcher,
    monkeypatch,
):
    loop = asyncio.get_running_loop()
    response = {"model": "test-model", "usage": {"prompt_tokens": 10, "completion_tokens": 2}}
    tools = [{"type": "function", "function": {"name": "lookup"}}]
    selector = SemanticToolSelector(fetcher)

    async def retrieve(prompt):
        assert asyncio.get_running_loop() is loop
        assert prompt == "look up a record"
        return tools

    monkeypatch.setattr(selector, "_retrieve_tools", retrieve)
    usage = AsyncMock()
    monkeypatch.setattr(fetcher.telemetry, "record_token_usage", usage)

    @_mark_legacy
    def generate(prompt: str, *, tools=None):
        async def complete():
            assert asyncio.get_running_loop() is loop
            assert tools == [{"type": "function", "function": {"name": "lookup"}}]
            return response

        return complete()

    wrapped = selector(generate)
    assert inspect.iscoroutinefunction(wrapped)
    assert await wrapped("look up a record") is response
    usage.assert_awaited_once()
    assert usage.await_args.args[0].prompt_tokens == 10
    assert usage.await_args.kwargs["model_name"] == "test-model"


async def test_legacy_handler_starts_on_the_calling_thread(fetcher):
    loop_thread = threading.get_ident()

    @_mark_legacy
    def double(value: int):
        """A marked wrapper that must be invoked like an async handler."""
        assert threading.get_ident() == loop_thread

        async def complete():
            return value * 2

        return complete()

    fetcher.register(double)
    result = await fetcher.execute(ToolCall(tool_name="double", arguments={"value": 3}))
    assert result.status == ExecutionStatus.SUCCESS
    assert result.result == 6


async def test_legacy_health_check_is_awaited(fetcher, monkeypatch):
    called = []

    @_mark_legacy
    def health_check():
        async def complete():
            called.append(True)
            return False

        return complete()

    monkeypatch.setattr(fetcher.telemetry, "health_check", health_check)
    health = await fetcher.health_check()
    assert health["telemetry"] is False
    assert called == [True]
