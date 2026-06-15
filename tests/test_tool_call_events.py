"""Tests for the framework-agnostic additions:

* ``AgentGantry.on_tool_call`` post-execution event emitter.
* ``agent_gantry.render_result`` result rendering helper.
* Library logging hygiene (NullHandler + no side-effect logger config).

None of these require an agent framework to be installed.
"""

from __future__ import annotations

import logging

import pytest

import agent_gantry as ag
from agent_gantry import AgentGantry, ToolCall, ToolCallEvent, render_result
from agent_gantry.schema.execution import BatchToolCall


@pytest.fixture
async def gantry() -> AgentGantry:
    g = AgentGantry()

    @g.register(tags=["math"], examples=["add two numbers"])
    async def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    @g.register(tags=["oops"])
    async def boom() -> str:
        """Always raises."""
        raise RuntimeError("kaboom")

    await g.sync()
    return g


class TestOnToolCall:
    @pytest.mark.asyncio
    async def test_sync_and_async_callbacks_fire(self, gantry: AgentGantry) -> None:
        seen: list[ToolCallEvent] = []
        asynced: list[ToolCallEvent] = []

        def sync_cb(ev: ToolCallEvent) -> None:
            seen.append(ev)

        async def async_cb(ev: ToolCallEvent) -> None:
            asynced.append(ev)

        gantry.on_tool_call(sync_cb)
        gantry.on_tool_call(async_cb)

        await gantry.execute(ToolCall(tool_name="add", arguments={"a": 2, "b": 3}))

        assert len(seen) == 1 and len(asynced) == 1
        ev = seen[0]
        assert ev.tool_name == "add"
        assert ev.ok is True
        assert ev.result.result == 5
        assert ev.call.arguments == {"a": 2, "b": 3}

    @pytest.mark.asyncio
    async def test_failure_still_emits_event(self, gantry: AgentGantry) -> None:
        events: list[ToolCallEvent] = []
        gantry.on_tool_call(events.append)

        await gantry.execute(ToolCall(tool_name="boom", arguments={}))

        assert len(events) == 1
        assert events[0].tool_name == "boom"
        assert events[0].ok is False

    @pytest.mark.asyncio
    async def test_broken_callback_does_not_break_execution(
        self, gantry: AgentGantry
    ) -> None:
        good: list[ToolCallEvent] = []

        def boom_cb(ev: ToolCallEvent) -> None:
            raise ValueError("bad listener")

        gantry.on_tool_call(boom_cb)
        gantry.on_tool_call(good.append)

        result = await gantry.execute(
            ToolCall(tool_name="add", arguments={"a": 1, "b": 1})
        )

        # Execution succeeds and the sibling callback still fired.
        assert result.result == 2
        assert len(good) == 1

    @pytest.mark.asyncio
    async def test_double_registration_fires_twice(self, gantry: AgentGantry) -> None:
        # Documented behaviour: registering the same callable twice fires it
        # twice; one unsubscribe removes one registration. Guards against a
        # future switch to a set-backed registry.
        calls: list[str] = []

        def cb(ev: ToolCallEvent) -> None:
            calls.append(ev.tool_name)

        gantry.on_tool_call(cb)
        unsub = gantry.on_tool_call(cb)

        await gantry.execute(ToolCall(tool_name="add", arguments={"a": 1, "b": 1}))
        assert len(calls) == 2  # fired once per registration

        unsub()  # removes one registration only
        await gantry.execute(ToolCall(tool_name="add", arguments={"a": 1, "b": 1}))
        assert len(calls) == 3

    @pytest.mark.asyncio
    async def test_unsubscribe(self, gantry: AgentGantry) -> None:
        events: list[ToolCallEvent] = []
        unsubscribe = gantry.on_tool_call(events.append)

        await gantry.execute(ToolCall(tool_name="add", arguments={"a": 1, "b": 1}))
        unsubscribe()
        await gantry.execute(ToolCall(tool_name="add", arguments={"a": 2, "b": 2}))

        assert len(events) == 1

    @pytest.mark.asyncio
    async def test_batch_emits_per_call_in_order(self, gantry: AgentGantry) -> None:
        events: list[ToolCallEvent] = []
        gantry.on_tool_call(events.append)

        await gantry.execute_batch(
            BatchToolCall(
                calls=[
                    ToolCall(tool_name="add", arguments={"a": 1, "b": 1}),
                    ToolCall(tool_name="add", arguments={"a": 5, "b": 5}),
                ],
                execution_strategy="sequential",
            )
        )

        assert [ev.result.result for ev in events] == [2, 10]
        # Each event pairs the originating call with its own result.
        assert [ev.call.arguments for ev in events] == [
            {"a": 1, "b": 1},
            {"a": 5, "b": 5},
        ]

    @pytest.mark.asyncio
    async def test_no_callbacks_is_cheap_noop(self, gantry: AgentGantry) -> None:
        # Nothing registered: execution path still works.
        result = await gantry.execute(
            ToolCall(tool_name="add", arguments={"a": 3, "b": 4})
        )
        assert result.result == 7


class TestRenderResult:
    def test_str_passthrough(self) -> None:
        assert render_result("hello") == "hello"

    def test_none_is_empty(self) -> None:
        assert render_result(None) == ""

    def test_bytes_decoded(self) -> None:
        assert render_result(b"hi") == "hi"

    def test_list_of_content_blocks(self) -> None:
        block_a = type("C", (), {"text": "hello"})()
        block_b = type("C", (), {"text": "world"})()
        assert render_result([block_a, block_b]) == "hello world"

    def test_object_with_text(self) -> None:
        block = type("C", (), {"text": "solo"})()
        assert render_result(block) == "solo"

    def test_dict_text_key(self) -> None:
        assert render_result({"text": "from-dict"}) == "from-dict"

    def test_limit_truncates_with_placeholder(self) -> None:
        out = render_result("x" * 50, limit=10)
        assert out == "x" * 10 + "…"

    def test_collapse_whitespace(self) -> None:
        assert render_result("a\n\n  b\tc", collapse_whitespace=True) == "a b c"

    def test_fallback_str(self) -> None:
        assert render_result(123) == "123"


class TestLoggingHygiene:
    def test_package_attaches_null_handler(self) -> None:
        lg = logging.getLogger("agent_gantry")
        assert any(isinstance(h, logging.NullHandler) for h in lg.handlers)

    def test_constructing_gantry_does_not_configure_logging(self) -> None:
        lg = logging.getLogger("agent_gantry")
        handlers_before = list(lg.handlers)
        level_before = lg.level

        AgentGantry()

        # Construction must not mutate the shared logger as a side effect.
        assert list(lg.handlers) == handlers_before
        assert lg.level == level_before

    def test_enable_console_logging_is_idempotent(self) -> None:
        lg = logging.getLogger("agent_gantry")
        handlers_before = list(lg.handlers)
        level_before = lg.level
        try:
            ag.enable_console_logging(logging.DEBUG)
            streams = [
                h
                for h in lg.handlers
                if isinstance(h, logging.StreamHandler)
                and not isinstance(h, logging.NullHandler)
            ]
            assert len(streams) == 1
            assert lg.level == logging.DEBUG

            # Second call must not stack another handler.
            ag.enable_console_logging(logging.INFO)
            streams2 = [
                h
                for h in lg.handlers
                if isinstance(h, logging.StreamHandler)
                and not isinstance(h, logging.NullHandler)
            ]
            assert len(streams2) == 1
            assert lg.level == logging.INFO
        finally:
            for h in list(lg.handlers):
                if isinstance(h, logging.StreamHandler) and not isinstance(
                    h, logging.NullHandler
                ):
                    lg.removeHandler(h)
            lg.setLevel(level_before)
            assert list(lg.handlers) == handlers_before

    def test_console_adapter_attach_handler_opt_in(self) -> None:
        from agent_gantry.observability.console import ConsoleTelemetryAdapter

        lg = logging.getLogger("agent_gantry")
        handlers_before = list(lg.handlers)
        level_before = lg.level
        try:
            ConsoleTelemetryAdapter(log_level=logging.DEBUG, attach_handler=True)
            streams = [
                h
                for h in lg.handlers
                if getattr(h, "_agent_gantry_console_handler", False)
            ]
            assert len(streams) == 1
            assert lg.level == logging.DEBUG
        finally:
            for h in list(lg.handlers):
                if getattr(h, "_agent_gantry_console_handler", False):
                    lg.removeHandler(h)
            lg.setLevel(level_before)
            assert list(lg.handlers) == handlers_before

    def test_enable_console_logging_with_only_file_handler(
        self, tmp_path: object
    ) -> None:
        # An app whose only handler is a FileHandler (a StreamHandler subclass)
        # must still get a real console handler — the false-positive Copilot fix.
        lg = logging.getLogger("agent_gantry")
        handlers_before = list(lg.handlers)
        level_before = lg.level
        file_handler = logging.FileHandler(str(tmp_path) + "/app.log")
        lg.addHandler(file_handler)
        try:
            ag.enable_console_logging(logging.INFO)
            console = [
                h
                for h in lg.handlers
                if getattr(h, "_agent_gantry_console_handler", False)
            ]
            assert len(console) == 1
        finally:
            for h in list(lg.handlers):
                if getattr(h, "_agent_gantry_console_handler", False) or h is file_handler:
                    lg.removeHandler(h)
            file_handler.close()
            lg.setLevel(level_before)
            assert list(lg.handlers) == handlers_before
