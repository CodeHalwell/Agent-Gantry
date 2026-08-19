"""Provider response -> tool calls -> execution -> formatted results.

Each dialect adapter could already parse one payload and format one result,
but nothing joined those ends: no wrapper, no facade method, and no example
used them, so every caller hand-rolled ``json.loads(tc.function.arguments)``
and the parallel-call case — the normal case for current models — was left to
each caller to discover.

Streaming was worse: nothing accumulated OpenAI ``delta.tool_calls`` fragments
or Anthropic ``input_json_delta`` at all.

Responses here are plain dicts and simple attribute holders rather than SDK
objects, since the SDKs are optional dependencies; both shapes are supported
and both are exercised.
"""

from __future__ import annotations

from typing import Any

import pytest

from agent_gantry import AgentGantry, StreamingToolCallAccumulator, extract_tool_calls
from agent_gantry.schema.tool import ToolDefinition


class _Obj:
    """Minimal attribute holder standing in for an SDK response object."""

    def __init__(self, **kwargs: Any) -> None:
        self.__dict__.update(kwargs)


# --------------------------------------------------------------------------- #
# extraction
# --------------------------------------------------------------------------- #


class TestExtractToolCalls:
    def test_openai_parallel_calls_all_survive(self) -> None:
        response = {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "function": {"name": "a", "arguments": '{"x": 1}'},
                            },
                            {
                                "id": "call_2",
                                "function": {"name": "b", "arguments": '{"y": 2}'},
                            },
                        ]
                    }
                }
            ]
        }

        calls = extract_tool_calls(response, "openai")

        assert [c.tool_name for c in calls] == ["a", "b"]
        assert [c.tool_call_id for c in calls] == ["call_1", "call_2"]
        assert calls[0].arguments == {"x": 1}

    def test_openai_sdk_style_objects_work_too(self) -> None:
        response = _Obj(
            choices=[
                _Obj(
                    message=_Obj(
                        tool_calls=[
                            _Obj(
                                id="call_9",
                                function=_Obj(name="lookup", arguments='{"q": "hi"}'),
                            )
                        ]
                    )
                )
            ]
        )

        calls = extract_tool_calls(response, "openai")

        assert calls[0].tool_name == "lookup"
        assert calls[0].arguments == {"q": "hi"}

    def test_no_tool_calls_is_empty_not_an_error(self) -> None:
        assert extract_tool_calls({"choices": [{"message": {"content": "hello"}}]}) == []
        assert extract_tool_calls({"choices": []}) == []

    def test_anthropic_tool_use_blocks(self) -> None:
        response = {
            "content": [
                {"type": "text", "text": "let me check"},
                {"type": "tool_use", "id": "toolu_1", "name": "search", "input": {"q": "x"}},
                {"type": "tool_use", "id": "toolu_2", "name": "fetch", "input": {}},
            ]
        }

        calls = extract_tool_calls(response, "anthropic")

        assert [c.tool_name for c in calls] == ["search", "fetch"]
        assert calls[0].arguments == {"q": "x"}

    def test_openai_responses_function_call_items(self) -> None:
        response = {
            "output": [
                {"type": "message", "content": "thinking"},
                {
                    "type": "function_call",
                    "call_id": "fc_1",
                    "name": "run",
                    "arguments": '{"n": 3}',
                },
            ]
        }

        calls = extract_tool_calls(response, "openai_responses")

        assert len(calls) == 1
        assert calls[0].tool_call_id == "fc_1"
        assert calls[0].arguments == {"n": 3}

    def test_gemini_function_call_parts(self) -> None:
        response = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"text": "sure"},
                            {"function_call": {"name": "go", "args": {"a": 1}, "id": "g1"}},
                        ]
                    }
                }
            ]
        }

        calls = extract_tool_calls(response, "gemini")

        assert calls[0].tool_name == "go"
        assert calls[0].arguments == {"a": 1}
        assert calls[0].tool_call_id == "g1"

    def test_unknown_dialect_is_a_clear_error(self) -> None:
        with pytest.raises(ValueError, match="No known response shape"):
            extract_tool_calls({}, "not_a_provider")

    def test_malformed_arguments_do_not_raise(self) -> None:
        response = {
            "choices": [
                {"message": {"tool_calls": [{"id": "c", "function": {"name": "t", "arguments": "{oops"}}]}}
            ]
        }

        assert extract_tool_calls(response)[0].arguments == {}


# --------------------------------------------------------------------------- #
# execute loop
# --------------------------------------------------------------------------- #


@pytest.fixture
async def gantry() -> AgentGantry:
    g = AgentGantry()

    async def weather(city: str) -> dict[str, Any]:
        return {"city": city, "temp_c": 18}

    async def explode() -> str:
        raise RuntimeError("upstream is down")

    await g.add_tool(
        ToolDefinition(
            name="get_weather",
            description="Get the current weather for a city.",
            parameters_schema={
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        ),
        handler=weather,
    )
    await g.add_tool(
        ToolDefinition(
            name="always_fails",
            description="A tool that always raises, for error-path tests.",
            parameters_schema={"type": "object", "properties": {}},
        ),
        handler=explode,
    )
    return g


def _openai_response(*calls: tuple[str, str, str]) -> dict[str, Any]:
    return {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {"id": cid, "function": {"name": name, "arguments": args}}
                        for cid, name, args in calls
                    ]
                }
            }
        ]
    }


class TestExecuteToolCalls:
    async def test_parallel_calls_are_all_executed(self, gantry: AgentGantry) -> None:
        response = _openai_response(
            ("call_1", "get_weather", '{"city": "Paris"}'),
            ("call_2", "get_weather", '{"city": "Tokyo"}'),
        )

        results = await gantry.execute_tool_calls(response)

        assert len(results) == 2
        assert [r["tool_call_id"] for r in results] == ["call_1", "call_2"]
        assert "Paris" in results[0]["content"]
        assert "Tokyo" in results[1]["content"]

    async def test_results_are_json_not_python_repr(self, gantry: AgentGantry) -> None:
        results = await gantry.execute_tool_calls(
            _openai_response(("c", "get_weather", '{"city": "Oslo"}'))
        )

        assert results[0]["content"] == '{"city": "Oslo", "temp_c": 18}'

    async def test_a_failing_tool_is_reported_not_raised(self, gantry: AgentGantry) -> None:
        """The model needs to see the failure; raising would end the turn."""
        results = await gantry.execute_tool_calls(
            _openai_response(("c", "always_fails", "{}"))
        )

        assert len(results) == 1
        assert "Error:" in results[0]["content"]

    async def test_one_failure_does_not_lose_the_other_results(
        self, gantry: AgentGantry
    ) -> None:
        results = await gantry.execute_tool_calls(
            _openai_response(
                ("ok", "get_weather", '{"city": "Rome"}'),
                ("bad", "always_fails", "{}"),
            )
        )

        assert len(results) == 2
        assert "Rome" in results[0]["content"]
        assert "Error:" in results[1]["content"]

    async def test_no_tool_calls_returns_empty(self, gantry: AgentGantry) -> None:
        assert await gantry.execute_tool_calls({"choices": [{"message": {}}]}) == []

    async def test_anthropic_dialect_formats_tool_result_blocks(
        self, gantry: AgentGantry
    ) -> None:
        response = {
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_1",
                    "name": "get_weather",
                    "input": {"city": "Lima"},
                }
            ]
        }

        results = await gantry.execute_tool_calls(response, dialect="anthropic")

        assert results[0]["type"] == "tool_result"
        assert results[0]["tool_use_id"] == "toolu_1"
        assert "is_error" not in results[0]

    async def test_anthropic_failure_sets_is_error(self, gantry: AgentGantry) -> None:
        response = {
            "content": [
                {"type": "tool_use", "id": "t", "name": "always_fails", "input": {}}
            ]
        }

        results = await gantry.execute_tool_calls(response, dialect="anthropic")

        assert results[0]["is_error"] is True

    async def test_sequential_mode_preserves_order(self, gantry: AgentGantry) -> None:
        results = await gantry.execute_tool_calls(
            _openai_response(
                ("a", "get_weather", '{"city": "A"}'),
                ("b", "get_weather", '{"city": "B"}'),
            ),
            parallel=False,
        )

        assert [r["tool_call_id"] for r in results] == ["a", "b"]

    async def test_accepts_already_extracted_payloads(self, gantry: AgentGantry) -> None:
        """The streaming path yields payloads, not a response object."""
        payloads = extract_tool_calls(
            _openai_response(("c", "get_weather", '{"city": "Bern"}'))
        )

        results = await gantry.execute_tool_calls(payloads)

        assert "Bern" in results[0]["content"]


# --------------------------------------------------------------------------- #
# streaming
# --------------------------------------------------------------------------- #


class TestStreamingAccumulation:
    def test_openai_fragments_are_reassembled(self) -> None:
        """Arguments arrive as string fragments to be concatenated per index."""
        acc = StreamingToolCallAccumulator("openai")
        chunks = [
            {"choices": [{"delta": {"tool_calls": [
                {"index": 0, "id": "call_1", "function": {"name": "get_weather", "arguments": ""}}
            ]}}]},
            {"choices": [{"delta": {"tool_calls": [
                {"index": 0, "function": {"arguments": '{"ci'}}
            ]}}]},
            {"choices": [{"delta": {"tool_calls": [
                {"index": 0, "function": {"arguments": 'ty": "Kyiv"}'}}
            ]}}]},
        ]
        for chunk in chunks:
            acc.add_chunk(chunk)

        calls = acc.tool_calls()

        assert len(calls) == 1
        assert calls[0].tool_name == "get_weather"
        assert calls[0].tool_call_id == "call_1"
        assert calls[0].arguments == {"city": "Kyiv"}

    def test_openai_parallel_streams_stay_separate(self) -> None:
        """Two interleaved calls must not have their arguments merged."""
        acc = StreamingToolCallAccumulator("openai")
        for chunk in [
            {"choices": [{"delta": {"tool_calls": [
                {"index": 0, "id": "c0", "function": {"name": "first", "arguments": '{"a"'}}
            ]}}]},
            {"choices": [{"delta": {"tool_calls": [
                {"index": 1, "id": "c1", "function": {"name": "second", "arguments": '{"b"'}}
            ]}}]},
            {"choices": [{"delta": {"tool_calls": [
                {"index": 0, "function": {"arguments": ": 1}"}}
            ]}}]},
            {"choices": [{"delta": {"tool_calls": [
                {"index": 1, "function": {"arguments": ": 2}"}}
            ]}}]},
        ]:
            acc.add_chunk(chunk)

        calls = acc.tool_calls()

        assert [c.tool_name for c in calls] == ["first", "second"]
        assert calls[0].arguments == {"a": 1}
        assert calls[1].arguments == {"b": 2}

    def test_anthropic_input_json_delta_is_reassembled(self) -> None:
        acc = StreamingToolCallAccumulator("anthropic")
        for event in [
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "tool_use", "id": "toolu_1", "name": "search"},
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "input_json_delta", "partial_json": '{"q": "ag'},
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "input_json_delta", "partial_json": 'ents"}'},
            },
            {"type": "content_block_stop", "index": 0},
        ]:
            acc.add_chunk(event)

        calls = acc.tool_calls()

        assert len(calls) == 1
        assert calls[0].tool_name == "search"
        assert calls[0].tool_call_id == "toolu_1"
        assert calls[0].arguments == {"q": "agents"}

    def test_anthropic_text_blocks_are_ignored(self) -> None:
        acc = StreamingToolCallAccumulator("anthropic")
        acc.add_chunk(
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            }
        )
        acc.add_chunk(
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "hello"},
            }
        )

        assert acc.tool_calls() == []

    def test_truncated_stream_yields_empty_arguments_not_an_exception(self) -> None:
        """A cut-off stream should fail as a tool with missing input, not a crash."""
        acc = StreamingToolCallAccumulator("openai")
        acc.add_chunk(
            {"choices": [{"delta": {"tool_calls": [
                {"index": 0, "id": "c", "function": {"name": "t", "arguments": '{"partial'}}
            ]}}]}
        )

        calls = acc.tool_calls()

        assert len(calls) == 1
        assert calls[0].arguments == {}

    def test_reset_clears_state_between_turns(self) -> None:
        acc = StreamingToolCallAccumulator("openai")
        acc.add_chunk(
            {"choices": [{"delta": {"tool_calls": [
                {"index": 0, "id": "c", "function": {"name": "t", "arguments": "{}"}}
            ]}}]}
        )
        assert acc.tool_calls()

        acc.reset()

        assert acc.tool_calls() == []

    def test_unsupported_dialect_is_rejected_up_front(self) -> None:
        with pytest.raises(ValueError, match="not implemented"):
            StreamingToolCallAccumulator("gemini")

    async def test_streamed_calls_feed_straight_into_execution(
        self, gantry: AgentGantry
    ) -> None:
        """The whole point: stream -> accumulate -> execute, with no hand-rolling."""
        acc = StreamingToolCallAccumulator("openai")
        for chunk in [
            {"choices": [{"delta": {"tool_calls": [
                {"index": 0, "id": "c", "function": {"name": "get_weather", "arguments": '{"ci'}}
            ]}}]},
            {"choices": [{"delta": {"tool_calls": [
                {"index": 0, "function": {"arguments": 'ty": "Doha"}'}}
            ]}}]},
        ]:
            acc.add_chunk(chunk)

        results = await gantry.execute_tool_calls(acc.tool_calls())

        assert "Doha" in results[0]["content"]
