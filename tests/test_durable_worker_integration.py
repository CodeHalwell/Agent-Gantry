"""
End-to-end integration test: real ``agent_framework`` agent driving real
gantry-wrapped tools, executed via ``asyncio.run`` per request to mirror
the loop topology of ``agent_framework_durabletask.DurableAIAgentWorker``.

``DurableAIAgentWorker._handle_request`` calls
``asyncio.run(self._agent_entity.run(request))`` for every inbound request,
which constructs and tears down a fresh event loop each time. A
module-level :class:`AgentGantry` therefore sees N distinct loops over its
lifetime — the exact topology that previously surfaced as
``"Error: Function failed."`` for integrators.

This test does the full thing without mocks of the gantry surface:

1. Build an ``AgentGantry`` at module scope (no running loop).
2. Build a real :class:`agent_framework.RawAgent` driven by a small
   :class:`agent_framework.BaseChatClient` subclass that emits one or
   more ``function_call`` items on the first turn and final text on the
   second turn — exercising AF's actual tool-invocation pipeline,
   ``GantryToolBridge``'s wrappers, the gantry executor, the rate
   limiter, and the registered handlers.
3. Drive the agent via ``asyncio.run`` repeatedly (and from a worker
   thread), asserting the tool results come back with the expected
   values rather than the AF catch-all error string.

``agent_framework`` is an optional dependency; the entire module is
``importorskip``-gated.
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any

import pytest

af = pytest.importorskip("agent_framework")

from agent_gantry import AgentGantry
from agent_gantry.adapters.embedders.simple import SimpleEmbedder
from agent_gantry.integrations.agent_framework_bridge import GantryToolBridge


class _FakeChatClient(af.FunctionInvocationLayer, af.BaseChatClient):  # type: ignore[misc]
    """Minimal ``BaseChatClient`` that drives a deterministic tool-call loop.

    First turn: emit ``num_calls`` ``function_call`` content items targeting
    ``tool_name`` with the supplied per-call arguments. Second turn: emit a
    final ``"done"`` text message. AF's ``FunctionInvocationLayer`` handles
    the round-trip in between, invoking each tool wrapper and feeding the
    ``function_result`` content back.
    """

    def __init__(self, tool_name: str, arguments_per_call: list[dict[str, Any]]) -> None:
        super().__init__()
        self._tool_name = tool_name
        self._arguments_per_call = arguments_per_call
        self._n = 0

    def _inner_get_response(  # type: ignore[override]
        self,
        *,
        messages: Any,
        stream: bool,
        options: Any,
        **kwargs: Any,
    ) -> Any:
        async def _go() -> af.ChatResponse:
            self._n += 1
            if self._n == 1:
                contents = [
                    af.Content.from_function_call(
                        call_id=f"c{i}", name=self._tool_name, arguments=args
                    )
                    for i, args in enumerate(self._arguments_per_call)
                ]
                return af.ChatResponse(
                    messages=[af.Message("assistant", contents)],
                    finish_reason="tool_calls",
                )
            return af.ChatResponse(
                messages=[af.Message("assistant", [af.Content("text", text="done")])],
                finish_reason="stop",
            )

        return _go()


# ---------------------------------------------------------------------------
# Module-scope gantry — mirrors the integrator's pattern of building the
# gantry at import time so the durable worker can register the instance.
# Tools are registered up front; ``sync()`` is called from a throwaway loop
# (also matches the documented "build at startup" pattern).
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def module_gantry() -> AgentGantry:
    g = AgentGantry(embedder=SimpleEmbedder())

    @g.register
    def add_one(x: int) -> int:
        """Returns ``x + 1``; sync handler exercising the to_thread path."""
        return x + 1

    @g.register
    async def slow_add(x: int) -> int:
        """Awaits briefly then returns ``x + 1``; async handler.

        The ``await`` makes parallel invocations actually overlap on the
        executor side, which is the realistic shape for tools that talk
        to a database or remote API.
        """
        await asyncio.sleep(0.01)
        return x + 1

    asyncio.run(g.sync())
    return g


@pytest.fixture(scope="module")
def function_tools(module_gantry: AgentGantry) -> list[Any]:
    """Pre-resolve the wrapped tools once, the way an integrator typically
    would (e.g. fetched at startup or via a context provider's
    ``before_run`` hook)."""
    bridge = GantryToolBridge(module_gantry, as_function_tool=True)
    return list(asyncio.run(bridge.get_tools("add", limit=2, score_threshold=0.0)))


def _extract_tool_results(resp: Any) -> list[Any]:
    return [
        c.result
        for m in resp.messages
        for c in m.contents
        if getattr(c, "type", None) == "function_result"
    ]


def _run_agent_request(
    tool_name: str,
    arguments_per_call: list[dict[str, Any]],
    function_tools: list[Any],
) -> list[Any]:
    """One DurableAIAgentWorker-style request: build the agent fresh,
    drive it via the fake client, return the tool results.

    Wrapped in an ``async def`` so the *caller* picks the loop policy
    (``asyncio.run`` for the in-process case, fresh loop in a thread for
    the worker-pool case).
    """
    client = _FakeChatClient(tool_name, arguments_per_call)
    agent = af.RawAgent(
        client=client, instructions="exec tools", tools=function_tools
    )

    async def _go() -> list[Any]:
        resp = await agent.run("please call the tool")
        return _extract_tool_results(resp)

    return asyncio.run(_go())


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_sync_tool_executes_across_repeated_asyncio_run(
    function_tools: list[Any],
) -> None:
    """Sequential ``asyncio.run`` requests, sync handler, single tool call.

    Each iteration tears down its event loop. The same module-level
    gantry must keep working — exercising the executor's ``to_thread``
    path on a fresh loop every time.
    """
    for i in range(5):
        results = _run_agent_request(
            "add_one", [{"x": i + 100}], function_tools=function_tools
        )
        assert results == [str(i + 101)], f"iteration {i} got {results}"


def test_async_tool_with_parallel_calls_across_loops(
    function_tools: list[Any],
) -> None:
    """Multiple parallel ``function_call`` items per request → AF dispatches
    them concurrently → real overlap on the rate limiter's lock and the
    executor pipeline. Run multiple such requests via ``asyncio.run`` so
    each request gets a fresh loop.
    """
    for i in range(3):
        seeds = list(range(i * 4, i * 4 + 4))
        results = _run_agent_request(
            "slow_add",
            [{"x": s} for s in seeds],
            function_tools=function_tools,
        )
        assert results == [str(s + 1) for s in seeds], (
            f"iteration {i} got {results}"
        )


def test_request_runs_in_worker_thread(function_tools: list[Any]) -> None:
    """``DurableAIAgentWorker`` runs requests on a thread-pool worker.
    Mirror that: spin up a worker thread, give it a fresh event loop,
    run the request via ``asyncio.run``. The module-level gantry — built
    on the main thread with no running loop — must work cleanly.
    """
    box: dict[str, Any] = {}

    def _runner() -> None:
        try:
            box["results"] = _run_agent_request(
                "slow_add",
                [{"x": 7}, {"x": 8}, {"x": 9}],
                function_tools=function_tools,
            )
        except BaseException as exc:  # noqa: BLE001
            box["error"] = exc

    t = threading.Thread(target=_runner, daemon=True)
    t.start()
    t.join(timeout=30.0)
    assert not t.is_alive(), "worker thread hung"
    assert "error" not in box, f"worker raised: {box.get('error')!r}"
    assert box["results"] == ["8", "9", "10"], box["results"]


def test_worker_thread_followed_by_main_thread(function_tools: list[Any]) -> None:
    """Cross-thread, cross-loop sequencing: run a request from a worker
    thread first (binding any cross-loop primitives to that loop), then
    run another from the main thread. Both must succeed.
    """
    # Worker thread first.
    box: dict[str, Any] = {}

    def _runner() -> None:
        box["worker"] = _run_agent_request(
            "slow_add", [{"x": 1}, {"x": 2}], function_tools=function_tools
        )

    t = threading.Thread(target=_runner, daemon=True)
    t.start()
    t.join(timeout=30.0)
    assert box.get("worker") == ["2", "3"]

    # Main thread immediately after — separate ``asyncio.run`` = separate loop.
    main_results = _run_agent_request(
        "slow_add", [{"x": 10}, {"x": 11}], function_tools=function_tools
    )
    assert main_results == ["11", "12"]


def test_no_function_failed_envelope_in_results(function_tools: list[Any]) -> None:
    """Belt-and-braces: ensure none of the returned tool results contain
    AF's catch-all ``"Error: Function failed."`` string. If a future
    regression makes ``gantry.execute`` raise, the bridge wrapper now
    surfaces a JSON ``{"error": ...}`` envelope, which would also fail
    this assertion.
    """
    results = _run_agent_request(
        "slow_add",
        [{"x": 1}, {"x": 2}, {"x": 3}, {"x": 4}],
        function_tools=function_tools,
    )
    for r in results:
        text = r if isinstance(r, str) else str(r)
        assert "Function failed" not in text, text
        assert '"error"' not in text, text
