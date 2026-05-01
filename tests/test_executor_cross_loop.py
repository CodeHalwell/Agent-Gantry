"""
Cross-event-loop integration tests for the executor and rate limiter.

These tests reproduce the scenario described by integrators using
``agent_framework.DurableAIAgentWorker``: the :class:`AgentGantry` is built
in one context (often module import time, often with no running loop) and
then driven from a *different* event loop running on a worker thread. With
synchronisation primitives that bind eagerly to a loop, this previously
surfaced as ``RuntimeError: ... is bound to a different event loop`` and
the bridge wrapper turned the failure into AF's opaque
``"Error: Function failed."`` string.

The fixes exercised here:

- ``RateLimiter`` keeps one ``asyncio.Lock`` per running loop
  (:meth:`RateLimiter._lock_for_running_loop`).
- ``ExecutionEngine._execute_with_timeout`` dispatches sync handlers via
  ``asyncio.to_thread`` instead of ``asyncio.get_event_loop().run_in_executor``,
  removing the deprecated cross-loop API call.
- ``GantryToolBridge``'s wrapper catches exceptions from
  ``gantry.execute(...)`` and surfaces them as a structured JSON error,
  so any future cross-loop regression yields a debuggable error string.
"""

from __future__ import annotations

import asyncio
import json
import threading
from typing import Any

import pytest

from agent_gantry import AgentGantry
from agent_gantry.adapters.embedders.simple import SimpleEmbedder
from agent_gantry.integrations.agent_framework_bridge import GantryToolBridge
from agent_gantry.schema.execution import ToolCall


def _build_gantry_no_loop() -> AgentGantry:
    """Build a populated gantry with no running event loop.

    Mirrors the integrator's setup where ``AgentGantry`` is constructed at
    module import time so a durable worker can register the instance
    correctly. We use ``SimpleEmbedder`` to avoid touching the network.
    """
    g = AgentGantry(embedder=SimpleEmbedder())

    @g.register
    def constant_tool(seed: int) -> str:
        """Return a fixed message; verifies the executor wires up correctly."""
        return f"ok-{seed}"

    @g.register
    async def async_constant_tool(seed: int) -> str:
        """Async variant — verifies the coroutine branch of the executor."""
        await asyncio.sleep(0)
        return f"async-ok-{seed}"

    return g


def _run_in_worker_loop(coro_factory: Any) -> Any:
    """Run ``coro_factory()`` on a brand-new loop in a worker thread.

    Returns the coroutine's result (or re-raises the exception). This is
    the same shape ``DurableAIAgentWorker`` produces — a different loop
    from the one (if any) that constructed the gantry.
    """
    box: dict[str, Any] = {}

    def _runner() -> None:
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            try:
                box["result"] = loop.run_until_complete(coro_factory())
            except BaseException as exc:  # noqa: BLE001 - propagate to test
                box["error"] = exc
        finally:
            loop.close()

    t = threading.Thread(target=_runner, daemon=True)
    t.start()
    t.join(timeout=30.0)
    assert not t.is_alive(), "worker thread hung"
    if "error" in box:
        raise box["error"]
    return box["result"]


# ---------------------------------------------------------------------------
# Direct executor: gantry built without a running loop, executed from a
# worker-thread loop. Both sync and async handlers must succeed.
# ---------------------------------------------------------------------------


def test_sync_handler_executes_on_worker_loop():
    g = _build_gantry_no_loop()

    async def _do() -> Any:
        return await g.execute(
            ToolCall(tool_name="constant_tool", arguments={"seed": 7})
        )

    result = _run_in_worker_loop(_do)
    assert result.status.value == "success"
    assert result.result == "ok-7"


def test_async_handler_executes_on_worker_loop():
    g = _build_gantry_no_loop()

    async def _do() -> Any:
        return await g.execute(
            ToolCall(tool_name="async_constant_tool", arguments={"seed": 9})
        )

    result = _run_in_worker_loop(_do)
    assert result.status.value == "success"
    assert result.result == "async-ok-9"


# ---------------------------------------------------------------------------
# RateLimiter: same instance acquired/released on two distinct loops in
# sequence. With the per-loop lock fix, both succeed.
# ---------------------------------------------------------------------------


def test_rate_limiter_lock_is_bound_to_running_loop_under_contention():
    """Verify per-loop lock isolation under genuine contention.

    ``asyncio.Lock`` only binds to the current loop when contention forces
    a waiter through ``_get_loop()`` (the uncontended fast path skips it
    entirely). We reproduce that here by holding the lock across an
    ``await`` while a second task waits on it. With the old single eager
    ``asyncio.Lock``, loop B then trips
    ``RuntimeError: ... is bound to a different event loop`` on its first
    contended acquire. The per-loop fix gives each loop its own lock.
    """
    g = _build_gantry_no_loop()
    rate_limiter = g._rate_limiter
    assert rate_limiter is not None, "rate limiter should be enabled by default"

    async def _force_contention() -> str:
        lock = rate_limiter._lock_for_running_loop()

        await lock.acquire()
        # Spawn a waiter that must traverse the slow path
        # (``_get_loop().create_future()``) and bind the lock to *this*
        # loop. Without this overlap, the integrator's bug is masked.
        async def _waiter() -> None:
            await lock.acquire()
            lock.release()

        waiter_task = asyncio.create_task(_waiter())
        await asyncio.sleep(0.05)  # let waiter actually wait
        lock.release()
        await waiter_task
        return "loop-ok"

    # Loop A: bind the loop A lock under contention.
    a = asyncio.new_event_loop()
    try:
        a_result = a.run_until_complete(_force_contention())
    finally:
        a.close()
    assert a_result == "loop-ok"

    # Loop B in a worker thread — different loop, same RateLimiter
    # instance. With the old code this raises on the first contended
    # acquire because the single eager lock is still bound to (the
    # now-closed) loop A. With the per-loop fix loop B gets its own lock.
    b_result = _run_in_worker_loop(_force_contention)
    assert b_result == "loop-ok"


def test_rate_limiter_full_acquire_release_works_across_loops():
    """End-to-end ``acquire``/``release`` on the public RateLimiter API.

    Mirrors the realistic integrator path where the limiter is exercised
    indirectly via :class:`AgentGantry.execute` from a worker-thread loop.
    """
    g = _build_gantry_no_loop()
    rate_limiter = g._rate_limiter
    assert rate_limiter is not None

    async def _do() -> str:
        await rate_limiter.acquire("constant_tool", "default")
        await rate_limiter.release("constant_tool", "default")
        return "loop-ok"

    # Loop A first (no contention here — just establishes prior usage).
    a = asyncio.new_event_loop()
    try:
        assert a.run_until_complete(_do()) == "loop-ok"
    finally:
        a.close()

    # Loop B from a worker thread.
    assert _run_in_worker_loop(_do) == "loop-ok"


# ---------------------------------------------------------------------------
# Bridge wrapper: a handler that raises must surface a structured error,
# not let the exception propagate up to AF as the opaque "Function failed".
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_bridge_wrapper_surfaces_handler_exception_as_json():
    g = AgentGantry(embedder=SimpleEmbedder())

    @g.register
    def boom(x: int) -> str:
        """Always raises so the wrapper's error path is exercised."""
        raise ValueError(f"intentional failure with x={x}")

    await g.sync()

    bridge = GantryToolBridge(g, as_function_tool=False)
    tools = await bridge.get_tools("trigger boom", limit=1, score_threshold=0.0)
    assert tools, "expected at least one tool"
    boom_wrapper = next(
        t for t in tools if (getattr(t, "__name__", None) == "boom")
    )

    out = await boom_wrapper(x=3)
    payload = json.loads(out)
    assert "error" in payload
    # The actual exception text should reach the integrator's logs.
    assert "intentional failure" in payload["error"]
    assert "ValueError" in payload["error"]


@pytest.mark.asyncio
async def test_bridge_wrapper_surfaces_executor_failure_as_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exceptions raised *inside* gantry.execute (not just inside the
    handler) are also wrapped — guarding against the cross-loop scenario
    where the rate limiter or any other infrastructure layer fails before
    the handler runs."""
    g = AgentGantry(embedder=SimpleEmbedder())

    @g.register
    def some_tool(x: int) -> str:
        """A tool used to exercise the wrapper error path."""
        return str(x)

    await g.sync()

    async def _broken_execute(_call: ToolCall) -> Any:
        raise RuntimeError("simulated cross-loop failure")

    monkeypatch.setattr(g, "execute", _broken_execute)

    bridge = GantryToolBridge(g, as_function_tool=False)
    wrapper = bridge.wrap_single(g.list_tools_sync()[0])

    out = await wrapper(x=1)
    payload = json.loads(out)
    assert "error" in payload
    assert "simulated cross-loop failure" in payload["error"]
    assert "RuntimeError" in payload["error"]


# ---------------------------------------------------------------------------
# High-fidelity DurableAIAgentWorker simulation:
# ``agent_framework_durabletask.DurableAIAgentWorker`` calls
# ``asyncio.run(self._agent_entity.run(request))`` per inbound request, which
# constructs and tears down a fresh event loop every time. A module-level
# gantry therefore sees N distinct loops over its lifetime — the exact
# scenario where eager ``asyncio.Lock`` binding fails.
# ---------------------------------------------------------------------------


def test_module_level_gantry_survives_asyncio_run_per_request():
    """Smoke-test ``DurableAIAgentWorker``'s per-request ``asyncio.run``
    pattern: ``DurableAIAgentWorker.process_request`` calls
    ``asyncio.run(self._agent_entity.run(request))`` per inbound request,
    constructing and tearing down a fresh event loop every time.

    A single ``AgentGantry`` is constructed at "module load" (no running
    loop), then driven via ``asyncio.run`` repeatedly. This exercises
    the full wrapper → executor → rate-limiter path on N distinct loops.

    Note: in Python 3.11 ``asyncio.Lock`` only binds to a loop on the
    contended slow path (``_get_loop().create_future()``), so this smoke
    test alone does not reliably reproduce the cross-loop bug — see
    :func:`test_rate_limiter_lock_is_bound_to_running_loop_under_contention`
    for the targeted regression test. This one guards against new
    cross-loop primitives sneaking into the per-request path.
    """
    gantry = _build_gantry_no_loop()
    bridge = GantryToolBridge(gantry, as_function_tool=False)
    sync_wrapper = bridge.wrap_single(
        next(t for t in gantry.list_tools_sync() if t.name == "constant_tool")
    )
    async_wrapper = bridge.wrap_single(
        next(t for t in gantry.list_tools_sync() if t.name == "async_constant_tool")
    )

    async def _one_request(seed: int) -> list[str]:
        # Three concurrent invocations within a single request force
        # contention on the rate limiter's lock — the slow path that
        # actually binds ``asyncio.Lock`` to the current loop.
        outs = await asyncio.gather(
            sync_wrapper(seed=seed),
            async_wrapper(seed=seed),
            sync_wrapper(seed=seed + 1),
        )
        return list(outs)

    for i in range(5):
        results = asyncio.run(_one_request(i))
        # No JSON-error envelopes — every wrapper call must succeed cleanly.
        for r in results:
            assert not r.startswith("{") or '"error"' not in r, (
                f"iteration {i}: wrapper surfaced an error: {r}"
            )
        assert any(f"ok-{i}" in r for r in results)
        assert any(f"async-ok-{i}" in r for r in results)
