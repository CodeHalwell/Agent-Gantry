"""The sync bridge must fan out, and must survive a nested invocation.

``ToolSpec.invoke`` runs a coroutine from synchronous framework code (CrewAI
``_run``, Haystack ``function``, Agno ``entrypoint``,
DSPy ``_fn``). When a loop is already running on the calling thread it hands
the coroutine to a worker thread instead.

That worker pool was ``max_workers=1`` and process-wide, so every sync tool
call in the process queued behind every other — one slow tool stalled a whole
multi-agent run — and a handler that itself called ``invoke`` waited on the
single worker it was occupying, deadlocking outright.
"""

from __future__ import annotations

import asyncio
import threading
import time

import pytest

from agent_gantry.integrations.frameworks import base as fw_base

#: Generous enough that a serialized run blows through it, tight enough that a
#: genuine deadlock fails the test rather than hanging the suite.
_TIMEOUT = 20.0


def _run_in_thread(fn, timeout: float = _TIMEOUT):
    """Run ``fn`` on a thread and fail if it does not finish in ``timeout``."""
    box: dict[str, object] = {}

    def _target() -> None:
        try:
            box["value"] = fn()
        except BaseException as exc:  # surfaced on the calling thread
            box["error"] = exc

    thread = threading.Thread(target=_target, daemon=True)
    thread.start()
    thread.join(timeout)
    if thread.is_alive():
        pytest.fail(f"timed out after {timeout}s — the sync bridge deadlocked")
    if "error" in box:
        raise box["error"]  # type: ignore[misc]
    return box.get("value")


async def test_concurrent_sync_calls_are_not_serialized() -> None:
    """Concurrent sync bridge calls must overlap, not queue behind one worker."""
    calls = 6
    sleep_for = 0.25

    async def slow() -> str:
        await asyncio.sleep(sleep_for)
        return "done"

    def blocking_call() -> object:
        return fw_base._run_coroutine_sync(slow())

    start = time.perf_counter()
    # Each to_thread hop lands on a thread with no running loop, so force the
    # bridge path by calling from inside the loop via the default executor.
    results = await asyncio.gather(
        *(asyncio.to_thread(_bridge_from_a_loop, slow) for _ in range(calls))
    )
    elapsed = time.perf_counter() - start

    assert all(r == "done" for r in results)
    # Serialized: >= calls * sleep_for. Concurrent: ~sleep_for.
    assert elapsed < calls * sleep_for * 0.6, (
        f"{calls} concurrent sync tool calls took {elapsed:.2f}s; "
        f"serialized would be ~{calls * sleep_for:.2f}s"
    )
    assert blocking_call  # keep the helper referenced for readers


def _bridge_from_a_loop(make_coro) -> object:
    """Call the bridge from a thread that *does* have a running loop.

    ``asyncio.to_thread`` hands us a bare worker thread, so start a loop here
    and invoke the bridge from inside it — the situation a framework creates
    when it calls a sync tool from its async agent loop.
    """

    async def _inner() -> object:
        # Calling the (blocking) bridge from inside a running loop is exactly
        # the case the bridge exists to handle.
        return await asyncio.to_thread(lambda: None) or _call_bridge(make_coro)

    return asyncio.run(_inner())


def _call_bridge(make_coro) -> object:
    return fw_base._run_coroutine_sync(make_coro())


def test_nested_invocation_does_not_deadlock() -> None:
    """A handler that calls back into the bridge must not wait on itself.

    With a single pooled worker this hangs forever: the nested call queues
    behind the very task that issued it.
    """

    async def inner() -> str:
        return "inner"

    async def outer() -> str:
        # Runs on a bridge worker; calling the bridge again from here is the
        # re-entrant case.
        nested = fw_base._run_coroutine_sync(inner())
        return f"outer+{nested}"

    def scenario() -> object:
        async def driver() -> object:
            return fw_base._run_coroutine_sync(outer())

        return asyncio.run(driver())

    assert _run_in_thread(scenario) == "outer+inner"


def test_no_running_loop_uses_asyncio_run() -> None:
    """The common case — no loop on this thread — needs no bridge at all."""

    async def work() -> int:
        return 42

    assert fw_base._run_coroutine_sync(work()) == 42


def test_exceptions_propagate_through_the_bridge() -> None:
    """A failing tool must raise on the caller's thread, not vanish."""

    async def boom() -> None:
        raise ValueError("tool exploded")

    def scenario() -> object:
        async def driver() -> object:
            return fw_base._run_coroutine_sync(boom())

        return asyncio.run(driver())

    with pytest.raises(ValueError, match="tool exploded"):
        _run_in_thread(scenario)


def test_pool_is_built_once_under_concurrency() -> None:
    """The lazy construction must not race two pools into existence."""
    fw_base._SYNC_BRIDGE_POOL = None
    seen: list[object] = []
    barrier = threading.Barrier(8)

    def grab() -> None:
        barrier.wait()
        seen.append(fw_base._bridge_pool())

    threads = [threading.Thread(target=grab) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len({id(pool) for pool in seen}) == 1


def test_pool_has_more_than_one_worker() -> None:
    fw_base._SYNC_BRIDGE_POOL = None
    assert fw_base._bridge_pool()._max_workers > 1
