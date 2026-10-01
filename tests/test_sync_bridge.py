"""The sync bridge must fan out, survive a nested invocation, and keep one loop.

``ToolSpec.invoke`` runs a coroutine from synchronous framework code (CrewAI
``_run``, Haystack ``function``, Agno ``entrypoint``, DSPy ``_fn``) on a
single long-lived bridge loop, whether or not a loop is already running on the
calling thread — so concurrent calls overlap, a handler that itself calls
``invoke`` does not wait on itself, and loop-affine state a tool opens (an MCP
session) survives from one sync call to the next.
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


def test_no_running_loop_runs_to_completion() -> None:
    """The common case — no loop on this thread — returns the coroutine's result."""

    async def work() -> int:
        return 42

    assert fw_base._run_coroutine_sync(work()) == 42


def test_both_entry_paths_share_the_bridge_loop() -> None:
    """With or without a running loop on the caller's thread, the coroutine
    runs on the one bridge loop, so resources a tool call opens survive across
    sync calls instead of dying with a per-call loop."""

    async def which_loop() -> asyncio.AbstractEventLoop:
        return asyncio.get_running_loop()

    direct = fw_base._run_coroutine_sync(which_loop())

    def from_inside_a_loop() -> object:
        async def driver() -> object:
            return fw_base._run_coroutine_sync(which_loop())

        return asyncio.run(driver())

    assert direct is _run_in_thread(from_inside_a_loop) is fw_base._bridge_loop()


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


def test_bridge_loop_is_built_once_under_concurrency() -> None:
    """The lazy construction must not race two loops into existence."""
    previous = fw_base._BRIDGE_LOOP
    fw_base._BRIDGE_LOOP = None
    seen: list[object] = []
    barrier = threading.Barrier(8)

    def grab() -> None:
        barrier.wait()
        seen.append(fw_base._bridge_loop())

    threads = [threading.Thread(target=grab) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len({id(loop) for loop in seen}) == 1
    if previous is not None:  # the replaced loop's thread is idle; let it exit
        previous.call_soon_threadsafe(previous.stop)
