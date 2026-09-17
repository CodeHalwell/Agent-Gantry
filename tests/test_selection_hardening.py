"""
Regression tests for the selection-layer audit.

Covers the ways a tool could fail to be selected, or a selection wrapper could
fail outright: schema property names that are not Python identifiers, message
shapes the prompt extractor mis-read, out-of-range knobs that failed on the
first turn instead of at construction, and the rate limiter's token-bucket
accounting.
"""

from __future__ import annotations

import asyncio
import time
from enum import Enum
from typing import Any
from unittest.mock import patch

import pytest

from agent_gantry import AgentGantry, with_semantic_tools
from agent_gantry.core.rate_limiter import RateLimiter, RateLimitExceeded
from agent_gantry.integrations.frameworks.base import GantryToolset
from agent_gantry.integrations.refresh import ToolRefresher
from agent_gantry.query import fallback_chain, last_user_text
from agent_gantry.query.strategies import (
    _msg_role,
    concatenate_recent,
    last_tool_result,
    latest_activity,
    tool_names_used,
    truncated,
)
from agent_gantry.schema.config import RateLimitConfig
from agent_gantry.schema.tool import ToolDefinition


@pytest.fixture
async def weather_gantry() -> AgentGantry:
    gantry = AgentGantry()

    @gantry.register(tags=["weather"])
    def get_weather(city: str) -> str:
        """Get the current weather for a city."""
        return f"sunny in {city}"

    await gantry.sync()
    return gantry


# --------------------------------------------------------------------------- #
# Schema property names that cannot be Python parameters
# --------------------------------------------------------------------------- #


class TestNonIdentifierProperties:
    @pytest.fixture
    async def toolset(self) -> GantryToolset:
        gantry = AgentGantry()
        tool = ToolDefinition(
            name="lookup_user",
            description="Look up a user record by id in the directory service.",
            parameters_schema={
                "type": "object",
                "properties": {
                    "user-id": {"type": "string"},
                    "from": {"type": "string"},
                    "_token": {"type": "string"},
                    "normal": {"type": "string"},
                },
                "required": ["user-id"],
            },
        )

        async def handler(**kwargs: Any) -> dict[str, Any]:
            return kwargs

        await gantry.add_tool(tool, handler=handler)
        return GantryToolset(gantry)

    @pytest.mark.asyncio
    async def test_a_signature_is_built_from_unusable_names(
        self, toolset: GantryToolset
    ) -> None:
        """``inspect.Parameter`` rejects ``user-id`` and ``from``; one such
        property made ``python_signature`` — and so the seven adapters built on
        ``callable_for_signature`` — raise for the whole tool."""
        spec = (await toolset.select("look up a user by id", limit=1))[0]
        signature = spec.python_signature()
        assert list(signature.parameters) == ["user_id", "from_", "token", "normal"]
        assert spec.parameter_aliases() == {
            "user_id": "user-id",
            "from_": "from",
            "token": "_token",
        }

    @pytest.mark.asyncio
    async def test_calling_by_alias_reaches_the_schema_name(
        self, toolset: GantryToolset
    ) -> None:
        """A framework calls with the aliases it read off the signature; the
        tool must receive the property names its schema declares."""
        spec = (await toolset.select("look up a user by id", limit=1))[0]
        fn = spec.callable_for_signature()
        assert await fn(user_id="u1", from_="eu", token="t", normal="n") == {
            "user-id": "u1",
            "from": "eu",
            "_token": "t",
            "normal": "n",
        }

    @pytest.mark.asyncio
    async def test_an_alias_never_shadows_a_real_property(self) -> None:
        gantry = AgentGantry()
        tool = ToolDefinition(
            name="clashing_tool",
            description="A tool whose property names collide once normalised.",
            parameters_schema={
                "type": "object",
                "properties": {"user_id": {"type": "string"}, "user-id": {"type": "string"}},
            },
        )
        await gantry.add_tool(tool, handler=lambda **kw: kw)
        spec = (await GantryToolset(gantry).select("clashing", limit=1))[0]
        assert list(spec.python_signature().parameters) == ["user_id", "user_id_2"]
        assert spec.parameter_aliases() == {"user_id_2": "user-id"}


# --------------------------------------------------------------------------- #
# Prompt extraction
# --------------------------------------------------------------------------- #


class _Message:
    """An SDK-style message object rather than a dict."""

    def __init__(self, role: str, content: Any) -> None:
        self.role = role
        self.content = content


class TestPromptExtraction:
    @pytest.mark.asyncio
    async def test_a_messages_list_with_no_user_text_selects_nothing(
        self, weather_gantry: AgentGantry
    ) -> None:
        """``str(value)`` on the list ran retrieval on ``"[]"`` or a list
        repr, so arbitrary tools were selected for a nonsense query."""
        seen: list[str] = []
        original = weather_gantry.retrieve

        async def spy(query: Any) -> Any:
            seen.append(query.context.query)
            return await original(query)

        weather_gantry.retrieve = spy  # type: ignore[method-assign]

        @with_semantic_tools(weather_gantry, prompt_param="messages", score_threshold=0.0)
        async def generate(messages: Any, *, tools: Any = None) -> Any:
            return tools

        assert await generate([]) is None
        assert await generate([{"role": "system", "content": "You are helpful."}]) is None
        assert seen == [], "no query should have been issued at all"

    @pytest.mark.asyncio
    async def test_message_objects_and_none_are_read_correctly(
        self, weather_gantry: AgentGantry
    ) -> None:
        seen: list[str] = []
        original = weather_gantry.retrieve

        async def spy(query: Any) -> Any:
            seen.append(query.context.query)
            return await original(query)

        weather_gantry.retrieve = spy  # type: ignore[method-assign]

        @with_semantic_tools(weather_gantry, prompt_param="messages", score_threshold=0.0)
        async def generate(messages: Any, *, tools: Any = None) -> Any:
            return tools

        # An object-shaped message is read like a dict one
        tools = await generate([_Message("user", "weather in Paris")])
        assert tools and tools[0]["function"]["name"] == "get_weather"
        assert seen == ["weather in Paris"]

        # An explicit None prompt must not become the query "None"
        @with_semantic_tools(weather_gantry, score_threshold=0.0)
        async def generate2(prompt: str | None = None, *, tools: Any = None) -> Any:
            return tools

        seen.clear()
        assert await generate2() is None
        assert seen == []


# --------------------------------------------------------------------------- #
# Query strategies
# --------------------------------------------------------------------------- #


class _Role(str, Enum):
    """The shape LlamaIndex ``MessageRole`` and Haystack ``ChatRole`` use."""

    USER = "user"
    TOOL = "tool"


class TestQueryStrategies:
    def test_enum_roles_are_unwrapped(self) -> None:
        """``str(Role.USER)`` is ``"_Role.USER"``, so every role check missed
        and user/tool messages were invisible to the router."""
        messages = [
            _Message(_Role.USER, "weather in Paris"),
            _Message(_Role.TOOL, "sunny 21C"),
        ]
        messages[1].name = "get_weather"

        assert _msg_role(messages[0]) == "user"
        assert _msg_role(messages[1]) == "tool"
        assert last_user_text(messages) == "weather in Paris"
        # last_tool_result names the tool it came from, as it does for dicts
        assert last_tool_result(messages) == "result of get_weather: sunny 21C"
        # latest_activity reports the newest activity's own text, unprefixed
        assert latest_activity(messages) == "sunny 21C"
        assert tool_names_used(messages) == ["get_weather"]

    def test_zero_caps_return_nothing(self) -> None:
        """``text[-0:]`` is the whole string and ``msgs[-0:]`` the whole list,
        so a zero cap returned everything."""
        assert truncated(lambda _m: "hello world", max_chars=0)(None) == ""
        messages = [{"role": "user", "content": "a"}, {"role": "user", "content": "b"}]
        assert concatenate_recent(messages, n=0) == ""
        assert concatenate_recent(messages, n=1) == "b"

    @pytest.mark.asyncio
    async def test_fallback_chain_accepts_async_generators(self) -> None:
        """Composing an async generator raised ``AttributeError`` on the
        coroutine and leaked it un-awaited."""

        async def empty(_messages: Any) -> str:
            return ""

        async def answer(_messages: Any) -> str:
            return "from async"

        chained = fallback_chain(empty, answer, last_user_text)
        assert asyncio.iscoroutinefunction(chained)
        assert await chained([{"role": "user", "content": "hi"}]) == "from async"

        # A chain of sync generators stays sync
        sync_chain = fallback_chain(lambda _m: "", last_user_text)
        assert not asyncio.iscoroutinefunction(sync_chain)
        assert sync_chain([{"role": "user", "content": "hi"}]) == "hi"

        # An async generator that yields nothing falls through to a sync one
        mixed = fallback_chain(empty, last_user_text)
        assert await mixed([{"role": "user", "content": "hi"}]) == "hi"

    @pytest.mark.asyncio
    async def test_an_async_callable_object_is_recognised(self) -> None:
        """``inspect.iscoroutinefunction`` is false for an object with
        ``async def __call__`` — the shape a stateful generator takes — so the
        sync path was built and ``.strip()`` was called on its coroutine."""

        class AsyncGenerator:
            def __init__(self, text: str) -> None:
                self._text = text

            async def __call__(self, _messages: Any) -> str:
                return self._text

        chained = fallback_chain(AsyncGenerator("from object"), last_user_text)
        assert asyncio.iscoroutinefunction(chained)
        assert await chained([{"role": "user", "content": "hi"}]) == "from object"

        # an empty one still falls through to the sync generator after it
        falls_through = fallback_chain(AsyncGenerator(""), last_user_text)
        assert await falls_through([{"role": "user", "content": "hi"}]) == "hi"

        # truncated() shares the detection
        capped = truncated(AsyncGenerator("abcdefgh"), max_chars=4)
        assert await capped([{"role": "user", "content": "hi"}]) == "efgh"

    @pytest.mark.asyncio
    async def test_an_awaitable_returned_by_a_sync_callable_is_rejected_cleanly(
        self, recwarn: pytest.WarningsRecorder
    ) -> None:
        """A plain ``def`` that hands back ``some_async_fn(...)`` cannot be
        spotted statically, so the *sync* path is built and there is no way to
        await what it returns. ``fallback_chain`` closed the value, which a
        future has no method for -- so it raised ``AttributeError`` rather than
        the explanation, and left the future dangling. ``truncated`` had no
        guard at all: the value reached ``_cap`` and failed with "object of
        type 'coroutine' has no len()", leaking the coroutine.
        """

        async def real_async(_messages: Any) -> str:
            return "async result"

        def returns_coroutine(messages: Any) -> Any:
            return real_async(messages)

        def returns_future(_messages: Any) -> Any:
            future: asyncio.Future[str] = asyncio.get_running_loop().create_future()
            future.set_result("future result")
            return future

        messages = [{"role": "user", "content": "hi"}]
        for generator in (returns_coroutine, returns_future):
            with pytest.raises(TypeError, match="no way to await it"):
                fallback_chain(generator, last_user_text)(messages)
            with pytest.raises(TypeError, match="no way to await it"):
                truncated(generator, max_chars=4)(messages)

        # ...and nothing is left un-awaited behind the error
        assert not [w for w in recwarn if "never awaited" in str(w.message)]


# --------------------------------------------------------------------------- #
# Out-of-range selection knobs
# --------------------------------------------------------------------------- #


class TestQueryBounds:
    @pytest.mark.asyncio
    async def test_the_decorator_fails_at_decoration_not_silently_every_call(
        self, weather_gantry: AgentGantry
    ) -> None:
        """``limit=60`` exceeds ``ToolQuery``'s cap; the resulting error was
        caught by the retrieval-failure handler, so the model was called with
        no tools on every request and nothing said so."""
        with pytest.raises(ValueError, match="limit"):
            with_semantic_tools(weather_gantry, limit=60)

        with pytest.raises(ValueError, match="score_threshold"):
            with_semantic_tools(weather_gantry, score_threshold=1.5)

    @pytest.mark.asyncio
    async def test_refresher_and_toolset_report_bounds_clearly(
        self, weather_gantry: AgentGantry
    ) -> None:
        with pytest.raises(ValueError, match="ToolRefresher"):
            ToolRefresher(weather_gantry, limit=60)

        with pytest.raises(ValueError, match="GantryToolset.select"):
            await GantryToolset(weather_gantry).select("weather", limit=60)

        # ``limit or default`` used to turn an explicit 0 into the default and
        # slip it past the check entirely.
        with pytest.raises(ValueError, match="limit"):
            await GantryToolset(weather_gantry).select("weather", limit=0)

    @pytest.mark.asyncio
    async def test_valid_bounds_still_work(self, weather_gantry: AgentGantry) -> None:
        assert ToolRefresher(weather_gantry, limit=50) is not None
        assert await GantryToolset(weather_gantry).select("weather", limit=1)


# --------------------------------------------------------------------------- #
# Sync durability
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_a_failed_sync_keeps_the_late_registration_pending() -> None:
    """``sync()`` drained the pending buffer before doing the work behind it.
    On a transient embedder failure the buffer was empty and ``_synced`` was
    still True from the previous run, so the new tool was never retried and
    stayed invisible even after the backend recovered."""
    gantry = AgentGantry()

    @gantry.register
    def first_tool(x: int) -> int:
        """An initial tool that syncs without trouble."""
        return x

    await gantry.sync()
    assert gantry._synced is True

    @gantry.register
    def late_tool(y: int) -> int:
        """A tool registered after the first successful sync."""
        return y

    healthy = gantry._embedder.embed_batch
    attempts = {"n": 0}

    async def flaky(texts: Any, batch_size: Any = None) -> Any:
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise RuntimeError("embedder down")
        return await healthy(texts, batch_size)

    gantry._embedder.embed_batch = flaky  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="embedder down"):
        await gantry.retrieve_tools("late tool", limit=5)

    # the registration survives, and the gantry knows it still owes a sync
    assert [tool.name for tool in gantry._pending_tools] == ["late_tool"]
    assert gantry._synced is False

    names = [t["function"]["name"] for t in await gantry.retrieve_tools("late tool", limit=5)]
    assert "late_tool" in names
    assert gantry._pending_tools == []


@pytest.mark.asyncio
async def test_a_registration_made_during_a_sync_is_not_discarded() -> None:
    """``sync()`` drained the whole pending buffer once its own work landed.

    A ``register()`` that arrived while the embedder or the store was being
    awaited is not in the snapshot the sync embedded, but it was cleared with
    it -- and ``_synced`` goes True on the way out, so ``ensure_synced`` then
    saw no work and the tool never became retrievable.
    """
    gantry = AgentGantry()

    @gantry.register
    def first_tool(x: int) -> int:
        """An initial tool, synced before the race begins."""
        return x

    await gantry.sync()

    @gantry.register
    def second_tool(x: int) -> int:
        """The tool whose sync the late registration races."""
        return x * 2

    healthy = gantry._embedder.embed_batch
    embedding = asyncio.Event()

    async def slow(texts: Any, batch_size: Any = None) -> Any:
        embedding.set()
        await asyncio.sleep(0.05)
        return await healthy(texts, batch_size)

    gantry._embedder.embed_batch = slow  # type: ignore[method-assign]

    async def register_midway() -> None:
        await embedding.wait()

        @gantry.register
        def third_tool(x: int) -> int:
            """A tool registered while the sync above is still in flight."""
            return x * 3

    await asyncio.gather(gantry.sync(), register_midway())
    gantry._embedder.embed_batch = healthy  # type: ignore[method-assign]

    # the tool the sync could not have known about is still owed
    assert [tool.name for tool in gantry._pending_tools] == ["third_tool"]

    # ...and a retrieval settles the debt on its own, via ensure_synced. The
    # assertion is on the store rather than on what comes back: which tools
    # clear a query's score threshold depends on the embedder installed, and
    # what this test is about is that the late tool gets embedded at all.
    await gantry.retrieve_tools("multiply a number", limit=10)
    stored = {tool.name for tool in await gantry._vector_store.list_all(limit=100)}
    assert stored == {"first_tool", "second_tool", "third_tool"}
    assert gantry._pending_tools == []


@pytest.mark.asyncio
async def test_a_tool_re_registered_during_a_sync_keeps_the_newer_definition() -> None:
    """Draining by qualified name would drop the update and leave the
    superseded definition in the store, so the drain matches on identity."""
    gantry = AgentGantry()

    @gantry.register
    def only_tool(x: int) -> int:
        """The original description, which the re-registration replaces."""
        return x

    healthy = gantry._embedder.embed_batch
    embedding = asyncio.Event()

    async def slow(texts: Any, batch_size: Any = None) -> Any:
        embedding.set()
        await asyncio.sleep(0.05)
        return await healthy(texts, batch_size)

    gantry._embedder.embed_batch = slow  # type: ignore[method-assign]

    async def re_register() -> None:
        await embedding.wait()

        @gantry.register(name="only_tool")
        def replacement(x: int) -> int:
            """The replacement description, registered mid-sync."""
            return x * 2

    await asyncio.gather(gantry.sync(), re_register())
    gantry._embedder.embed_batch = healthy  # type: ignore[method-assign]

    assert [tool.name for tool in gantry._pending_tools] == ["only_tool"]
    await gantry.sync()
    stored = await gantry._vector_store.list_all(limit=10)
    assert [t.description for t in stored] == [
        "The replacement description, registered mid-sync."
    ]



# --------------------------------------------------------------------------- #
# Rate limiter
# --------------------------------------------------------------------------- #


class TestPruneSafety:
    @pytest.mark.asyncio
    async def test_an_empty_registry_does_not_prune_a_shared_store(self) -> None:
        """``sync()`` pruned before checking whether anything was registered,
        so a gantry that had not registered yet asked the store to delete
        everything it did not know about -- which, on a store shared between
        gantries, is everything the others put there."""
        from agent_gantry.adapters.vector_stores.memory import InMemoryVectorStore

        shared = InMemoryVectorStore()
        producer = AgentGantry(vector_store=shared)
        for i in range(3):
            await producer.add_tool(
                ToolDefinition(
                    name=f"important_tool_{i}",
                    description=f"A tool another service registered, number {i}",
                    parameters_schema={"type": "object", "properties": {}},
                ),
                handler=lambda **kw: kw,
            )
        await producer.sync()
        stored = sorted(tool.name for tool in await shared.list_all(limit=100))
        assert len(stored) == 3

        # a second gantry on the same store, pruning on, nothing registered yet
        latecomer = AgentGantry(vector_store=shared)
        assert latecomer.export_tools() == []
        await latecomer.sync(prune=True)
        assert sorted(tool.name for tool in await shared.list_all(limit=100)) == stored

        # ...and the same through the config flag, and through the public method
        from agent_gantry.schema.config import AgentGantryConfig

        configured = AgentGantry(
            vector_store=shared, config=AgentGantryConfig(prune_on_sync=True)
        )
        await configured.sync()
        assert await configured.prune_stale_tools() == 0
        assert sorted(tool.name for tool in await shared.list_all(limit=100)) == stored
        await producer.close()

    @pytest.mark.asyncio
    async def test_pruning_still_removes_a_tool_dropped_from_the_code(self) -> None:
        """The guard above must not disable pruning when the gantry does know
        which tools belong to it."""
        from agent_gantry.adapters.vector_stores.memory import InMemoryVectorStore

        def _tool(name: str) -> ToolDefinition:
            return ToolDefinition(
                name=name,
                description=f"Tool {name} used in the prune regression check",
                parameters_schema={"type": "object", "properties": {}},
            )

        shared = InMemoryVectorStore()
        first = AgentGantry(vector_store=shared)
        for name in ("kept_one", "kept_two", "removed_from_code"):
            await first.add_tool(_tool(name), handler=lambda **kw: kw)
        await first.sync()

        # a later run of the same service, one tool deleted from the code
        second = AgentGantry(vector_store=shared)
        for name in ("kept_one", "kept_two"):
            await second.add_tool(_tool(name), handler=lambda **kw: kw)
        await second.sync(prune=True)
        assert sorted(tool.name for tool in await shared.list_all(limit=100)) == [
            "kept_one",
            "kept_two",
        ]
        # an explicit keep list prunes too
        assert await second.prune_stale_tools(keep=[_tool("kept_one")]) == 1
        assert [tool.name for tool in await shared.list_all(limit=100)] == ["kept_one"]
        await first.close()


class TestRateLimitStats:
    @pytest.mark.asyncio
    async def test_an_idle_key_does_not_report_calls_it_made_hours_ago(self) -> None:
        """``calls_last_hour`` was the raw length of the history deque, which
        is pruned only when a call is *admitted*. A key that went quiet kept
        reporting hours-old calls, while ``calls_last_minute`` -- measured
        against the clock -- correctly read 0, so the two disagreed."""
        for strategy in ("token_bucket", "fixed_window", "sliding_window"):
            limiter = RateLimiter(
                RateLimitConfig(
                    strategy=strategy,
                    max_calls_per_minute=1000,
                    max_calls_per_hour=10000,
                    max_concurrent=1000,
                )
            )
            now = time.time()
            for _ in range(3):
                await limiter.acquire("tool", "ns")
            assert limiter.get_stats("tool", "ns")["calls_last_hour"] == 3, strategy

            # two hours later, with no further calls
            with patch("time.time", return_value=now + 7200):
                idle = limiter.get_stats("tool", "ns")
            assert idle["calls_last_hour"] == 0, strategy
            assert idle["calls_last_minute"] == 0, strategy

            # ...and a window that has only partly elapsed counts what remains
            with patch("time.time", return_value=now + 120):
                partly = limiter.get_stats("tool", "ns")
            assert partly["calls_last_hour"] == 3, strategy
            assert partly["calls_last_minute"] == 0, strategy


class TestTokenBucket:
    @pytest.mark.asyncio
    async def test_a_fresh_bucket_holds_burst_size(self) -> None:
        """The bucket was seeded with the per-minute rate, so ``burst_size``
        never applied to the first burst (nor after ``reset``)."""
        limiter = RateLimiter(
            RateLimitConfig(
                strategy="token_bucket",
                max_calls_per_minute=10,
                burst_size=100,
                max_concurrent=1000,
            )
        )
        admitted = 0
        for _ in range(100):
            try:
                await limiter.acquire("tool", "ns")
            except RateLimitExceeded:
                break
            admitted += 1
        assert admitted == 100

        await limiter.reset("tool", "ns")
        assert limiter.get_stats("tool", "ns")["tokens"] == pytest.approx(100.0)

    @pytest.mark.asyncio
    async def test_a_zero_rate_refuses_instead_of_dividing_by_zero(self) -> None:
        """``retry_after = 1 / refill_rate`` raised ZeroDivisionError for the
        natural "block everything" setting."""
        limiter = RateLimiter(
            RateLimitConfig(strategy="token_bucket", max_calls_per_minute=0, max_concurrent=10)
        )
        assert limiter.would_exceed("tool", "ns") is not None
        with pytest.raises(RateLimitExceeded) as excinfo:
            await limiter.acquire("tool", "ns")
        assert excinfo.value.retry_after is None

    @pytest.mark.asyncio
    async def test_stats_count_calls_for_every_strategy(self) -> None:
        """``get_stats`` read only the sliding-window history, so token-bucket
        and fixed-window reported zero calls however many had run."""
        for strategy in ("token_bucket", "fixed_window", "sliding_window"):
            limiter = RateLimiter(
                RateLimitConfig(
                    strategy=strategy, max_calls_per_minute=60, max_concurrent=100
                )
            )
            for _ in range(5):
                await limiter.acquire("tool", "ns")
                await limiter.release("tool", "ns")
            stats = limiter.get_stats("tool", "ns")
            assert stats["calls_last_minute"] == 5, strategy
            assert stats["calls_last_hour"] == 5, strategy
