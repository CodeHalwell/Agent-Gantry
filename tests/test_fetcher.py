"""
Tests for the DotFetch main facade.
"""

from __future__ import annotations

import pytest

from dotfetch import DotFetch, create_default_fetcher


class TestCreateDefaultDotFetch:
    """Tests for create_default_fetcher factory function."""

    def test_create_default_fetcher_returns_instance(self) -> None:
        """Test that create_default_fetcher returns a DotFetch instance."""
        fetcher = create_default_fetcher()
        assert isinstance(fetcher, DotFetch)
        assert fetcher is not None
        assert fetcher.tool_count == 0

    def test_create_default_fetcher_with_dimension(self) -> None:
        """Test create_default_fetcher with custom dimension parameter."""
        fetcher = create_default_fetcher(dimension=512)
        assert isinstance(fetcher, DotFetch)
        assert fetcher is not None

    def test_create_default_fetcher_multiple_instances(self) -> None:
        """Test that create_default_fetcher creates independent instances."""
        fetcher1 = create_default_fetcher()
        fetcher2 = create_default_fetcher()
        assert fetcher1 is not fetcher2

        # Register tool on one instance
        @fetcher1.register
        def test_tool(x: int) -> int:
            """A test tool."""
            return x * 2

        # Should not affect the other instance
        assert fetcher1.tool_count == 1
        assert fetcher2.tool_count == 0

    def test_create_default_fetcher_can_register_tools(self) -> None:
        """Test that fetcher from create_default_fetcher can register tools."""
        fetcher = create_default_fetcher()

        @fetcher.register(tags=["math"])
        def add(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

        assert fetcher.tool_count == 1


class TestDotFetch:
    """Tests for DotFetch class."""

    def test_create_instance(self, fetcher: DotFetch) -> None:
        """Test creating a DotFetch instance."""
        assert fetcher is not None
        assert fetcher.tool_count == 0

    def test_register_decorator(self, fetcher: DotFetch) -> None:
        """Test registering a tool with the decorator."""

        @fetcher.register
        def my_tool(x: int) -> str:
            """A simple test tool that multiplies a number."""
            return str(x * 2)

        assert fetcher.tool_count == 1

    def test_register_decorator_with_options(self, fetcher: DotFetch) -> None:
        """Test registering a tool with decorator options."""

        @fetcher.register(name="custom_name", tags=["test"])
        def another_tool(y: float) -> float:
            """A test tool with custom options."""
            return y * 3.0

        assert fetcher.tool_count == 1

    def test_multiple_registrations(self, fetcher: DotFetch) -> None:
        """Test registering multiple tools."""

        @fetcher.register
        def tool_one(a: str) -> str:
            """First test tool that processes strings."""
            return a.upper()

        @fetcher.register
        def tool_two(b: int) -> int:
            """Second test tool that processes integers."""
            return b + 1

        assert fetcher.tool_count == 2


class TestDotFetchHealthCheck:
    """Tests for health check functionality."""

    @pytest.mark.asyncio
    async def test_health_check(self, fetcher: DotFetch) -> None:
        """Test the health check method."""
        health = await fetcher.health_check()
        assert isinstance(health, dict)
        assert "vector_store" in health
        assert "embedder" in health
        assert "telemetry" in health


class TestDotFetchModuleImport:
    """Tests for from_modules and collect_tools_from_modules functionality."""

    @pytest.mark.asyncio
    async def test_from_modules_basic(self) -> None:
        """Test from_modules with single module."""
        fetcher = await DotFetch.from_modules(["tests.test_modules.module_a"])
        assert fetcher.tool_count == 2
        tools = fetcher._registry.list_tools()
        tool_names = {t.name for t in tools}
        assert "tool_a1" in tool_names
        assert "tool_a2" in tool_names

    @pytest.mark.asyncio
    async def test_from_modules_multiple(self) -> None:
        """Test from_modules with multiple modules."""
        fetcher = await DotFetch.from_modules(
            ["tests.test_modules.module_a", "tests.test_modules.module_b"]
        )
        assert fetcher.tool_count == 4
        tools = fetcher._registry.list_tools()
        tool_names = {t.name for t in tools}
        assert "tool_a1" in tool_names
        assert "tool_a2" in tool_names
        assert "tool_b1" in tool_names
        assert "tool_b2" in tool_names

    @pytest.mark.asyncio
    async def test_from_modules_custom_attr(self) -> None:
        """Test from_modules with custom attribute name."""
        fetcher = await DotFetch.from_modules(
            ["tests.test_modules.module_custom_attr"],
            attr="my_custom_tools",
        )
        assert fetcher.tool_count == 1
        tools = fetcher._registry.list_tools()
        assert tools[0].name == "custom_tool"

    @pytest.mark.asyncio
    async def test_from_modules_missing_attribute(self) -> None:
        """Test from_modules raises error when attribute doesn't exist."""
        with pytest.raises(ValueError, match="does not expose a DotFetch instance"):
            await DotFetch.from_modules(["tests.test_modules.module_no_tools"])

    @pytest.mark.asyncio
    async def test_from_modules_wrong_type(self) -> None:
        """Test from_modules raises error when attribute is not DotFetch."""
        with pytest.raises(ValueError, match="does not expose a DotFetch instance"):
            await DotFetch.from_modules(
                ["tests.test_modules.module_no_tools"],
                attr="some_variable",
            )

    @pytest.mark.asyncio
    async def test_collect_tools_from_modules_basic(self, fetcher: DotFetch) -> None:
        """Test collect_tools_from_modules adds tools to existing fetcher."""

        # Register a tool first
        @fetcher.register
        def existing_tool(x: int) -> int:
            """An existing tool."""
            return x

        assert fetcher.tool_count == 1

        # Now collect tools from module
        count = await fetcher.collect_tools_from_modules(["tests.test_modules.module_a"])
        assert count == 2
        assert fetcher.tool_count == 3

    @pytest.mark.asyncio
    async def test_collect_tools_from_modules_duplicate_handling(self, fetcher: DotFetch) -> None:
        """Test that duplicate tools within same batch are skipped."""
        # Import from module_a and module_c_duplicate in one call
        # module_c_duplicate has tool_a1 which duplicates module_a's tool_a1
        count = await fetcher.collect_tools_from_modules(
            ["tests.test_modules.module_a", "tests.test_modules.module_c_duplicate"]
        )
        # Should import 2 from module_a, and skip the duplicate from module_c_duplicate
        assert count == 2
        assert fetcher.tool_count == 2

    @pytest.mark.asyncio
    async def test_collect_tools_from_modules_handlers(self, fetcher: DotFetch) -> None:
        """Test that tool handlers are properly registered."""
        await fetcher.collect_tools_from_modules(["tests.test_modules.module_a"])

        # Check that handlers are registered (keyed by namespace-qualified
        # name so same-named tools across namespaces never clobber each other)
        assert fetcher._registry.get_handler("default.tool_a1") is not None
        assert fetcher._registry.get_handler("default.tool_a2") is not None

        # Verify handlers are callable
        handler1 = fetcher._registry.get_handler("default.tool_a1")
        assert callable(handler1)
        result = handler1(5)
        assert result == 10

    @pytest.mark.asyncio
    async def test_collect_tools_from_modules_custom_attr(self, fetcher: DotFetch) -> None:
        """Test collect_tools_from_modules with custom attribute name."""
        count = await fetcher.collect_tools_from_modules(
            ["tests.test_modules.module_custom_attr"],
            module_attr="my_custom_tools",
        )
        assert count == 1
        assert fetcher.tool_count == 1

    @pytest.mark.asyncio
    async def test_collect_tools_from_modules_error_handling(self, fetcher: DotFetch) -> None:
        """Test collect_tools_from_modules error handling."""
        with pytest.raises(ValueError, match="does not expose a DotFetch instance"):
            await fetcher.collect_tools_from_modules(["tests.test_modules.module_no_tools"])


@pytest.mark.asyncio
async def test_delete_tool_purges_registry_and_handlers():
    """delete_tool must remove the tool everywhere: vector store, registry
    (so list_tools_sync / required= pin resolution stop seeing it), and the
    handler map (so execute stops running it)."""
    from dotfetch.adapters.embedders.simple import SimpleEmbedder
    from dotfetch.schema.execution import ExecutionStatus, ToolCall

    fetcher = DotFetch(embedder=SimpleEmbedder(dimension=64))

    @fetcher.register(tags=["email"])
    def send_email(to: str) -> str:
        "Send an email message."
        return f"sent:{to}"

    await fetcher.sync()
    assert any(t.name == "send_email" for t in fetcher.list_tools_sync())

    count_before = fetcher.tool_count
    assert await fetcher.delete_tool("send_email") is True

    assert all(t.name != "send_email" for t in fetcher.list_tools_sync())
    assert fetcher.tool_count == count_before - 1, (
        "tool_count counts registry handlers, which delete_tool must purge"
    )
    result = await fetcher.execute(ToolCall(tool_name="send_email", arguments={"to": "x"}))
    assert result.status != ExecutionStatus.SUCCESS


@pytest.mark.asyncio
async def test_concurrent_first_use_initialises_the_store_once() -> None:
    import asyncio

    from dotfetch.adapters.embedders.simple import SimpleEmbedder
    from dotfetch.adapters.vector_stores.memory import InMemoryVectorStore

    class CountingStore(InMemoryVectorStore):
        def __init__(self) -> None:
            super().__init__()
            self.initialisations = 0

        async def initialize(self) -> None:
            self.initialisations += 1
            await asyncio.sleep(0)  # yield, so a second caller can race in
            await super().initialize()

    store = CountingStore()
    fetcher = DotFetch(vector_store=store, embedder=SimpleEmbedder(dimension=64))

    @fetcher.register(tags=["math"])
    def double(x: int) -> int:
        """Double a number."""
        return x * 2

    # Two retrieves before any sync, through the public path.
    first, second = await asyncio.gather(
        fetcher.retrieve_tools("double a number", score_threshold=0.0),
        fetcher.retrieve_tools("double a number", score_threshold=0.0),
    )
    assert first and second
    assert store.initialisations == 1


@pytest.mark.asyncio
async def test_close_closes_the_embedder() -> None:
    from dotfetch.adapters.embedders.simple import SimpleEmbedder

    class ClosableEmbedder(SimpleEmbedder):
        closed = False

        def close(self) -> None:
            ClosableEmbedder.closed = True

    await DotFetch(embedder=ClosableEmbedder()).close()
    assert ClosableEmbedder.closed


@pytest.mark.asyncio
async def test_disabled_telemetry_still_retrieves_and_executes() -> None:
    from dotfetch.adapters.embedders.simple import SimpleEmbedder
    from dotfetch.schema.config import DotFetchConfig, TelemetryConfig
    from dotfetch.schema.execution import ExecutionStatus, ToolCall

    fetcher = DotFetch(
        config=DotFetchConfig(telemetry=TelemetryConfig(enabled=False)),
        embedder=SimpleEmbedder(dimension=64),
    )

    @fetcher.register(tags=["math"])
    def double(x: int) -> int:
        """Double a number."""
        return x * 2

    assert await fetcher.retrieve_tools("double a number", score_threshold=0.0)
    result = await fetcher.execute(ToolCall(tool_name="double", arguments={"x": 2}))
    assert result.status is ExecutionStatus.SUCCESS and result.result == 4


@pytest.mark.asyncio
async def test_delete_tool_leaves_the_registry_alone_when_the_store_fails() -> None:
    from dotfetch.adapters.embedders.simple import SimpleEmbedder
    from dotfetch.adapters.vector_stores.memory import InMemoryVectorStore

    class FailingDeleteStore(InMemoryVectorStore):
        async def delete(self, name: str, namespace: str = "default") -> bool:
            raise RuntimeError("store unavailable")

    fetcher = DotFetch(vector_store=FailingDeleteStore(), embedder=SimpleEmbedder(dimension=64))

    @fetcher.register
    def keep_me(x: int) -> int:
        """Keep this tool around."""
        return x

    with pytest.raises(RuntimeError, match="store unavailable"):
        await fetcher.delete_tool("keep_me")
    # The store is asked first, so a store failure leaves the tool executable.
    assert fetcher._registry.has_tool("keep_me")
    assert fetcher.tool_count == 1


@pytest.mark.asyncio
async def test_reuse_after_close_initialises_the_store_again() -> None:
    from dotfetch.adapters.embedders.simple import SimpleEmbedder
    from dotfetch.adapters.vector_stores.memory import InMemoryVectorStore

    class CountingStore(InMemoryVectorStore):
        def __init__(self) -> None:
            super().__init__()
            self.initialisations = 0

        async def initialize(self) -> None:
            self.initialisations += 1
            await super().initialize()

    store = CountingStore()
    fetcher = DotFetch(vector_store=store, embedder=SimpleEmbedder(dimension=64))

    @fetcher.register(tags=["math"])
    def double(x: int) -> int:
        """Double a number."""
        return x * 2

    assert await fetcher.retrieve_tools("double a number", score_threshold=0.0)
    await fetcher.close()
    # close() drops the initialised flag and the lock, so the next call
    # starts the store again instead of assuming it is still open.
    assert await fetcher.retrieve_tools("double a number", score_threshold=0.0)
    assert store.initialisations == 2


@pytest.mark.asyncio
async def test_delete_tool_forgets_the_handlers_coercers() -> None:
    from dotfetch.adapters.embedders.simple import SimpleEmbedder
    from dotfetch.core import executor as executor_module
    from dotfetch.schema.execution import ExecutionStatus, ToolCall

    fetcher = DotFetch(embedder=SimpleEmbedder(dimension=64))

    @fetcher.register
    def shout(text: str) -> str:
        """Upper-case some text."""
        return text.upper()

    result = await fetcher.execute(ToolCall(tool_name="shout", arguments={"text": "hi"}))
    assert result.status is ExecutionStatus.SUCCESS and result.result == "HI"
    assert shout in executor_module._COERCER_CACHE  # memoised by the first call

    assert await fetcher.delete_tool("shout") is True
    assert shout not in executor_module._COERCER_CACHE
