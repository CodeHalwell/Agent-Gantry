"""Large MCP catalogues: searching within one server, and selecting above the ceiling.

A single MCP server can carry several hundred tools. Two things used to give
out at that size: there was no way to search within one server, and the
selector stopped selecting altogether once the catalogue passed its ceiling.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any

import pytest

from dotfetch import DotFetch
from dotfetch.adapters.embedders.simple import SimpleEmbedder
from dotfetch.adapters.selectors.system_one import SystemOneSelector
from dotfetch.adapters.system_one_client import SystemOneVerdict
from dotfetch.schema.mcp import mcp_server_tag
from dotfetch.schema.query import ConversationContext, ToolQuery
from dotfetch.schema.selection import SelectionCandidate
from dotfetch.schema.tool import ToolDefinition

SELECTOR_LOG = "dotfetch.adapters.selectors.system_one"

_NOUNS = ["invoice", "ticket", "cluster", "dashboard", "backup", "webhook", "report", "user"]
_VERBS = ["create", "list", "delete", "update", "export", "archive", "search", "restore"]


def _server_tools(server: str, count: int) -> list[ToolDefinition]:
    """``count`` tools as discovery would convert them from ``server``."""
    pytest.importorskip("mcp")
    from dotfetch.adapters.executors.mcp_client import MCPClient
    from dotfetch.schema.config import MCPServerConfig

    client = MCPClient(MCPServerConfig(name=server, url=f"https://{server}.example/mcp"))
    tools = []
    for i in range(count):
        verb, noun = _VERBS[i % len(_VERBS)], _NOUNS[(i // len(_VERBS)) % len(_NOUNS)]
        tools.append(
            client._convert_tool(
                SimpleNamespace(
                    name=f"{server}_{verb}_{noun}_{i}",
                    description=f"{verb.title()} the {noun} numbered {i}.",
                    inputSchema={},
                )
            )
        )
    return tools


def _native_tools(count: int) -> list[ToolDefinition]:
    return [
        ToolDefinition(
            name=f"local_{_VERBS[i % len(_VERBS)]}_{_NOUNS[i % len(_NOUNS)]}_{i}",
            description=f"{_VERBS[i % len(_VERBS)].title()} the {_NOUNS[i % len(_NOUNS)]} locally.",
            parameters_schema={"type": "object", "properties": {}},
        )
        for i in range(count)
    ]


async def _fetcher(tools: list[ToolDefinition], **kwargs: Any) -> DotFetch:
    fetcher = DotFetch(embedder=SimpleEmbedder(), **kwargs)
    for tool in tools:
        await fetcher.add_tool(tool)
    await fetcher.sync()
    return fetcher


def _query(text: str, **fields: Any) -> ToolQuery:
    return ToolQuery(context=ConversationContext(query=text), score_threshold=0.0, **fields)


def _names(result: Any) -> list[str]:
    return [scored.tool.name for scored in result.tools]


# --------------------------------------------------------------------------- #
# Searching within one server
# --------------------------------------------------------------------------- #

_QUERIES = [
    "restore the backup numbered 44",
    "export the report",
    "delete the webhook numbered 10",
    "list every cluster",
    "archive the dashboard",
]


async def test_a_scoped_search_matches_searching_that_server_alone() -> None:
    """The acceptance test: two servers of 120 tools each, among 60 others."""
    datadog = _server_tools("datadog", 120)
    mixed = await _fetcher([*datadog, *_server_tools("azure", 120), *_native_tools(60)])
    alone = await _fetcher(datadog)
    try:
        for text in _QUERIES:
            scoped = await mixed.retrieve(_query(text, limit=5, mcp_servers=["datadog"]))
            expected = await alone.retrieve(_query(text, limit=5))

            assert len(scoped.tools) == 5
            assert all(t.tool.metadata["mcp_server"] == "datadog" for t in scoped.tools)
            assert _names(scoped) == _names(expected)
    finally:
        await mixed.close()
        await alone.close()


async def test_the_scope_is_searched_in_full_not_through_the_pool() -> None:
    """One server's five tools, behind 300 that out-score them.

    Filtering after the candidate pool is drawn (as ``sources`` does) would
    find none of them: the pool is 32 wide and they rank far below it.
    """
    small = _server_tools("fathom", 5)
    noise = [
        ToolDefinition(
            name=f"restore_backup_{i}",
            description="Restore the backup. Restore the backup. Restore the backup.",
            parameters_schema={"type": "object", "properties": {}},
        )
        for i in range(300)
    ]
    fetcher = await _fetcher([*noise, *small])
    try:
        unscoped = await fetcher.retrieve(_query("restore the backup", limit=5))
        assert not any(t.tool.metadata.get("mcp_server") for t in unscoped.tools)

        scoped = await fetcher.retrieve(_query("restore the backup", limit=5, mcp_servers=["fathom"]))
        assert sorted(_names(scoped)) == sorted(t.name for t in small)
    finally:
        await fetcher.close()


async def test_several_servers_can_be_named_and_none_means_nothing() -> None:
    fetcher = await _fetcher(
        [*_server_tools("datadog", 10), *_server_tools("azure", 10), *_server_tools("clickup", 10)]
    )
    try:
        both = await fetcher.retrieve(_query("list", limit=30, mcp_servers=["datadog", "azure"]))
        assert {t.tool.metadata["mcp_server"] for t in both.tools} == {"datadog", "azure"}
        assert len(both.tools) == 20

        assert (await fetcher.retrieve(_query("list", mcp_servers=[]))).tools == []
        assert (await fetcher.retrieve(_query("list", mcp_servers=["no-such-server"]))).tools == []
    finally:
        await fetcher.close()


async def test_a_hand_written_tag_does_not_make_a_native_tool_a_servers() -> None:
    impostor = ToolDefinition(
        name="list_everything",
        description="List the invoice, ticket and cluster records.",
        parameters_schema={"type": "object", "properties": {}},
        tags=[mcp_server_tag("datadog")],
    )
    fetcher = await _fetcher([impostor, *_server_tools("datadog", 8)])
    try:
        result = await fetcher.retrieve(_query("list", limit=20, mcp_servers=["datadog"]))
        assert len(result.tools) == 8
        assert "list_everything" not in _names(result)
    finally:
        await fetcher.close()


async def test_the_convenience_call_returns_only_that_servers_schemas() -> None:
    fetcher = await _fetcher([*_server_tools("datadog", 20), *_server_tools("azure", 20)])
    try:
        schemas = await fetcher.retrieve_tools_from_server("azure", "export the report", limit=4)
        assert len(schemas) == 4
        assert all(s["function"]["name"].startswith("azure_") for s in schemas)

        with pytest.raises(TypeError, match="mcp_servers"):
            await fetcher.retrieve_tools_from_server("azure", "export", mcp_servers=["datadog"])
    finally:
        await fetcher.close()


class _Client:
    """A decision model that knows the right answer and records what it was asked."""

    def __init__(self, target: str, *, group: str | None = None) -> None:
        self.target = target
        self.group = group
        self.asked: list[list[str]] = []

    async def score(self, state: Any, candidates: Any, question: str) -> SystemOneVerdict:
        self.asked.append([c.id for c in candidates])
        return SystemOneVerdict(
            scores={
                c.id: 0.95 if c.id == self.group or c.name.endswith(self.target) else 0.05
                for c in candidates
            },
            requests=1,
        )


async def test_a_selector_is_only_shown_the_scoped_servers_tools() -> None:
    client = _Client("azure_export_invoice_4")
    fetcher = await _fetcher(
        [*_server_tools("datadog", 20), *_server_tools("azure", 20), *_native_tools(10)],
        selector=SystemOneSelector(client=client),
    )
    try:
        result = await fetcher.retrieve(_query("export", limit=3, mcp_servers=["azure"]))
        assert result.backend == "selector"
        assert _names(result) == ["azure_export_invoice_4"]
        assert len(client.asked) == 1 and len(client.asked[0]) == 20
        assert all(".azure_" in candidate_id for candidate_id in client.asked[0])
    finally:
        await fetcher.close()


# --------------------------------------------------------------------------- #
# Selecting above the ceiling
# --------------------------------------------------------------------------- #


def _candidates(sizes: dict[str, int]) -> list[SelectionCandidate]:
    return [
        SelectionCandidate(
            id=f"{group}.tool_{i}", name=f"{group}.tool_{i}", description="A tool.", group=group
        )
        for group, size in sizes.items()
        for i in range(size)
    ]


async def test_over_the_ceiling_a_grouped_catalogue_is_narrowed_not_abandoned() -> None:
    """Nine hundred candidates across four servers, against a ceiling of 512."""
    client = _Client("clickup.tool_7", group="clickup")
    selector = SystemOneSelector(client=client, max_candidates=512, max_groups=2)
    candidates = _candidates({"datadog": 300, "azure": 250, "clickup": 200, "fathom": 150})

    result = await selector.select("find the task", candidates, limit=5)

    assert not result.fallback
    assert result.selected == ["clickup.tool_7"]
    groups_pass, members_pass = client.asked
    assert sorted(groups_pass) == ["azure", "clickup", "datadog", "fathom"]
    # The member pass is held to the ceiling the one-stage pass is held to.
    assert 200 <= len(members_pass) <= 512
    assert "clickup.tool_7" in members_pass
    assert result.requests == 2


async def test_the_extra_pass_is_announced_once_per_catalogue_size(
    caplog: pytest.LogCaptureFixture,
) -> None:
    selector = SystemOneSelector(client=_Client("a.tool_1", group="a"), max_candidates=10)
    candidates = _candidates({"a": 8, "b": 8})

    with caplog.at_level(logging.DEBUG, logger=SELECTOR_LOG):
        for _ in range(3):
            assert not (await selector.select("q", candidates, limit=3)).fallback

    ceiling = [r for r in caplog.records if "ceiling" in r.getMessage()]
    assert [r.levelname for r in ceiling] == ["WARNING", "DEBUG", "DEBUG"]
    assert "narrowing to the best of 2 groups" in ceiling[0].getMessage()
    assert "falls back" not in ceiling[0].getMessage()


async def test_one_group_over_the_ceiling_still_falls_back() -> None:
    """There is nothing to narrow by, and a prefix would hide the rest."""
    client = _Client("default.tool_1")
    selector = SystemOneSelector(client=client, max_candidates=10)

    result = await selector.select("q", _candidates({"default": 30}), limit=3)

    assert result.fallback and "exceeds max_candidates=10" in (result.reason or "")
    assert client.asked == []


async def test_a_best_group_too_big_for_the_ceiling_is_a_fallback_not_a_skip() -> None:
    """Scoring the smaller, worse group instead would report success on the wrong tools."""
    client = _Client("big.tool_3", group="big")
    selector = SystemOneSelector(client=client, max_candidates=10)

    result = await selector.select("q", _candidates({"big": 30, "small": 4}), limit=3)

    assert result.fallback
    assert result.reason == "group 'big' holds 30 candidates, over max_candidates=10"
    assert len(client.asked) == 1  # the group pass; no member pass was attempted


async def test_groups_are_taken_best_first_until_the_next_would_not_fit() -> None:
    class _Ranked(_Client):
        async def score(self, state: Any, candidates: Any, question: str) -> SystemOneVerdict:
            self.asked.append([c.id for c in candidates])
            order = {"first": 0.9, "second": 0.8, "third": 0.7}
            return SystemOneVerdict(
                scores={c.id: order.get(c.id, 0.6) for c in candidates}, requests=1
            )

    client = _Ranked("unused")
    selector = SystemOneSelector(client=client, max_candidates=10, max_groups=5)

    await selector.select("q", _candidates({"first": 6, "second": 6, "third": 3}), limit=3)

    # ``second`` does not fit beside ``first``; ``third`` would, but is not
    # reached past it.
    assert {candidate_id.split(".")[0] for candidate_id in client.asked[1]} == {"first"}


async def test_a_large_mcp_registry_is_selected_end_to_end() -> None:
    """Through the fetcher: MCP tools share a namespace and still form groups."""
    client = _Client("azure_export_invoice_4", group="azure")
    tools = [
        *_server_tools("datadog", 300),
        *_server_tools("azure", 250),
        *_server_tools("clickup", 200),
        *_server_tools("fathom", 150),
    ]
    fetcher = await _fetcher(tools, selector=SystemOneSelector(client=client, max_groups=1))
    try:
        result = await fetcher.retrieve(_query("export the report", limit=3))

        assert result.backend == "selector"
        assert result.selector_fallback is False
        assert _names(result) == ["azure_export_invoice_4"]
        assert len(client.asked[0]) == 4 and len(client.asked[1]) == 250
    finally:
        await fetcher.close()


# --------------------------------------------------------------------------- #
# Cataloguing a server without committing its tools
# --------------------------------------------------------------------------- #


class _CountingEmbedder(SimpleEmbedder):
    """Records every text it is asked to embed."""

    def __init__(self) -> None:
        super().__init__()
        self.embedded: list[str] = []

    async def embed_text(self, text: str) -> list[float]:
        self.embedded.append(text)
        return await super().embed_text(text)


class _Server:
    """Stands in for a live MCP server behind a registered client."""

    def __init__(self, fetcher: DotFetch, name: str, tools: list[ToolDefinition]) -> None:
        self.tools = tools
        self.listings = 0
        fetcher.register_mcp_server(
            name, url=f"https://{name}.example/mcp", description="Observability platform."
        )
        client = fetcher._mcp_registry.get_client(name)
        client.list_tools = self._list_tools  # type: ignore[method-assign]

    async def _list_tools(self) -> list[ToolDefinition]:
        self.listings += 1
        return list(self.tools)


def _registered(fetcher: DotFetch, server: str) -> list[str]:
    return sorted(
        t.name for t in fetcher.list_tools_sync() if t.metadata.get("mcp_server") == server
    )


async def test_cataloguing_five_hundred_tools_embeds_and_registers_none() -> None:
    embedder = _CountingEmbedder()
    fetcher = DotFetch(embedder=embedder)
    try:
        server = _Server(fetcher, "datadog", _server_tools("datadog", 500))

        catalogue = await fetcher.list_server_tools("datadog")

        assert len(catalogue) == 500
        assert embedder.embedded == []
        assert _registered(fetcher, "datadog") == []
        assert (await fetcher.retrieve(_query("restore the backup"))).tools == []
        assert server.listings == 1
    finally:
        await fetcher.close()


async def test_adopting_five_tools_embeds_five_and_does_not_ask_the_server_again() -> None:
    embedder = _CountingEmbedder()
    fetcher = DotFetch(embedder=embedder)
    try:
        server = _Server(fetcher, "datadog", _server_tools("datadog", 500))
        catalogue = await fetcher.list_server_tools("datadog")
        wanted = [t.name for t in catalogue[:5]]

        assert await fetcher.adopt_server_tools("datadog", wanted) == 5

        assert _registered(fetcher, "datadog") == sorted(wanted)
        assert len(embedder.embedded) == 5
        assert server.listings == 1
        # Adopted tools are retrievable; the other 495 are not there to compete.
        result = await fetcher.retrieve(_query("create the invoice", limit=10))
        assert sorted(_names(result)) == sorted(wanted)
    finally:
        await fetcher.close()


async def test_adoption_adds_to_what_was_adopted_and_drops_what_the_server_dropped() -> None:
    fetcher = DotFetch(embedder=SimpleEmbedder())
    try:
        tools = _server_tools("datadog", 6)
        server = _Server(fetcher, "datadog", tools)

        await fetcher.adopt_server_tools("datadog", [tools[0].name, tools[1].name])
        await fetcher.adopt_server_tools("datadog", [tools[2].name])
        assert _registered(fetcher, "datadog") == sorted(t.name for t in tools[:3])

        server.tools = tools[1:]  # the server withdraws the first tool
        await fetcher.adopt_server_tools("datadog", [tools[3].name], refresh=True)
        assert _registered(fetcher, "datadog") == sorted(t.name for t in tools[1:4])
    finally:
        await fetcher.close()


async def test_a_tool_can_be_adopted_by_the_name_the_server_gives_it() -> None:
    pytest.importorskip("mcp")
    from dotfetch.adapters.executors.mcp_client import MCPClient
    from dotfetch.schema.config import MCPServerConfig

    client = MCPClient(MCPServerConfig(name="datadog", url="https://datadog.example/mcp"))
    tool = client._convert_tool(
        SimpleNamespace(name="Search-Logs", description="Search the application logs.", inputSchema={})
    )
    assert tool.name != "Search-Logs"
    fetcher = DotFetch(embedder=SimpleEmbedder())
    try:
        _Server(fetcher, "datadog", [tool])
        assert await fetcher.adopt_server_tools("datadog", ["Search-Logs"]) == 1
        assert _registered(fetcher, "datadog") == [tool.name]
    finally:
        await fetcher.close()


async def test_an_unknown_name_is_refused_and_nothing_is_adopted() -> None:
    fetcher = DotFetch(embedder=SimpleEmbedder())
    try:
        tools = _server_tools("datadog", 4)
        _Server(fetcher, "datadog", tools)

        with pytest.raises(ValueError, match="has no tool named 'no_such_tool'"):
            await fetcher.adopt_server_tools("datadog", [tools[0].name, "no_such_tool"])
        assert _registered(fetcher, "datadog") == []
    finally:
        await fetcher.close()


async def test_discovery_still_adopts_everything_and_asks_afresh() -> None:
    fetcher = DotFetch(embedder=SimpleEmbedder())
    try:
        server = _Server(fetcher, "datadog", _server_tools("datadog", 12))
        await fetcher.list_server_tools("datadog")

        assert await fetcher.discover_tools_from_server("datadog") == 12
        assert len(_registered(fetcher, "datadog")) == 12
        assert server.listings == 2
    finally:
        await fetcher.close()


async def test_a_server_with_a_long_description_can_be_indexed() -> None:
    """Its searchable text used to overrun the pseudo-tool's 2000-character
    description and fail validation, so the server could not be synced at all."""
    fetcher = DotFetch(embedder=SimpleEmbedder())
    try:
        _Server(fetcher, "datadog", [])
        fetcher._mcp_registry.get_server("datadog").extended_description = "logs and traces " * 500
        fetcher._mcp_registry.add_pending(fetcher._mcp_registry.get_server("datadog"))

        assert await fetcher.sync_mcp_servers() == 1
        assert [s.name for s in await fetcher.retrieve_mcp_servers("logs and traces")] == ["datadog"]
    finally:
        await fetcher.close()


async def test_a_failure_to_embed_does_not_mark_the_server_unhealthy() -> None:
    """The server answered; it is the embedder that failed."""

    class _Broken(SimpleEmbedder):
        async def embed_batch(self, texts: list[str], batch_size: int | None = None) -> Any:
            raise RuntimeError("the embedding service is down")

    fetcher = DotFetch(embedder=_Broken())
    try:
        _Server(fetcher, "datadog", _server_tools("datadog", 3))
        with pytest.raises(RuntimeError, match="embedding service is down"):
            await fetcher.discover_tools_from_server("datadog")

        health = fetcher._mcp_registry.get_server("datadog").health
        assert health.available is True
        assert health.consecutive_failures == 0
    finally:
        await fetcher.close()


async def test_a_reused_listing_is_as_old_as_the_call_that_made_it() -> None:
    """Documented, and the reason ``refresh`` exists."""
    fetcher = DotFetch(embedder=SimpleEmbedder())
    try:
        tools = _server_tools("datadog", 4)
        server = _Server(fetcher, "datadog", tools)
        await fetcher.list_server_tools("datadog")
        server.tools = tools[1:]  # dropped after the listing was made

        assert await fetcher.adopt_server_tools("datadog", [tools[0].name]) == 1
        with pytest.raises(ValueError, match="has no tool named"):
            await fetcher.adopt_server_tools("datadog", [tools[0].name], refresh=True)
        # Asking again also removed the tool adopted from the stale listing.
        assert _registered(fetcher, "datadog") == []
    finally:
        await fetcher.close()


# --------------------------------------------------------------------------- #
# A server is routed to by what it lists
# --------------------------------------------------------------------------- #


async def _with_clear_servers(fetcher: DotFetch) -> None:
    for name, description in (
        ("mailroom", "Send and search email, and manage mailboxes."),
        ("paydesk", "Charge cards and issue refunds."),
        ("mapkit", "Directions, distances and place search."),
    ):
        fetcher.register_mcp_server(name, url=f"https://{name}.example/mcp", description=description)


async def _server_rows(fetcher: DotFetch) -> list[ToolDefinition]:
    from dotfetch.schema.tool import MCP_SERVER_NAMESPACE

    return await fetcher.list_tools(namespace=MCP_SERVER_NAMESPACE)


async def test_a_vaguely_described_server_is_found_by_what_its_tools_do() -> None:
    fetcher = DotFetch(embedder=SimpleEmbedder())
    try:
        await _with_clear_servers(fetcher)
        _Server(fetcher, "datadog", _server_tools("datadog", 16))  # "Observability platform."

        async def first(query: str) -> str:
            return (await fetcher.retrieve_mcp_servers(query, limit=1))[0].name

        question = "restore the backup numbered 7"
        assert await first(question) != "datadog"

        await fetcher.list_server_tools("datadog")

        assert await first(question) == "datadog"
        # The registration is untouched: the catalogue is indexed beside it.
        server = fetcher._mcp_registry.get_server("datadog")
        assert server.description == "Observability platform."
        assert server.extended_description is None
    finally:
        await fetcher.close()


async def test_listing_embeds_nothing_and_the_next_server_search_embeds_the_pieces() -> None:
    embedder = _CountingEmbedder()
    fetcher = DotFetch(embedder=embedder)
    try:
        _Server(fetcher, "datadog", _server_tools("datadog", 40))
        await fetcher.sync_mcp_servers()
        embedder.embedded.clear()

        await fetcher.list_server_tools("datadog")
        assert embedder.embedded == []

        await fetcher.retrieve_mcp_servers("restore the backup", limit=1)
        pieces = [text for text in embedder.embedded if text.startswith("datadog tools: ")]
        assert 2 <= len(pieces) < 40  # several tools to a piece
    finally:
        await fetcher.close()


async def test_one_servers_many_pieces_do_not_crowd_the_others_out() -> None:
    fetcher = DotFetch(embedder=SimpleEmbedder())
    try:
        await _with_clear_servers(fetcher)
        _Server(fetcher, "datadog", _server_tools("datadog", 400))
        await fetcher.list_server_tools("datadog")

        servers = await fetcher.retrieve_mcp_servers("restore the backup", limit=4)

        assert len(await _server_rows(fetcher)) > 4 + 12  # far more rows than servers
        assert [s.name for s in servers][0] == "datadog"
        assert sorted(s.name for s in servers) == ["datadog", "mailroom", "mapkit", "paydesk"]
    finally:
        await fetcher.close()


async def test_a_namespace_scope_is_not_lost_among_other_namespaces_servers() -> None:
    """Seventy better-matching servers elsewhere; the one in scope is still found."""
    pytest.importorskip("mcp")
    fetcher = DotFetch(embedder=SimpleEmbedder())
    try:
        for i in range(70):
            fetcher.register_mcp_server(
                f"backup_{i}",
                url=f"https://backup-{i}.example/mcp",
                description="Backup backup backup: restore the backup.",
                namespace="infra",
            )
        fetcher.register_mcp_server(
            "vault", url="https://vault.example/mcp", description="Restores a backup.", namespace="ops"
        )

        servers = await fetcher.retrieve_mcp_servers("backup", limit=1, namespaces=["ops"])
        assert [s.name for s in servers] == ["vault"]
        assert (await fetcher.retrieve_mcp_servers("backup", limit=1))[0].namespace == "infra"
    finally:
        await fetcher.close()


async def test_an_unreachable_servers_pieces_do_not_hide_the_healthy_one_behind_them() -> None:
    """One server asked for; the unreachable one fills the first window with its pieces.

    Finding one server used to end the search, and marking it down could not
    then put it below a healthy server the window had not reached.
    """
    fetcher = DotFetch(embedder=SimpleEmbedder())
    try:
        _Server(fetcher, "datadog", _server_tools("datadog", 400))
        await fetcher.list_server_tools("datadog")
        fetcher.register_mcp_server(
            "vault", url="https://vault.example/mcp", description="Restore a backup."
        )
        query = "restore the backup numbered 44"

        both = await fetcher.retrieve_mcp_servers(query, limit=2)
        assert [s.name for s in both] == ["datadog", "vault"]  # datadog matches better
        rows = [r for r in await _server_rows(fetcher) if r.metadata["server_name"] == "datadog"]
        assert len(rows) > 4  # more of its rows than the first window of four

        fetcher._mcp_registry.update_health(
            "datadog", "default", available=False, consecutive_failures=5
        )
        assert [s.name for s in await fetcher.retrieve_mcp_servers(query, limit=1)] == ["vault"]
    finally:
        await fetcher.close()


async def test_catalogue_pieces_follow_the_listing_and_go_with_it() -> None:
    fetcher = DotFetch(embedder=SimpleEmbedder())
    try:
        server = _Server(fetcher, "datadog", _server_tools("datadog", 60))

        async def pieces() -> int:
            await fetcher.sync_mcp_servers()
            return len(await _server_rows(fetcher)) - 1  # less the server's own entry

        await fetcher.list_server_tools("datadog")
        full = await pieces()
        assert full >= 3

        server.tools = _server_tools("datadog", 6)
        await fetcher.list_server_tools("datadog")
        assert 1 <= await pieces() < full

        server.tools = []
        await fetcher.list_server_tools("datadog")
        assert await pieces() == 0

        # A server registered again at a different endpoint is another server.
        server.tools = _server_tools("datadog", 6)
        await fetcher.list_server_tools("datadog")
        assert await pieces() >= 1
        fetcher.register_mcp_server(
            "datadog", url="https://elsewhere.example/mcp", description="Observability platform."
        )
        assert await pieces() == 0
    finally:
        await fetcher.close()


async def test_syncing_servers_counts_servers_not_index_rows() -> None:
    fetcher = DotFetch(embedder=SimpleEmbedder())
    try:
        _Server(fetcher, "datadog", _server_tools("datadog", 60))
        await fetcher.list_server_tools("datadog")
        assert await fetcher.sync_mcp_servers() == 1
        assert await fetcher.sync_mcp_servers() == 0
    finally:
        await fetcher.close()


async def test_catalogue_pieces_are_never_returned_as_tools() -> None:
    from dotfetch.schema.tool import is_reserved_namespace

    fetcher = await _fetcher(_native_tools(8))
    try:
        _Server(fetcher, "datadog", _server_tools("datadog", 60))
        await fetcher.list_server_tools("datadog")
        await fetcher.sync_mcp_servers()

        result = await fetcher.retrieve(_query("restore the backup", limit=8))
        assert len(result.tools) == 8
        assert not [t for t in result.tools if is_reserved_namespace(t.tool.namespace)]
    finally:
        await fetcher.close()


def test_a_catalogue_is_cut_into_named_and_described_tools_or_names_alone() -> None:
    from dotfetch.core.mcp_manager import MAX_CATALOGUE_CHUNKS, catalogue_chunks
    from dotfetch.schema.mcp import MCPServerDefinition

    server = MCPServerDefinition(
        name="datadog", description="Observability platform.", url="https://datadog.example/mcp"
    )
    few = catalogue_chunks(server, _server_tools("datadog", 3))
    assert [c.description for c in few] == [
        "datadog tools: datadog create invoice 0: Create the invoice numbered 0; "
        "datadog list invoice 1: List the invoice numbered 1; "
        "datadog delete invoice 2: Delete the invoice numbered 2"
    ]
    assert few[0].metadata["server_name"] == "datadog"

    huge = catalogue_chunks(server, _server_tools("datadog", 1200))
    assert len(huge) <= MAX_CATALOGUE_CHUNKS
    assert "numbered" not in huge[0].description  # names alone, to fit more of them
    assert len({c.name for c in huge}) == len(huge)
    assert catalogue_chunks(server, []) == []
