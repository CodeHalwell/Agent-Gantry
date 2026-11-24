"""A long description is findable by all of it, not only by its opening.

An embedder reads a bounded number of tokens, so a tool whose description runs
to several thousand characters was searchable by roughly its first thousand.
Long text is now indexed a second time in pieces, and a tool is scored by the
best of its own vector and its pieces'.
"""

from __future__ import annotations

import re
from types import SimpleNamespace
from typing import Any

import pytest

from agent_gantry import AgentGantry
from agent_gantry.adapters.embedders.simple import SimpleEmbedder
from agent_gantry.core.chunks import (
    CHUNK_AFTER,
    CHUNK_CHARS,
    CHUNK_SCORE_WEIGHT,
    MAX_CHUNKS,
    MAX_SKILL_CHUNKS,
    chunk_parent,
    split_text,
    tool_chunks,
)
from agent_gantry.schema.config import AgentGantryConfig, RoutingConfig
from agent_gantry.schema.query import ConversationContext, ToolQuery
from agent_gantry.schema.tool import (
    ToolDefinition,
    chunk_namespace,
    is_reserved_namespace,
)

_FILLER = "Results are paged and times are in UTC unless a zone is given. "
_TAIL = "Zebra quagga okapi: use this for striped ungulate questions."


def _tool(name: str, description: str, **fields: Any) -> ToolDefinition:
    fields.setdefault("parameters_schema", {"type": "object", "properties": {}})
    return ToolDefinition(name=name, description=description, **fields)


def _long_tool(name: str = "search_archive", **fields: Any) -> ToolDefinition:
    """A tool whose only distinguishing sentence is its last, about 1500 characters in."""
    return _tool(name, "Search the archive. " + _FILLER * 23 + _TAIL, **fields)


def _query(text: str, **fields: Any) -> ToolQuery:
    return ToolQuery(context=ConversationContext(query=text), score_threshold=0.0, **fields)


async def _gantry(tools: list[ToolDefinition], **kwargs: Any) -> AgentGantry:
    gantry = AgentGantry(embedder=SimpleEmbedder(), **kwargs)
    for tool in tools:
        await gantry.add_tool(tool)
    await gantry.sync()
    return gantry


def _distractors(count: int = 40) -> list[ToolDefinition]:
    return [_tool(f"read_report_{i}", f"Read report number {i} to the user.") for i in range(count)]


async def _chunk_rows(gantry: AgentGantry, namespace: str = "default") -> list[ToolDefinition]:
    return await gantry.list_tools(namespace=chunk_namespace(namespace))


def _names(result: Any) -> list[str]:
    return [scored.tool.name for scored in result.tools]


# --------------------------------------------------------------------------- #
# Cutting text
# --------------------------------------------------------------------------- #


def test_text_is_cut_at_sentence_ends_and_nothing_is_lost() -> None:
    text = " ".join(f"Sentence number {i} says a little about the subject." for i in range(60))

    pieces = split_text(text)

    assert " ".join(pieces) == text
    assert all(len(piece) <= CHUNK_CHARS for piece in pieces)
    assert all(piece.endswith(".") for piece in pieces)


def test_a_sentence_longer_than_a_chunk_is_cut_at_a_space_or_mid_word() -> None:
    assert "".join(split_text("x" * 2000)) == "x" * 2000
    spaced = " ".join(["word"] * 500)
    pieces = split_text(spaced)
    assert " ".join(pieces) == spaced and all(len(p) <= CHUNK_CHARS for p in pieces)


def test_a_few_words_between_long_sentences_join_a_neighbour() -> None:
    """A piece of a few words is no vector of its own, and may be too short to store."""
    long_one = "A" * 698 + "."
    for text in (
        f"{long_one} {long_one} OK.",  # left at the end
        f"{long_one} OK. {long_one}",  # caught in the middle
        f"OK. {long_one} {long_one}",  # at the start
    ):
        pieces = split_text(text)
        assert " ".join(pieces) == text
        assert len(pieces) == 2 and all(len(piece) >= 100 for piece in pieces)

    # The case that failed a whole sync: a one-word name, no tags, and a last
    # piece too short to make a valid description with it.
    chunks = tool_chunks(_tool("run", f"{long_one} {long_one} OK."))
    assert len(chunks) == 2 and chunks[-1].description.endswith("OK.")


def test_only_long_text_is_chunked() -> None:
    assert tool_chunks(_tool("ping_host", "x" * CHUNK_AFTER)) == []
    assert tool_chunks(_tool("ping_host", "Check whether a host answers.")) == []
    assert len(tool_chunks(_long_tool())) >= 2


def test_a_chunk_says_what_it_is_a_piece_of() -> None:
    tool = _long_tool(namespace="ops", tags=["mcp:datadog"])
    chunks = tool_chunks(tool)

    assert all(c.namespace == chunk_namespace("ops") == "__chunks__ops" for c in chunks)
    assert all(is_reserved_namespace(c.namespace) for c in chunks)
    assert not is_reserved_namespace("ops")
    assert all(re.fullmatch(r"[a-z][a-z0-9_]*", c.name) for c in chunks)
    assert len({c.name for c in chunks}) == len(chunks)
    assert all(chunk_parent(c) == ("ops", "search_archive") for c in chunks)
    # The tags travel, so a tag filter finds the tool by its chunks too.
    assert all(c.tags == ["mcp:datadog"] for c in chunks)
    assert all(c.description.startswith("search_archive mcp:datadog ") for c in chunks)
    assert _TAIL in chunks[-1].description
    assert chunk_parent(tool) is None


def test_many_tags_do_not_push_a_piece_out_of_its_chunk() -> None:
    """Eighty tags are 1,700 characters: with them all in front, the piece was cut off."""
    tags = [f"capability-number-{i:03d}" for i in range(80)]
    chunks = tool_chunks(_long_tool(tags=tags))

    assert _TAIL in chunks[-1].description  # the whole piece is still there
    assert all(len(c.description) < 1000 for c in chunks)
    assert all(c.description.startswith("search_archive capability-number-000 ") for c in chunks)
    assert all(c.tags == tags for c in chunks)  # a tag filter still sees every tag


def test_the_full_text_is_the_extended_description_when_there_is_one() -> None:
    """MCP discovery keeps a long description there, with only its head in ``description``."""
    tool = _tool("search_logs", "Search the logs.", extended_description=_FILLER * 30 + _TAIL)
    chunks = tool_chunks(tool)
    assert chunks and _TAIL in chunks[-1].description


def test_chunks_are_bounded_without_losing_the_end_of_the_text() -> None:
    """Sentence ends would cut this into 23 pieces; the cap is 16, and all of it is kept."""
    # Twenty-three sentences of 400 characters: no two fit in one chunk.
    text = " ".join(["x" * 399 + "."] * 23 + [_TAIL])
    tool = _tool("search_logs", "Search the logs.", extended_description=text)
    assert len(split_text(text)) == 23
    chunks = tool_chunks(tool)

    assert len(chunks) == MAX_CHUNKS
    assert _TAIL in chunks[-1].description  # not dropped with the last seven
    assert all(len(piece) <= CHUNK_CHARS for piece in split_text(text, max_pieces=MAX_CHUNKS))
    rejoined = "".join(split_text(text, max_pieces=MAX_CHUNKS))
    assert rejoined.replace(" ", "") == text.replace(" ", "")

    # The same for a skill's body, at its own cap.
    body = " ".join(["y" * 399 + "."] * 120 + [_TAIL])
    pieces = skill_chunks(_skill("handbook", "The keeper handbook.", body))
    assert len(pieces) == MAX_SKILL_CHUNKS and _TAIL.split(":")[0] in pieces[-1].content


# --------------------------------------------------------------------------- #
# Retrieval
# --------------------------------------------------------------------------- #


async def test_a_tool_is_found_by_the_last_sentence_of_a_long_description() -> None:
    gantry = await _gantry([_long_tool(), *_distractors()])
    try:
        result = await gantry.retrieve(_query("zebra quagga okapi striped ungulate", limit=3))
        assert _names(result)[0] == "search_archive"
    finally:
        await gantry.close()


async def test_without_chunks_that_sentence_is_out_of_reach() -> None:
    """The same registry with chunking off: what the change is responsible for."""
    config = AgentGantryConfig(routing=RoutingConfig(chunk_long_text=False))
    chunked = await _gantry([_long_tool(), *_distractors()])
    plain = await _gantry([_long_tool(), *_distractors()], config=config)
    try:
        assert await _chunk_rows(plain) == []
        assert await _chunk_rows(chunked) != []

        async def score(gantry: AgentGantry) -> float:
            result = await gantry.retrieve(_query("zebra quagga okapi striped ungulate", limit=50))
            return next(t.semantic_score for t in result.tools if t.tool.name == "search_archive")

        assert await score(chunked) > await score(plain) + 0.05
    finally:
        await chunked.close()
        await plain.close()


async def test_chunks_are_never_returned_listed_or_counted_as_tools() -> None:
    gantry = await _gantry([_long_tool(), *_distractors(5)])
    try:
        result = await gantry.retrieve(_query("zebra quagga okapi striped ungulate", limit=50))
        assert not [t for t in result.tools if is_reserved_namespace(t.tool.namespace)]
        assert len(_names(result)) == len(set(_names(result))) == 6

        listed = await gantry.list_tools()
        assert sorted(t.name for t in listed) == sorted(
            ["search_archive", *(f"read_report_{i}" for i in range(5))]
        )
    finally:
        await gantry.close()


async def test_a_namespace_filter_still_reaches_the_chunks_of_that_namespace_only() -> None:
    gantry = await _gantry(
        [
            _long_tool("search_archive", namespace="ops"),
            _long_tool("search_vault", namespace="finance"),
            *_distractors(),
        ]
    )
    try:
        result = await gantry.retrieve(
            _query("zebra quagga okapi striped ungulate", limit=5, namespaces=["ops"])
        )
        assert _names(result) == ["search_archive"]
    finally:
        await gantry.close()


async def test_a_tool_found_only_through_a_chunk_joins_the_candidates() -> None:
    """Sixty tools out-score its own vector, so only a chunk puts it in the pool of 32."""
    noise = [
        _tool(f"zebra_fact_{i}", "Zebra. Zebra facts about a zebra and another zebra.")
        for i in range(60)
    ]
    gantry = await _gantry([_long_tool(), *noise])
    try:
        result = await gantry.retrieve(_query("zebra quagga okapi striped ungulate", limit=5))
        assert "search_archive" in _names(result)
        # ...and asking for diversity, which wants each candidate's embedding,
        # copes with one that arrived without it.
        diverse = await gantry.retrieve(
            _query("zebra quagga okapi striped ungulate", limit=5, diversity_factor=0.5)
        )
        assert "search_archive" in _names(diverse)
    finally:
        await gantry.close()


async def test_a_search_scoped_to_one_mcp_server_finds_a_tool_by_its_chunks() -> None:
    pytest.importorskip("mcp")
    from agent_gantry.adapters.executors.mcp_client import MCPClient
    from agent_gantry.schema.config import MCPServerConfig

    def discovered(server: str, name: str) -> ToolDefinition:
        client = MCPClient(MCPServerConfig(name=server, url=f"https://{server}.example/mcp"))
        description = "Search the archive. " + _FILLER * 40 + _TAIL  # past the 2000 ceiling
        return client._convert_tool(
            SimpleNamespace(name=name, description=description, inputSchema={})
        )

    here, there = discovered("datadog", "search_archive"), discovered("azure", "search_vault")
    assert _TAIL not in here.description and _TAIL in (here.extended_description or "")
    gantry = await _gantry([here, there, *_distractors()])
    try:
        result = await gantry.retrieve(
            _query("zebra quagga okapi striped ungulate", limit=5, mcp_servers=["datadog"])
        )
        assert _names(result) == ["search_archive"]
    finally:
        await gantry.close()


# --------------------------------------------------------------------------- #
# Chunks live and die with their tool
# --------------------------------------------------------------------------- #


async def test_a_shorter_description_leaves_no_chunks_behind() -> None:
    """A tool re-synced with fewer chunks, or none, loses the ones it no longer has."""
    from agent_gantry.adapters.vector_stores.memory import InMemoryVectorStore
    from agent_gantry.core.chunks import sync_tool_chunks

    store, embedder = InMemoryVectorStore(), SimpleEmbedder()
    await store.initialize()

    async def chunk_count() -> int:
        return len(await store.list_all(namespace=chunk_namespace("default")))

    long = _tool("search_archive", "Search the archive.", extended_description=_FILLER * 60)
    shorter = _tool("search_archive", "Search the archive. " + _FILLER * 23)
    assert len(tool_chunks(long)) > len(tool_chunks(shorter)) >= 2

    await sync_tool_chunks(store, embedder, [long])
    assert await chunk_count() == len(tool_chunks(long))

    stored = {"default.search_archive"}
    await sync_tool_chunks(store, embedder, [shorter], previously_stored=stored)
    assert await chunk_count() == len(tool_chunks(shorter))

    short = _tool("search_archive", "Search the archive.")
    await sync_tool_chunks(store, embedder, [short], previously_stored=stored)
    assert await chunk_count() == 0


async def test_deleting_a_tool_deletes_its_chunks() -> None:
    gantry = await _gantry([_long_tool(), *_distractors(3)])
    try:
        assert await _chunk_rows(gantry) != []
        assert await gantry.delete_tool("search_archive")
        assert await _chunk_rows(gantry) == []
    finally:
        await gantry.close()


async def test_pruning_keeps_the_chunks_of_kept_tools_and_drops_the_rest() -> None:
    first = await _gantry([_long_tool("search_archive"), _long_tool("search_vault")])
    try:
        before = len(await _chunk_rows(first))
        assert before >= 4

        # A second gantry on the same store registers only one of the two.
        second = AgentGantry(embedder=SimpleEmbedder(), vector_store=first._vector_store)
        await second.add_tool(_long_tool("search_archive"))
        await second.sync(prune=True)

        remaining = await _chunk_rows(second)
        assert len(remaining) == before // 2
        assert {chunk_parent(c) for c in remaining} == {("default", "search_archive")}
        assert [t.name for t in await second.list_tools()] == ["search_archive"]
    finally:
        await first.close()


# --------------------------------------------------------------------------- #
# Skills: the body is searchable
# --------------------------------------------------------------------------- #

from agent_gantry.core.chunks import skill_chunks  # noqa: E402
from agent_gantry.schema.skill import Skill, SkillCategory  # noqa: E402


def _skill(name: str, description: str, body: str, **fields: Any) -> Skill:
    return Skill(name=name, description=description, content=body, **fields)


def _log_skill(**fields: Any) -> Skill:
    return _skill(
        "log_search_syntax",
        "Logwatch reference card.",
        "# Reference\n\n" + _FILLER * 12 + "Narrow by service with service:checkout and env:prod.",
        **fields,
    )


def _other_skills(count: int = 12) -> list[Skill]:
    return [
        _skill(f"playbook_{i}", f"Playbook number {i} for the team.", f"Step one of playbook {i}.")
        for i in range(count)
    ]


async def _skills_gantry(skills: list[Skill], **kwargs: Any) -> AgentGantry:
    gantry = AgentGantry(embedder=SimpleEmbedder(), **kwargs)
    await gantry.add_skills(skills)
    return gantry


def test_a_skills_content_is_chunked_whatever_its_length() -> None:
    short = _skill("greeting", "How to greet a customer.", "Say hello and use their name.")
    chunks = skill_chunks(short)

    assert len(chunks) == 1
    assert chunks[0].is_chunk and not short.is_chunk
    assert chunks[0].namespace == chunk_namespace("default")
    assert chunk_parent(chunks[0]) == ("default", "greeting")
    assert chunks[0].to_embedding_text() == "greeting Say hello and use their name."
    # Nothing to add when the body only repeats the description.
    assert skill_chunks(_skill("greeting", "How to greet a customer.", "How to greet a customer.")) == []


def test_a_chunk_keeps_the_filters_its_skill_is_found_by() -> None:
    skill = _log_skill(category=SkillCategory.GUIDELINE, tags=["observability"], namespace="ops")
    for chunk in skill_chunks(skill):
        assert chunk.category is SkillCategory.GUIDELINE
        assert chunk.tags == ["observability"]


async def test_a_skill_is_found_by_what_only_its_body_says() -> None:
    gantry = await _skills_gantry([_log_skill(), *_other_skills()])
    try:
        results = await gantry.retrieve_skills("narrow by service:checkout and env:prod", limit=3)

        assert results[0].skill.name == "log_search_syntax"
        assert not any(r.skill.is_chunk for r in results)
        assert len({r.skill.name for r in results}) == len(results) == 3
        # The whole skill comes back, not the piece that matched.
        assert results[0].skill.content.startswith("# Reference")
    finally:
        await gantry.close()


async def test_without_chunks_the_body_is_out_of_reach() -> None:
    config = AgentGantryConfig(routing=RoutingConfig(chunk_long_text=False))
    chunked = await _skills_gantry([_log_skill(), *_other_skills()])
    plain = await _skills_gantry([_log_skill(), *_other_skills()], config=config)
    try:

        async def score(gantry: AgentGantry) -> float:
            results = await gantry.retrieve_skills("service:checkout env:prod", limit=20)
            return next(r.score for r in results if r.skill.name == "log_search_syntax")

        assert await score(chunked) > await score(plain) + 0.2
    finally:
        await chunked.close()
        await plain.close()


async def test_skill_chunks_are_not_listed_counted_or_shown_to_a_selector() -> None:
    from agent_gantry.schema.selection import SelectionResult

    seen: list[str] = []

    class _Selector:
        async def select(self, query: str, candidates: Any, limit: int, *, kind: str = "tool"):
            seen.extend(c.name for c in candidates)
            return SelectionResult(selected=[c.id for c in candidates][:limit])

    skills = [_log_skill(), *_other_skills(4)]
    gantry = await _skills_gantry(skills)
    selecting = await _skills_gantry(skills, selector=_Selector())
    try:
        assert await gantry.count_skills() == 5
        assert sorted(s.name for s in await gantry.list_skills()) == sorted(s.name for s in skills)
        assert [s.name for s in await gantry.list_skills(limit=2, offset=1)] == [
            s.name for s in (await gantry.list_skills())[1:3]
        ]
        assert await gantry.count_skills(namespace=chunk_namespace("default")) >= 5

        await selecting.retrieve_skills("anything at all", limit=3)
        assert sorted(seen) == sorted(f"default.{s.name}" for s in skills)
    finally:
        await gantry.close()
        await selecting.close()


async def test_namespace_and_category_filters_reach_the_right_bodies() -> None:
    gantry = await _skills_gantry(
        [
            _log_skill(namespace="ops"),
            _log_skill(namespace="finance", category=SkillCategory.GUIDELINE),
            *_other_skills(),
        ]
    )
    try:
        ops = await gantry.retrieve_skills("service:checkout env:prod", limit=3, namespace="ops")
        assert [(r.skill.namespace, r.skill.name) for r in ops][0] == ("ops", "log_search_syntax")
        assert all(r.skill.namespace == "ops" for r in ops)

        guidelines = await gantry.retrieve_skills(
            "service:checkout env:prod", limit=3, category=SkillCategory.GUIDELINE.value
        )
        assert [(r.skill.namespace, r.skill.name) for r in guidelines] == [
            ("finance", "log_search_syntax")
        ]
    finally:
        await gantry.close()


async def test_a_shorter_body_and_a_deleted_skill_leave_no_chunks() -> None:
    gantry = await _skills_gantry([_log_skill()])
    try:

        async def chunk_count() -> int:
            return await gantry.count_skills(namespace=chunk_namespace("default"))

        assert await chunk_count() >= 2
        await gantry.add_skill(_skill("log_search_syntax", "Logwatch reference card.", "See the wiki."))
        assert await chunk_count() == 1

        assert await gantry.delete_skill("log_search_syntax")
        assert await chunk_count() == 0
        assert await gantry.count_skills() == 0
    finally:
        await gantry.close()


# --------------------------------------------------------------------------- #
# Regressions
# --------------------------------------------------------------------------- #


async def test_many_pieces_of_a_few_skills_do_not_leave_the_result_short() -> None:
    """Eleven skills match, each by 72 pieces: ten are asked for and ten come back.

    Three searches of 40, 160 and 640 rows held the pieces of nine.
    """
    body = "Zebra quagga okapi striped ungulate notes for the keeper on duty today. " * 690
    skills = [_skill(f"rota_{i}", "The keeper rota for the week.", body) for i in range(11)]
    assert all(len(skill_chunks(skill)) == MAX_SKILL_CHUNKS for skill in skills)
    gantry = await _skills_gantry(skills)
    try:
        found = await gantry.retrieve_skills("zebra quagga okapi striped ungulate", limit=10)
        assert len(found) == 10
    finally:
        await gantry.close()


async def test_another_namespaces_chunks_cannot_crowd_a_scoped_search() -> None:
    """Two hundred chunks elsewhere that out-score the one tool the scope holds.

    With one chunk namespace shared by all, they were admitted to the scoped
    search, filled its window three times over and left it with nothing.
    """
    foreign = [
        _tool(
            f"zebra_archive_{i}",
            "Zebra quagga okapi. " * 60 + "Striped ungulate questions go here.",
            namespace="zoo",
        )
        for i in range(40)
    ]
    wanted = _tool("ask_keeper", "Ask the keeper about a striped ungulate.", namespace="ops")
    gantry = await _gantry([*foreign, wanted])
    try:
        assert len(await _chunk_rows(gantry, "zoo")) >= 80
        result = await gantry.retrieve(
            _query("zebra quagga okapi striped ungulate", limit=5, namespaces=["ops"])
        )
        assert _names(result) == ["ask_keeper"]
    finally:
        await gantry.close()


async def test_a_skill_search_scoped_to_a_namespace_is_not_crowded_either() -> None:
    elsewhere = [
        _skill(f"zoo_note_{i}", f"Zoo note number {i} for the keepers.", "Zebra quagga okapi. " * 30, namespace="zoo")
        for i in range(60)
    ]
    here = _skill("keeper_rota", "The keeper rota for the week.", "Zebra duty is on Monday.", namespace="ops")
    gantry = await _skills_gantry([*elsewhere, here])
    try:
        results = await gantry.retrieve_skills("zebra quagga okapi", limit=3, namespace="ops")
        assert [r.skill.name for r in results] == ["keeper_rota"]
        assert await gantry.count_skills() == 61
    finally:
        await gantry.close()


async def test_a_tool_whose_chunks_failed_to_index_is_tried_again() -> None:
    """The tool's fingerprint is what marks it synced, so its chunks go first."""

    class _Flaky(SimpleEmbedder):
        fail = True

        async def embed_batch(self, texts: list[str], batch_size: int | None = None) -> Any:
            if self.fail and any(text.startswith("search_archive ") for text in texts):
                raise RuntimeError("the embedding service is down")
            return await super().embed_batch(texts, batch_size)

    embedder = _Flaky()
    gantry = AgentGantry(embedder=embedder, config=AgentGantryConfig(auto_sync=False))
    try:
        await gantry.add_tool(_long_tool())
        with pytest.raises(RuntimeError, match="embedding service is down"):
            await gantry.sync()
        # Straight from the store: listing through the gantry would sync again.
        assert await gantry._vector_store.list_all(namespace=chunk_namespace("default")) == []
        assert await gantry._vector_store.get_by_name("search_archive") is None

        embedder.fail = False
        assert await gantry.sync() == 1  # not "already up to date"
        assert len(await _chunk_rows(gantry)) >= 2
        result = await gantry.retrieve(_query("zebra quagga okapi striped ungulate", limit=1))
        assert _names(result) == ["search_archive"]
    finally:
        await gantry.close()


async def test_many_pieces_of_a_few_tools_do_not_leave_the_result_short() -> None:
    """Fifteen tools match, each by sixteen pieces: ten are asked for and ten come back.

    The pool was widened by a fixed number of rows per search, so 240 pieces
    that all out-score their tools' own rows filled three searches with the
    pieces of eight tools.
    """
    body = "Zebra quagga okapi striped ungulate notes for the keeper on duty today. " * 138
    tools = [
        _tool(f"ledger_{i}", "Posts a journal line to the ledger.", extended_description=body)
        for i in range(15)
    ]
    assert all(len(tool_chunks(tool)) == MAX_CHUNKS for tool in tools)
    gantry = await _gantry(tools)
    try:
        result = await gantry.retrieve(_query("zebra quagga okapi striped ungulate", limit=10))
        assert len(result.tools) == 10
    finally:
        await gantry.close()


async def test_a_match_on_a_piece_counts_for_less_than_a_match_on_a_tool() -> None:
    """A long tool has many pieces, so many chances to match on incidental text."""
    gantry = await _gantry([_long_tool(), _tool("ping_host", "Check whether a host answers.")])
    try:
        router = gantry._router
        long_tool = await gantry.get_tool("search_archive")
        short_tool = await gantry.get_tool("ping_host")
        piece = tool_chunks(long_tool)[-1]

        # The same raw score from the store: the tool matched whole comes first.
        folded = await router._fold_chunks([(piece, 0.8), (short_tool, 0.8)], _query("anything"))
        scores = {tool.name: score for tool, score, *_ in folded}
        assert scores["ping_host"] == 0.8
        assert scores["search_archive"] == pytest.approx(0.8 * CHUNK_SCORE_WEIGHT)

        # A tool's own score is never lowered by a weaker piece of itself...
        folded = await router._fold_chunks([(long_tool, 0.75), (piece, 0.8)], _query("anything"))
        assert [(t.name, s) for t, s, *_ in folded] == [("search_archive", 0.75)]
        # ...and is still raised by one that is better even after the weight.
        folded = await router._fold_chunks([(long_tool, 0.5), (piece, 0.8)], _query("anything"))
        assert folded[0][1] == pytest.approx(0.8 * CHUNK_SCORE_WEIGHT)
    finally:
        await gantry.close()


async def test_a_store_failure_fetching_a_chunks_tool_is_reported(
    caplog: pytest.LogCaptureFixture,
) -> None:
    noise = [
        _tool(f"zebra_fact_{i}", "Zebra. Zebra facts about a zebra and another zebra.")
        for i in range(60)
    ]
    gantry = await _gantry([_long_tool(), *noise])
    try:

        async def broken(name: str, namespace: str = "default") -> Any:
            raise ConnectionError("store unreachable")

        gantry._vector_store.get_by_name = broken  # type: ignore[method-assign]
        with caplog.at_level("WARNING", logger="agent_gantry.core.router"):
            result = await gantry.retrieve(_query("zebra quagga okapi striped ungulate", limit=5))

        assert "search_archive" not in _names(result)  # left out, and said so
        assert any("default.search_archive" in r.getMessage() for r in caplog.records)
    finally:
        await gantry.close()


async def test_a_store_without_fingerprints_still_loses_a_shortened_tools_chunks() -> None:
    """Such a store answers ``{}`` whatever it holds, which is not "nothing was stored"."""
    from agent_gantry.adapters.vector_stores.base import VectorStoreAdapter
    from agent_gantry.adapters.vector_stores.memory import InMemoryVectorStore
    from agent_gantry.core.registry import ToolRegistry
    from agent_gantry.core.sync_manager import SyncManager

    class _NoFingerprints(InMemoryVectorStore):
        # The protocol's own stub, as a store that never implemented it has.
        get_stored_fingerprints = VectorStoreAdapter.get_stored_fingerprints

    store, embedder = _NoFingerprints(), SimpleEmbedder()
    await store.initialize()
    manager = SyncManager(vector_store=store, embedder=embedder, registry=ToolRegistry())

    await manager.sync_batches([_long_tool()], batch_size=10)
    assert await store.list_all(namespace=chunk_namespace("default")) != []

    await manager.sync_batches([_tool("search_archive", "Search the archive.")], batch_size=10)
    assert await store.list_all(namespace=chunk_namespace("default")) == []


async def test_a_failure_to_delete_a_chunk_is_reported(caplog: pytest.LogCaptureFixture) -> None:
    from agent_gantry.core.chunks import delete_tool_chunks

    class _Store:
        async def delete(self, name: str, namespace: str) -> bool:
            raise ConnectionError("store unreachable")

    with caplog.at_level("WARNING", logger="agent_gantry.core.chunks"):
        assert await delete_tool_chunks(_Store(), "search_archive", "default") == 0  # type: ignore[arg-type]
    assert any("default.search_archive" in r.getMessage() for r in caplog.records)


async def test_turning_chunking_off_does_not_leave_old_pieces_matching() -> None:
    """A store indexed with chunking on, reopened with it off, then updated.

    Skipping the pieces altogether left the old ones in place, still matching
    a query for text the update had removed.
    """
    off = AgentGantryConfig(routing=RoutingConfig(chunk_long_text=False))
    first = await _gantry([_long_tool(), *_distractors()])
    await first.add_skills([_log_skill()])
    second = AgentGantry(embedder=SimpleEmbedder(), vector_store=first._vector_store, config=off)
    try:
        store = first._vector_store
        assert await store.list_all(namespace=chunk_namespace("default"))
        assert await first.count_skills(namespace=chunk_namespace("default")) >= 2

        # The tool loses its last sentence, and the skill its body.
        await second.add_tool(_tool("search_archive", "Search the archive. " + _FILLER * 23))
        await second.sync()
        await second.add_skill(_skill("log_search_syntax", "Logwatch reference card.", "See the wiki."))

        assert await store.list_all(namespace=chunk_namespace("default")) == []
        assert await second.count_skills(namespace=chunk_namespace("default")) == 0
        found = await second.retrieve_skills("service:checkout env:prod", limit=1, score_threshold=0.3)
        assert [r.skill.name for r in found] == []
    finally:
        await second.close()
        await first.close()
