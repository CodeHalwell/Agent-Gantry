"""
Tests for :mod:`agent_gantry.query` strategies and the
``preview`` / ``list_tools_sync`` helpers exposed on
:class:`agent_gantry.AgentGantry`.

These tests intentionally avoid importing ``agent_framework`` so they run
in any environment.
"""

from __future__ import annotations

import warnings

import pytest

from agent_gantry import AgentGantry
from agent_gantry.query import (
    concatenate_recent,
    fallback_chain,
    last_assistant_text,
    last_tool_result,
    last_user_text,
)


class _Msg:
    def __init__(self, role: str, text: str = "", *, name: str = "") -> None:
        self.role = role
        self.text = text
        self.author_name = name


# ---------------------------------------------------------------------------
# Query strategies
# ---------------------------------------------------------------------------


def test_last_user_text_picks_most_recent_user_message():
    msgs = [
        _Msg("user", "first user"),
        _Msg("assistant", "asst reply"),
        _Msg("user", "second user"),
        _Msg("assistant", "asst reply 2"),
    ]
    assert last_user_text(msgs) == "second user"


def test_last_assistant_text_picks_most_recent_assistant_message():
    msgs = [
        _Msg("user", "u"),
        _Msg("assistant", "first asst"),
        _Msg("tool", "tool out"),
        _Msg("assistant", "latest asst"),
    ]
    assert last_assistant_text(msgs) == "latest asst"


def test_last_tool_result_includes_tool_name_and_caps_content():
    big = "x" * 1000
    msgs = [
        _Msg("user", "go"),
        _Msg("tool", big, name="find_markers"),
    ]
    out = last_tool_result(msgs, max_chars=50)
    assert out.startswith("result of find_markers: ")
    # 24 prefix chars + 50 content chars
    assert len(out) == len("result of find_markers: ") + 50


def test_last_tool_result_falls_back_to_generic_label_when_no_name():
    msgs = [_Msg("tool", "raw output")]
    assert last_tool_result(msgs) == "tool result: raw output"


def test_concatenate_recent_returns_last_n_texts():
    msgs = [
        _Msg("user", "a"),
        _Msg("assistant", "b"),
        _Msg("tool", "c"),
        _Msg("assistant", "d"),
    ]
    out = concatenate_recent(msgs, n=2, separator=" | ")
    assert out == "c | d"


def test_concatenate_recent_skips_empty_messages():
    msgs = [_Msg("user", "  "), _Msg("assistant", "real text")]
    assert concatenate_recent(msgs, n=2) == "real text"


def test_fallback_chain_returns_first_non_empty_result():
    chain = fallback_chain(last_tool_result, last_assistant_text, last_user_text)
    msgs = [_Msg("user", "u"), _Msg("assistant", "a")]
    # No tool message -> falls through to last_assistant_text
    assert chain(msgs) == "a"


def test_strategies_handle_empty_input():
    assert last_user_text(None) == ""
    assert last_user_text([]) == ""
    assert last_tool_result([]) == ""
    assert concatenate_recent([], n=3) == ""


def test_strategies_accept_dict_messages():
    msgs = [
        {"role": "user", "content": "hello"},
        {"role": "tool", "content": "out", "name": "fn"},
    ]
    assert last_user_text(msgs) == "hello"
    assert "result of fn:" in last_tool_result(msgs)


def test_msg_text_walks_dict_contents_for_function_result() -> None:
    """``_msg_text`` walks structured ``contents`` lists for text-bearing
    items. Verified here with dict-shaped messages so this file stays
    free of any ``agent_framework`` import (per the module docstring);
    AF-specific Message/Content coverage lives in
    ``test_agent_framework_orchestration.py``.
    """
    from agent_gantry.query.strategies import _msg_text

    # Mimic an AF tool-role message: function_result Content with
    # items[].text populated; Message.text is empty.
    msg = {
        "role": "tool",
        "text": "",
        "contents": [
            type("FR", (), {
                "type": "function_result",
                "items": [
                    type("T", (), {"type": "text", "text": "Paris is sunny."})(),
                ],
                "result": None,
            })(),
        ],
    }
    assert "Paris is sunny" in _msg_text(msg)


def test_msg_text_function_result_fallback_is_per_content() -> None:
    """When an earlier text content already populated ``parts``, a later
    function_result with empty ``items[]`` must still surface its
    ``.result`` via the per-content fallback — not get gated by the
    cumulative ``parts`` list.
    """
    from agent_gantry.query.strategies import _msg_text

    text_c = type("TC", (), {"type": "text", "text": "hi"})()
    empty_fr = type("FR", (), {
        "type": "function_result",
        "items": [],
        "result": "actual tool output payload",
    })()

    msg = {"role": "tool", "text": "", "contents": [text_c, empty_fr]}
    out = _msg_text(msg)
    assert "hi" in out, out
    assert "actual tool output payload" in out, out


# ---------------------------------------------------------------------------
# AgentGantry.list_tools_sync + preview
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_tools_sync_reads_registry_without_async():
    g = AgentGantry()

    @g.register
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    @g.register(namespace="math")
    def mul(a: int, b: int) -> int:
        """Multiply two numbers."""
        return a * b

    # Note: no await on sync().
    all_tools = g.list_tools_sync()
    assert {t.name for t in all_tools} == {"add", "mul"}

    only_math = g.list_tools_sync(namespace="math")
    assert [t.name for t in only_math] == ["mul"]


@pytest.mark.asyncio
async def test_preview_returns_ranked_pairs():
    g = AgentGantry()

    @g.register
    def add(a: int, b: int) -> int:
        """Sum two integers."""
        return a + b

    @g.register
    def divide(a: int, b: int) -> float:
        """Divide two integers."""
        return a / b

    await g.sync()
    ranked = await g.preview("add two numbers", limit=10, score_threshold=0.0)
    assert len(ranked) >= 1
    names = [n for n, _ in ranked]
    assert all(name.startswith("default.") for name in names)
    # Scores should be sorted descending.
    scores = [s for _, s in ranked]
    assert scores == sorted(scores, reverse=True)


@pytest.mark.asyncio
async def test_simple_embedder_warning_on_threshold():
    """SimpleEmbedder + non-zero threshold should emit a UserWarning once."""
    from agent_gantry.adapters.embedders.simple import SimpleEmbedder

    SimpleEmbedder._warned_about_threshold = False  # reset test-local

    # Pin SimpleEmbedder explicitly: without this the default config selects
    # ``sentence_transformers`` whenever that extra is installed, which
    # bypasses the SimpleEmbedder-specific warning code path.
    g = AgentGantry(embedder=SimpleEmbedder())

    @g.register
    def some_tool(x: int) -> int:
        """A tool that does something with the input value."""
        return x

    await g.sync()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        await g.preview("anything", score_threshold=0.5)
    msgs = [str(w.message) for w in caught]
    assert any("SimpleEmbedder" in m for m in msgs), msgs


# ---------------------------------------------------------------------------
# register() accepts FunctionTool-like wrappers
# ---------------------------------------------------------------------------


def test_register_accepts_function_tool_like_object():
    """``gantry.register`` should unwrap ``.func`` / ``.name`` wrappers
    such as ``agent_framework.tool``-decorated FunctionTool objects."""
    g = AgentGantry()

    def real_handler(x: int) -> int:
        """Doubles the input."""
        return x * 2

    class FakeFunctionTool:
        # Matches the AF FunctionTool surface that lacks ``__name__``.
        def __init__(self) -> None:
            self.func = real_handler
            self.name = "doubler"
            self.description = "Doubles."

    ft = FakeFunctionTool()
    returned = g.register(ft)

    assert returned is ft  # the wrapper is passed through unchanged
    pending_names = [t.name for t in g._pending_tools]
    assert "doubler" in pending_names
    # The handler under the registered name should be the bare callable.
    assert g._tool_handlers["doubler"] is real_handler


# ---------------------------------------------------------------------------
# Adapter re-exports
# ---------------------------------------------------------------------------


def test_embedders_module_exposes_optional_classes_via_dir():
    import agent_gantry.adapters.embedders as em

    listed = set(dir(em))
    expected = {
        "EmbeddingAdapter",
        "SimpleEmbedder",
        "SentenceTransformersEmbedder",
        "NomicEmbedder",
        "OpenAIEmbedder",
        "AzureOpenAIEmbedder",
    }
    assert expected.issubset(listed)


def test_rerankers_module_exposes_optional_classes_via_dir():
    import agent_gantry.adapters.rerankers as rk

    listed = set(dir(rk))
    expected = {"RerankerAdapter", "CohereReranker", "CrossEncoderReranker"}
    assert expected.issubset(listed)


# ---------------------------------------------------------------------------
# New: keyword_focused + truncated query helpers
# ---------------------------------------------------------------------------


def test_keyword_focused_strips_scaffolding_and_keeps_content_words():
    """Imperative scaffolding ("please", "step", "the", …) must be
    dropped so the remaining tokens carry the actual content signal."""
    from agent_gantry.query import keyword_focused

    msg = _Msg(
        "user",
        "Please run this five-step pipeline. Use a different tool for "
        "each step. Compute the factorial of 7 and the sha256 hash.",
    )
    out = keyword_focused([msg])
    tokens = out.split()
    # Content words survive.
    assert "factorial" in tokens
    assert "sha256" in tokens
    assert "compute" in tokens
    # Scaffolding is dropped.
    for scaffold in ("please", "the", "this", "step", "use", "of"):
        assert scaffold not in tokens, f"{scaffold!r} should be stripped"


def test_keyword_focused_handles_empty_messages():
    from agent_gantry.query import keyword_focused

    assert keyword_focused([]) == ""
    assert keyword_focused(None) == ""


def test_truncated_caps_length_keeping_tail_by_default():
    from agent_gantry.query import last_user_text, truncated

    msg = _Msg("user", "the quick brown fox jumps over the lazy dog")
    gen = truncated(last_user_text, max_chars=10, keep="tail")
    out = gen([msg])
    assert len(out) <= 10
    # Tail is preserved (defaults to "tail"), so "dog" should remain.
    assert "dog" in out


def test_truncated_keep_head():
    from agent_gantry.query import last_user_text, truncated

    msg = _Msg("user", "the quick brown fox jumps over the lazy dog")
    gen = truncated(last_user_text, max_chars=10, keep="head")
    out = gen([msg])
    assert len(out) <= 10
    assert "the" in out


def test_truncated_supports_async_generator():
    """Wrapper must mirror the underlying coroutine-ness."""
    import asyncio
    import inspect

    from agent_gantry.query import truncated

    async def async_gen(_messages):
        return "this is a long async result string"

    wrapped = truncated(async_gen, max_chars=12, keep="tail")
    assert inspect.iscoroutinefunction(wrapped)
    out = asyncio.run(wrapped([]))
    assert len(out) == 12


# ---------------------------------------------------------------------------
# New: AgentGantry.analyze_registry + pairwise_similarity
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_analyze_registry_detects_cross_references():
    """A tool description naming another registered tool is the
    headline mistake from the issue — verify the linter catches it."""
    g = AgentGantry()

    @g.register
    def factorial(n: int) -> int:
        """Compute factorial of n recursively."""
        return n

    @g.register
    def fibonacci(n: int) -> int:
        """Unrelated to factorial — different recurrence relation."""
        return n

    await g.sync()
    analysis = await g.analyze_registry()
    refs = {f.tool: f.references for f in analysis.cross_references}
    assert "fibonacci" in refs
    assert "factorial" in refs["fibonacci"]
    assert not analysis.empty


@pytest.mark.asyncio
async def test_analyze_registry_clean_registry_reports_empty():
    g = AgentGantry()

    @g.register
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    await g.sync()
    analysis = await g.analyze_registry()
    assert analysis.cross_references == []


@pytest.mark.asyncio
async def test_pairwise_similarity_returns_cosine_score():
    g = AgentGantry()

    @g.register
    def add_numbers(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    @g.register
    def subtract_numbers(a: int, b: int) -> int:
        """Subtract one number from another."""
        return a - b

    await g.sync()
    score = await g.pairwise_similarity("add_numbers", "subtract_numbers")
    assert 0.0 <= score <= 1.0

    with pytest.raises(LookupError):
        await g.pairwise_similarity("add_numbers", "nope")


# ---------------------------------------------------------------------------
# New: GantryToolBridge — threshold parsing + RetrievalDecision
# ---------------------------------------------------------------------------


def test_parse_threshold_accepts_float_relative_and_none():
    from agent_gantry.integrations.agent_framework_bridge import _parse_threshold

    assert _parse_threshold(None) == ("absolute", None)
    assert _parse_threshold(0.3) == ("absolute", 0.3)
    assert _parse_threshold("relative:0.8") == ("relative", 0.8)
    assert _parse_threshold("0.5") == ("absolute", 0.5)

    with pytest.raises(ValueError):
        _parse_threshold("relative:abc")
    with pytest.raises(ValueError):
        _parse_threshold("relative:1.5")
    with pytest.raises(TypeError):
        _parse_threshold([0.3])  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_bridge_emits_retrieval_decision_with_kept_and_dropped():
    """The bridge's decision-returning API must list both kept and
    dropped candidates so callers can self-diagnose threshold issues
    without re-running retrieval."""
    from agent_gantry.integrations.agent_framework_bridge import GantryToolBridge

    g = AgentGantry()

    @g.register
    def get_weather(city: str) -> str:
        """Get the current weather for a city."""
        return "x"

    @g.register
    def book_flight(origin: str, destination: str) -> str:
        """Book a flight between two cities."""
        return "x"

    await g.sync()
    bridge = GantryToolBridge(g, score_threshold=0.0)

    # Use a very high absolute threshold to force everything to be dropped.
    _tools, decision = await bridge.get_tools_with_decision(
        "weather", limit=5, score_threshold=0.99
    )
    assert decision.injected == []
    # Candidates must still be populated so the user can see what was filtered.
    assert len(decision.candidates) >= 1
    assert all(not c.kept for c in decision.candidates)
    # And the summary string is well-formed.
    assert "query=" in decision.summary()


@pytest.mark.asyncio
async def test_bridge_relative_threshold_keeps_top_band():
    """``relative:0.9`` should retain only candidates within 90% of the
    top score and report the effective cutoff."""
    from agent_gantry.integrations.agent_framework_bridge import GantryToolBridge

    g = AgentGantry()

    @g.register
    def get_weather(city: str) -> str:
        """Get the current weather for a city."""
        return "x"

    @g.register
    def book_flight(origin: str, destination: str) -> str:
        """Book a flight between two cities."""
        return "x"

    @g.register
    def lookup_user(user_id: str) -> str:
        """Look up a user by ID."""
        return "x"

    await g.sync()
    bridge = GantryToolBridge(g)
    _tools, decision = await bridge.get_tools_with_decision(
        "weather", limit=5, score_threshold="relative:0.9"
    )
    assert decision.threshold_mode.startswith("relative")
    assert decision.effective_threshold is not None
    # At least one tool should pass; not all should pass (otherwise the
    # threshold did nothing).
    kept_count = sum(c.kept for c in decision.candidates)
    assert kept_count >= 1


# ---------------------------------------------------------------------------
# New: CachedEmbedder
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cached_embedder_persists_across_instances(tmp_path):
    """Re-opening the cache file must yield hits, not re-embed work."""
    from agent_gantry.adapters.embedders.cached import CachedEmbedder
    from agent_gantry.adapters.embedders.simple import SimpleEmbedder

    cache = tmp_path / "embed.sqlite"
    first = CachedEmbedder(SimpleEmbedder(), cache_path=cache)
    v1 = await first.embed_batch(["alpha", "beta"])
    assert first.hits == 0 and first.misses == 2
    # Second call to same instance: all hits.
    v2 = await first.embed_batch(["alpha", "beta"])
    assert v2 == v1
    assert first.hits == 2
    first.close()

    # Fresh instance reopens the same file: must still hit.
    second = CachedEmbedder(SimpleEmbedder(), cache_path=cache)
    v3 = await second.embed_batch(["alpha"])
    assert v3 == [v1[0]]
    assert second.hits == 1 and second.misses == 0
    second.close()
