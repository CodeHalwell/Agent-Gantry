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


def test_last_tool_result_extracts_text_from_af_function_result_message() -> None:
    """AF wraps tool output as a ``function_result`` Content inside a
    tool-role Message — ``Message.text`` is empty in that case and the
    payload lives in ``Content.items[].text``. ``_msg_text`` must walk
    ``contents`` to surface it, otherwise ``last_tool_result`` returns
    "" and the per-call refresh's query collapses back to the original
    user prompt across the entire run.
    """
    af = pytest.importorskip("agent_framework")

    fr = af.Content.from_function_result(
        call_id="r0",
        result="Paris is sunny. Invoice INV-9 payment is overdue.",
    )
    msg = af.Message(role="tool", contents=[fr])

    out = last_tool_result([msg])
    assert "Paris is sunny" in out, out
    assert "INV-9" in out, out
    # AF Messages don't carry a tool name at the message level; the
    # generic-label fallback should kick in.
    assert out.startswith("tool result:") or out.startswith("result of"), out


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
