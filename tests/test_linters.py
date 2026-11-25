"""Lint rules added for #434: missing examples, inverted thresholds, unclosed gantries."""

from __future__ import annotations

from pathlib import Path

import pytest

from dotfetch import DotFetch
from dotfetch.cli.main import main
from dotfetch.utils.source_linter import analyze_paths, analyze_source

# --- source linter: score_threshold ------------------------------------------


def test_an_inverted_threshold_comment_is_flagged() -> None:
    src = (
        "async def go(fetcher, query):\n"
        "    # Using score_threshold=0.1 for SimpleEmbedder compatibility\n"
        "    return await fetcher.retrieve_tools(query, limit=1, score_threshold=0.1)\n"
    )
    analysis = analyze_source(src, "demo.py")
    assert len(analysis.thresholds) == 1
    finding = analysis.thresholds[0]
    assert finding.line == 3 and finding.value == pytest.approx(0.1) and finding.inverted


def test_a_bare_non_zero_threshold_is_flagged() -> None:
    src = "async def go(fetcher, q):\n    return await fetcher.retrieve_tools(q, score_threshold=0.4)\n"
    analysis = analyze_source(src, "demo.py")
    assert [(f.line, f.inverted) for f in analysis.thresholds] == [(2, False)]


def test_a_justified_threshold_is_not_flagged() -> None:
    src = (
        "async def go(fetcher, q):\n"
        "    # Tighten the cutoff: this demo shows what a strict threshold drops.\n"
        "    return await fetcher.retrieve_tools(q, score_threshold=0.4)\n"
    )
    assert analyze_source(src, "demo.py").empty


def test_a_zero_threshold_and_a_non_literal_are_not_flagged() -> None:
    src = (
        "async def go(fetcher, q, t):\n"
        "    await fetcher.retrieve_tools(q, score_threshold=0.0)\n"
        "    await fetcher.retrieve_tools(q, score_threshold=t)\n"
    )
    assert analyze_source(src, "demo.py").empty


def test_a_comment_above_a_multiline_call_counts() -> None:
    src = (
        "from dotfetch.schema.query import ToolQuery\n"
        "def q(ctx):\n"
        "    # Deliberately strict: the filtering demo.\n"
        "    return ToolQuery(\n"
        "        context=ctx,\n"
        "        score_threshold=0.3,\n"
        "    )\n"
    )
    assert analyze_source(src, "demo.py").empty


# --- source linter: unclosed gantries -----------------------------------------

_MAIN = '\nif __name__ == "__main__":\n    import asyncio\n    asyncio.run(run())\n'


def test_a_main_script_that_never_closes_its_fetcher_is_flagged() -> None:
    src = "from dotfetch import DotFetch\n\nasync def run():\n    fetcher = DotFetch()\n" + _MAIN
    analysis = analyze_source(src, "demo.py")
    assert [f.line for f in analysis.unclosed] == [4]


def test_a_main_script_that_closes_its_fetcher_is_clean() -> None:
    src = (
        "from dotfetch import DotFetch\n\nasync def run():\n    fetcher = DotFetch()\n"
        "    try:\n        pass\n    finally:\n        await fetcher.close()\n" + _MAIN
    )
    assert analyze_source(src, "demo.py").empty


def test_an_async_with_fetcher_counts_as_closed() -> None:
    src = (
        "from dotfetch import DotFetch\n\nasync def run():\n"
        "    async with DotFetch() as fetcher:\n        pass\n" + _MAIN
    )
    assert analyze_source(src, "demo.py").empty


def test_quick_start_and_from_config_are_constructions_too() -> None:
    src = (
        "from dotfetch import DotFetch\n\nasync def run():\n"
        "    fetcher = DotFetch.quick_start()\n    other = DotFetch.from_config('x.yaml')\n"
        + _MAIN
    )
    assert [f.line for f in analyze_source(src, "demo.py").unclosed] == [4]


def test_a_library_module_without_a_main_guard_is_not_flagged() -> None:
    src = "from dotfetch import DotFetch\n\nfetcher = DotFetch()\n"
    assert analyze_source(src, "tools.py").empty


def test_analyze_paths_walks_directories(tmp_path: Path) -> None:
    (tmp_path / "ok.py").write_text("x = 1\n")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "bad.py").write_text(
        "async def go(g, q):\n    return await g.retrieve_tools(q, score_threshold=0.5)\n"
    )
    analysis = analyze_paths([tmp_path])
    assert analysis.files_scanned == 2
    assert [f.path for f in analysis.thresholds] == [str(tmp_path / "sub" / "bad.py")]
    assert "score_threshold=0.5" in analysis.format_text()


# --- registry linter: missing examples ----------------------------------------


async def test_a_tool_without_examples_is_reported() -> None:
    fetcher = DotFetch()
    try:

        @fetcher.register(tags=["math"])
        async def add(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

        @fetcher.register(tags=["math"], examples=["what is 2 plus 2"])
        async def subtract(a: int, b: int) -> int:
            """Subtract one number from another."""
            return a - b

        analysis = await fetcher.analyze_registry()
        assert [f.tool for f in analysis.missing_examples] == ["default.add"]
        assert not analysis.empty
        assert "default.add" in analysis.format_text()
    finally:
        await fetcher.close()


# --- CLI ---------------------------------------------------------------------


def test_lint_source_scans_files_without_building_a_fetcher(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bad = tmp_path / "demo.py"
    bad.write_text(
        "from dotfetch import DotFetch\n\nasync def run():\n    fetcher = DotFetch()\n"
        "    # score_threshold=0.1 for SimpleEmbedder compatibility\n"
        "    await fetcher.retrieve_tools('x', score_threshold=0.1)\n" + _MAIN
    )
    assert main(["lint", "--source", str(bad)]) == 1
    out = capsys.readouterr().out
    assert "score_threshold=0.1" in out and "never closed" in out

    good = tmp_path / "clean.py"
    good.write_text("x = 1\n")
    assert main(["lint", "--source", str(good)]) == 0
    assert "No issues found" in capsys.readouterr().out


def test_lint_source_with_a_missing_path_is_an_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["lint", "--source", str(tmp_path / "nope.py")]) == 2
    assert "nope.py" in capsys.readouterr().err
