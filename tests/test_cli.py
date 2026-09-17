"""
The ``agent-gantry`` CLI against real registries (``--module``) and its
``serve-mcp`` plumbing.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from agent_gantry import AgentGantry
from agent_gantry.cli.main import build_gantry, main


def test_single_module_uses_the_instance_itself() -> None:
    from tests.test_modules import module_a

    gantry = build_gantry(["tests.test_modules.module_a"])
    assert gantry is module_a.tools

    custom = build_gantry(["tests.test_modules.module_custom_attr:my_custom_tools"])
    assert custom.tool_count == 1
    by_attr = build_gantry(["tests.test_modules.module_custom_attr"], attr="my_custom_tools")
    assert by_attr is custom


def test_several_modules_are_merged_into_one_registry() -> None:
    gantry = build_gantry(
        ["tests.test_modules.module_a", "tests.test_modules.module_custom_attr:my_custom_tools"]
    )
    names = sorted(t.name for t in gantry.list_tools_sync())
    assert names == ["custom_tool", "tool_a1", "tool_a2"]


def test_a_duplicate_across_modules_warns_and_keeps_the_first(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """``collect_tools_from_modules`` dedupes per call, so collecting one
    module at a time reset its ``seen`` set and the later module silently
    overwrote the earlier tool's definition and handler."""
    import logging

    with caplog.at_level(logging.WARNING):
        gantry = build_gantry(
            [
                "tests.test_modules.module_a",
                "tests.test_modules.module_c_duplicate",
            ]
        )
    assert "Skipping duplicate tool 'default.tool_a1'" in caplog.text
    handler = gantry._registry.get_handler("default.tool_a1")
    assert handler is not None
    assert handler.__module__ == "tests.test_modules.module_a", "the first module must win"


def test_missing_module_or_attribute_is_a_clear_error(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit, match="could not import module"):
        build_gantry(["tests.test_modules.does_not_exist"])
    with pytest.raises(SystemExit, match="no AgentGantry at attribute 'tools'"):
        build_gantry(["tests.test_modules.module_no_tools"])


def test_demo_registry_is_announced_on_stderr(capsys: pytest.CaptureFixture[str]) -> None:
    gantry = build_gantry(None)
    assert isinstance(gantry, AgentGantry)
    assert gantry.tool_count == 3
    assert "demo tools" in capsys.readouterr().err
    build_gantry(None, quiet=True)
    assert capsys.readouterr().err == ""


def test_list_and_search_commands(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["list", "--module", "tests.test_modules.module_a"]) == 0
    out = capsys.readouterr().out
    assert "default.tool_a1: First tool from module A." in out
    assert out.index("tool_a1") < out.index("tool_a2")

    assert (
        main(["search", "second tool", "--module", "tests.test_modules.module_a", "--limit", "1"])
        == 0
    )
    out = capsys.readouterr().out
    assert out.count("\n") == 1 and "default.tool_a" in out

    assert (
        main(["search", "x", "--module", "tests.test_modules.module_a", "--namespace", "nope"]) == 0
    )
    assert capsys.readouterr().out.strip() == "No tools found."


def test_sync_dry_run_reports_new_and_stale_tools(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["sync", "--dry-run", "--prune", "--module", "tests.test_modules.module_b"]) == 0
    out = capsys.readouterr().out
    assert "would be (re-)embedded" in out
    assert "No stale tools to prune." in out


def test_prune_defers_to_the_config_when_the_flag_is_absent(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``--prune`` defaulted to False rather than None, so it overrode
    ``sync()``'s "None means use the config" and a config asking for pruning
    was ignored — in the dry-run report as well as the real sync."""
    config = tmp_path / "gantry.yaml"
    config.write_text("prune_on_sync: true\n", encoding="utf-8")
    assert (
        main(
            [
                "sync",
                "--dry-run",
                "--config",
                str(config),
                "--module",
                "tests.test_modules.module_a",
            ]
        )
        == 0
    )
    assert "prune" in capsys.readouterr().out.lower()

    # ...and without the config setting, a dry run says nothing about pruning
    assert main(["sync", "--dry-run", "--module", "tests.test_modules.module_a"]) == 0
    assert "prune" not in capsys.readouterr().out.lower()


def test_the_command_runs_on_one_event_loop() -> None:
    """The gantry has to be built on the loop its command then runs on: a
    loop-bound backend (a pgvector pool) initialised under one ``asyncio.run``
    and used under a second is talking to a closed loop."""
    loops: list[int] = []
    real_retrieve = AgentGantry.retrieve

    async def spy(self: AgentGantry, query: Any) -> Any:
        loops.append(id(asyncio.get_running_loop()))
        return await real_retrieve(self, query)

    original = AgentGantry.collect_tools_from_modules

    async def collect_spy(self: AgentGantry, *args: Any, **kwargs: Any) -> Any:
        loops.append(id(asyncio.get_running_loop()))
        return await original(self, *args, **kwargs)

    with (
        patch.object(AgentGantry, "retrieve", spy),
        patch.object(AgentGantry, "collect_tools_from_modules", collect_spy),
    ):
        # two --module entries force the collect path, which used to run in
        # its own asyncio.run() before the command opened another
        assert (
            main(
                [
                    "search",
                    "first tool",
                    "--module",
                    "tests.test_modules.module_a",
                    "--module",
                    "tests.test_modules.module_b",
                ]
            )
            == 0
        )
    assert len(loops) >= 2
    assert len(set(loops)) == 1, "collection and the command must share one loop"


def test_serve_mcp_plumbs_transport_options() -> None:
    with patch.object(AgentGantry, "serve_mcp", new_callable=AsyncMock) as serve:
        assert (
            main(
                [
                    "serve-mcp",
                    "--module",
                    "tests.test_modules.module_a",
                    "--transport",
                    "http",
                    "--mode",
                    "hybrid",
                    "--expose",
                    "tool_a1",
                    "--port",
                    "8765",
                ]
            )
            == 0
        )
    serve.assert_awaited_once_with(
        transport="streamable_http",
        mode="hybrid",
        name="agent-gantry",
        host="127.0.0.1",
        port=8765,
        path=None,
        expose=["tool_a1"],
    )


def test_serve_mcp_advertises_the_endpoint_it_will_serve(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The status line always claimed ``/sse`` for the SSE transport, so a user
    who passed ``--path`` was told to connect to a route the server was not
    serving. Each transport's own default stands in for an omitted ``--path``."""

    def _serve(*args: str) -> str:
        with patch.object(AgentGantry, "serve_mcp", new_callable=AsyncMock):
            assert main(["serve-mcp", "--module", "tests.test_modules.module_a", *args]) == 0
        return capsys.readouterr().err

    assert "http://127.0.0.1:8000/custom" in _serve("--transport", "sse", "--path", "/custom")
    assert "http://127.0.0.1:8000/sse" in _serve("--transport", "sse")
    assert "http://127.0.0.1:8000/custom" in _serve("--transport", "http", "--path", "/custom")
    assert "http://127.0.0.1:8000/mcp" in _serve("--transport", "http")

    # Both transports route on ``"/" + path.strip("/")``, so a path given
    # without a leading slash must print the route the server will serve --
    # the raw argument gave the unusable ``http://127.0.0.1:8000custom``.
    assert "http://127.0.0.1:8000/custom" in _serve("--transport", "http", "--path", "custom")
    assert "http://127.0.0.1:8000/custom" in _serve("--transport", "sse", "--path", "custom/")


def test_the_sync_builder_refuses_to_persist(tmp_path: Path) -> None:
    """``build_gantry`` initialises inside an ``asyncio.run`` it then closes,
    so persisting there would hand back a loop-bound backend (a pgvector pool)
    holding a closed loop, to fail on first use in the caller's own."""
    config = tmp_path / "gantry.yaml"
    config.write_text("auto_sync: false\n", encoding="utf-8")
    with pytest.raises(ValueError, match="build_gantry_async"):
        build_gantry(
            ["tests.test_modules.module_a"], config=str(config), persist=True
        )
    # the read-only default still works
    gantry = build_gantry(["tests.test_modules.module_a"], config=str(config))
    assert gantry.tool_count == 2


def test_a_duplicate_under_a_different_attribute_is_still_caught(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Specs naming different attributes were collected in separate passes,
    and ``collect_tools_from_modules`` starts a fresh ``seen`` set per call, so
    a cross-pass duplicate slipped through and the later module silently
    overwrote the earlier tool's definition and handler."""
    import logging

    with caplog.at_level(logging.WARNING):
        gantry = build_gantry(
            [
                "tests.test_modules.module_a",
                "tests.test_modules.module_d_duplicate_attr:other_tools",
            ]
        )
    assert "Skipping duplicate tool 'default.tool_a1'" in caplog.text
    handler = gantry._registry.get_handler("default.tool_a1")
    assert handler is not None
    assert handler.__module__ == "tests.test_modules.module_a", "the first module must win"


def test_building_the_gantry_does_not_write_to_the_store(tmp_path: Path) -> None:
    """Collecting modules embedded every tool and upserted it into the
    configured store, before the command had decided anything. ``sync
    --dry-run`` promises to report what *would* be embedded, so with
    ``--config`` pointing at a real backend it mutated the very state it was
    asked only to inspect -- and then reported the tools as already current.
    """
    config = tmp_path / "gantry.yaml"
    config.write_text("auto_sync: false\n", encoding="utf-8")
    gantry = build_gantry(
        ["tests.test_modules.module_a", "tests.test_modules.module_b"], config=str(config)
    )

    async def inspect() -> list[Any]:
        await gantry._ensure_initialized()
        return await gantry._vector_store.list_all(limit=100)

    assert asyncio.run(inspect()) == [], "building must not embed or store anything"
    # ...but the tools are all there to be inspected, and a later sync stores
    # them, so nothing is lost by holding the write back.
    assert gantry.tool_count == 4
    assert len(asyncio.run(_sync_and_list(gantry))) == 4


async def _sync_and_list(gantry: AgentGantry) -> list[Any]:
    await gantry.sync()
    return await gantry._vector_store.list_all(limit=100)


def test_a_dry_run_leaves_a_persistent_store_untouched(tmp_path: Path) -> None:
    """The end-to-end shape of the above: two dry runs in a row against one
    store both report the same work as outstanding, because neither did it."""
    store = tmp_path / "store"
    config = tmp_path / "gantry.yaml"
    config.write_text(
        f"vector_store:\n  type: lancedb\n  db_path: {store}\n", encoding="utf-8"
    )
    pytest.importorskip("lancedb")

    for _ in range(2):
        assert (
            main(
                [
                    "sync",
                    "--dry-run",
                    "--config",
                    str(config),
                    "--module",
                    "tests.test_modules.module_a",
                ]
            )
            == 0
        )
    # LanceDB lays out its directory on connect, so count rows, not files.
    async def rows() -> list[Any]:
        from agent_gantry.schema.config import AgentGantryConfig

        probe = AgentGantry(config=AgentGantryConfig.from_yaml(str(config)))
        await probe._ensure_initialized()
        try:
            return await probe._vector_store.list_all(limit=100)
        finally:
            await probe.close()

    assert asyncio.run(rows()) == [], "a dry run must not embed or store anything"


def test_config_option_builds_the_gantry_from_yaml(tmp_path: Path) -> None:
    config = tmp_path / "gantry.yaml"
    config.write_text("auto_sync: false\nexecution:\n  max_retries: 1\n", encoding="utf-8")
    gantry = build_gantry(
        ["tests.test_modules.module_a", "tests.test_modules.module_b"], config=str(config)
    )
    assert gantry._config.auto_sync is False
    assert gantry._config.execution.max_retries == 1
    assert gantry.tool_count == 4


def test_module_specs_resolve_against_the_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """``agent-gantry`` is an installed console script, so the directory it is
    run from is not on ``sys.path`` the way ``python -m`` would put it there.
    ``--module pkg.tools`` from a project root -- the flag's own documented
    use -- failed against a local, uninstalled package."""
    package = tmp_path / "localpkg"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "tools.py").write_text(
        "from agent_gantry import AgentGantry\n\ntools = AgentGantry()\n"
    )

    monkeypatch.chdir(tmp_path)
    monkeypatch.delitem(sys.modules, "localpkg", raising=False)
    monkeypatch.delitem(sys.modules, "localpkg.tools", raising=False)
    path_before = list(sys.path)
    try:
        assert main(["list", "--module", "localpkg.tools"]) == 0
    finally:
        sys.path[:] = path_before


def test_a_bad_config_is_reported_as_one_line(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A missing file or malformed YAML reached the user as a raw traceback,
    while the neighbouring ``--module`` failure was a single line."""
    with pytest.raises(SystemExit) as missing:
        main(["list", "--config", str(tmp_path / "nope.yaml")])
    assert "could not load config" in str(missing.value)

    malformed = tmp_path / "bad.yaml"
    malformed.write_text("a: [1,\n")
    with pytest.raises(SystemExit) as broken:
        main(["list", "--config", str(malformed)])
    assert "could not load config" in str(broken.value)
