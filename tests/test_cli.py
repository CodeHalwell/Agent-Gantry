"""
The ``agent-gantry`` CLI against real registries (``--module``) and its
``serve-mcp`` plumbing.
"""

from __future__ import annotations

from pathlib import Path
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
        path="/mcp",
        expose=["tool_a1"],
    )


def test_config_option_builds_the_gantry_from_yaml(tmp_path: Path) -> None:
    config = tmp_path / "gantry.yaml"
    config.write_text("auto_sync: false\nexecution:\n  max_retries: 1\n", encoding="utf-8")
    gantry = build_gantry(
        ["tests.test_modules.module_a", "tests.test_modules.module_b"], config=str(config)
    )
    assert gantry._config.auto_sync is False
    assert gantry._config.execution.max_retries == 1
    assert gantry.tool_count == 4
