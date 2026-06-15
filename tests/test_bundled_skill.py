"""
Guard tests for the bundled Agent-Gantry Claude Skill.

These keep the skill release-safe: it must ship inside the package, expose a
resolvable path, install via the CLI, and carry valid Anthropic Agent Skills
frontmatter (`name` + `description` within the documented limits).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from agent_gantry.cli.main import main
from agent_gantry.skills import install_to, skill_path

# Anthropic Agent Skills frontmatter limits.
_NAME_MAX = 64
_DESCRIPTION_MAX = 1024


def _read_frontmatter(skill_md: Path) -> dict[str, str]:
    """Parse the simple ``name``/``description`` YAML frontmatter from SKILL.md."""
    text = skill_md.read_text(encoding="utf-8")
    assert text.startswith("---"), "SKILL.md must open with a YAML frontmatter block"
    _, frontmatter, _body = text.split("---", 2)
    name = re.search(r"^name:\s*(.+)$", frontmatter, re.MULTILINE)
    description = re.search(r"^description:\s*(.+)", frontmatter, re.MULTILINE | re.DOTALL)
    assert name is not None, "frontmatter is missing required 'name'"
    assert description is not None, "frontmatter is missing required 'description'"
    return {"name": name.group(1).strip(), "description": description.group(1).strip()}


def test_skill_path_resolves_to_bundled_skill() -> None:
    """The skill ships inside the package and exposes SKILL.md + references/."""
    path = skill_path()
    assert path.is_dir()
    assert (path / "SKILL.md").is_file()
    assert (path / "references" / "cookbook.md").is_file()


def test_skill_frontmatter_is_anthropic_compliant() -> None:
    """name/description exist, are non-empty, and stay within Anthropic limits."""
    fm = _read_frontmatter(skill_path() / "SKILL.md")

    assert fm["name"] == "agent-gantry"
    assert re.fullmatch(r"[a-z0-9-]+", fm["name"]), "name must be lowercase/hyphenated"
    assert 0 < len(fm["name"]) <= _NAME_MAX

    assert fm["description"], "description must be non-empty"
    assert len(fm["description"]) <= _DESCRIPTION_MAX


def test_install_to_copies_the_skill(tmp_path: Path) -> None:
    """install_to() vendors the skill (incl. references/) into a target dir."""
    dst = install_to(tmp_path / "skills")
    assert dst == (tmp_path / "skills" / "agent-gantry").resolve()
    assert (dst / "SKILL.md").is_file()
    assert (dst / "references" / "cookbook.md").is_file()

    # Default is non-destructive: a second call without overwrite must refuse.
    with pytest.raises(FileExistsError):
        install_to(tmp_path / "skills")

    # ...and overwrite=True replaces it cleanly.
    again = install_to(tmp_path / "skills", overwrite=True)
    assert again.is_dir()


def test_cli_install_skill_command(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """`agent-gantry install-skill --target ...` copies the skill and reports it."""
    target = tmp_path / "vendored"
    exit_code = main(["install-skill", "--target", str(target)])
    assert exit_code == 0
    assert (target / "agent-gantry" / "SKILL.md").is_file()
    assert "Installed Agent-Gantry skill" in capsys.readouterr().out


def test_cli_install_skill_print_path(capsys: pytest.CaptureFixture[str]) -> None:
    """`--print-path` prints the bundled skill location without copying."""
    exit_code = main(["install-skill", "--print-path"])
    assert exit_code == 0
    printed = capsys.readouterr().out.strip()
    assert printed == str(skill_path())
