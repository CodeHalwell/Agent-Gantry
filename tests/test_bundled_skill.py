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
import yaml

from agent_gantry.cli.main import main
from agent_gantry.skills import install_to, skill_path

# Anthropic Agent Skills frontmatter limits.
_NAME_MAX = 64
_DESCRIPTION_MAX = 1024


def _read_frontmatter(skill_md: Path) -> dict[str, str]:
    """Parse the ``name``/``description`` YAML frontmatter from SKILL.md."""
    text = skill_md.read_text(encoding="utf-8")
    assert text.startswith("---"), "SKILL.md must open with a YAML frontmatter block"
    # Split only on lines that are exactly "---" so a literal "---" inside the
    # description can never carve up the frontmatter incorrectly.
    _, frontmatter, _body = re.split(r"^---\s*$", text, maxsplit=2, flags=re.MULTILINE)
    data = yaml.safe_load(frontmatter)
    assert isinstance(data, dict), "frontmatter must be a YAML mapping"
    assert "name" in data, "frontmatter is missing required 'name'"
    assert "description" in data, "frontmatter is missing required 'description'"
    return {"name": str(data["name"]).strip(), "description": str(data["description"]).strip()}


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


def test_cli_install_skill_default_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With neither --target nor --claude, the default ./skills target is used."""
    monkeypatch.chdir(tmp_path)
    assert main(["install-skill"]) == 0
    assert (tmp_path / "skills" / "agent-gantry" / "SKILL.md").is_file()


def test_cli_install_skill_claude_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`--claude` installs into ~/.claude/skills, expanding ~ via expanduser()."""
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    # Path.expanduser() reads HOME on POSIX and USERPROFILE on Windows.
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setenv("USERPROFILE", str(fake_home))

    exit_code = main(["install-skill", "--claude"])
    assert exit_code == 0
    assert (fake_home / ".claude" / "skills" / "agent-gantry" / "SKILL.md").is_file()
    # No literal "~" directory leaked into the cwd.
    assert not Path("~").exists()
    assert "Installed Agent-Gantry skill" in capsys.readouterr().out


def test_cli_install_skill_claude_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A second --claude install with --overwrite replaces the existing copy."""
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setenv("USERPROFILE", str(fake_home))

    assert main(["install-skill", "--claude"]) == 0
    # Without --overwrite the second install must refuse (non-zero exit).
    assert main(["install-skill", "--claude"]) == 2
    # With --overwrite it succeeds and the skill is still present.
    assert main(["install-skill", "--claude", "--overwrite"]) == 0
    assert (fake_home / ".claude" / "skills" / "agent-gantry" / "SKILL.md").is_file()


def test_cli_install_skill_rejects_target_and_claude_together(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """--target and --claude are mutually exclusive (argparse exits with code 2)."""
    with pytest.raises(SystemExit) as exc:
        main(["install-skill", "--target", "./x", "--claude"])
    # Exit code 2 is argparse's documented "usage error" contract; the exact
    # wording of the conflict message is an internal detail, so assert only that
    # the offending option is named in stderr rather than pinning the phrasing.
    assert exc.value.code == 2
    assert "--claude" in capsys.readouterr().err


def test_cli_install_skill_print_path(capsys: pytest.CaptureFixture[str]) -> None:
    """`--print-path` prints the bundled skill location without copying."""
    exit_code = main(["install-skill", "--print-path"])
    assert exit_code == 0
    printed = capsys.readouterr().out.strip()
    assert printed == str(skill_path())


# ---------------------------------------------------------------------------
# The bundled skill must fit the limits this library applies to skills it
# loads. At 80k characters its body tripped the loader's own 50k cap, so a
# registry that indexed it stored (and injected) only the first two-thirds.
# The per-framework material now lives in ``references/``; these keep the hub
# within the cap and its links pointing at real files and headings.
# ---------------------------------------------------------------------------


def _anchor(heading: str) -> str:
    """GitHub's anchor for a Markdown heading."""
    title = heading.lstrip("#").strip().lower()
    return re.sub(r"[^a-z0-9 \-_]", "", title).replace(" ", "-")


def _anchors(markdown: str) -> set[str]:
    anchors, fenced = set(), False
    for line in markdown.splitlines():
        if line.lstrip().startswith("```"):
            fenced = not fenced
        elif not fenced and re.match(r"#{1,6} ", line):
            anchors.add(_anchor(line))
    return anchors


def test_bundled_skill_fits_the_loader_body_limit(caplog: pytest.LogCaptureFixture) -> None:
    import logging

    from agent_gantry.skills import load_skill
    from agent_gantry.skills.loader import _CONTENT_MAX, parse_skill_markdown

    text = (skill_path() / "SKILL.md").read_text(encoding="utf-8")
    _, body = parse_skill_markdown(text)
    assert len(body.strip()) <= _CONTENT_MAX, (
        "SKILL.md body is over the loader's cap; move detail into references/"
    )
    with caplog.at_level(logging.WARNING, logger="agent_gantry.skills.loader"):
        skill = load_skill(skill_path())
    assert "truncated" not in caplog.text
    assert skill.content.rstrip().endswith(body.strip()[-200:])  # stored whole


def test_bundled_skill_links_resolve() -> None:
    """Every relative link in the skill names a file that ships, and every
    ``#anchor`` a heading that exists in the file it points into."""
    root = skill_path()
    files = [root / "SKILL.md", *sorted((root / "references").glob("*.md"))]
    assert len(files) > 2
    anchors = {f: _anchors(f.read_text(encoding="utf-8")) for f in files}

    broken = []
    for source in files:
        text = source.read_text(encoding="utf-8")
        for target in re.findall(r"\]\(([^)\s]+)\)", text):
            if re.match(r"[a-z][a-z0-9+.-]*:", target):  # http(s), mailto, ...
                continue
            path, _, fragment = target.partition("#")
            destination = (source.parent / path).resolve() if path else source
            if not destination.is_file():
                broken.append(f"{source.name}: {target} (no such file)")
            elif fragment and fragment not in anchors.get(destination, set()):
                broken.append(f"{source.name}: {target} (no such heading)")
        for name in re.findall(r"`references/([a-z0-9-]+\.md)`", text):
            if not (root / "references" / name).is_file():
                broken.append(f"{source.name}: references/{name} (no such file)")
    assert not broken, broken


def test_install_to_copies_every_reference(tmp_path: Path) -> None:
    dst = install_to(tmp_path / "skills")
    shipped = {f.name for f in (skill_path() / "references").glob("*.md")}
    assert {f.name for f in (dst / "references").glob("*.md")} == shipped
    assert {"openai-agents.md", "selection.md", "api-reference.md"} <= shipped
