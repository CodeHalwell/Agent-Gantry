"""
Loading Agent Skills (``SKILL.md``) directories into the semantic skill store.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from agent_gantry import AgentGantry, SkillCategory
from agent_gantry.skills import (
    SkillParseError,
    load_skill,
    load_skills_from_directory,
    skill_from_markdown,
    skill_path,
)
from agent_gantry.skills.loader import iter_skill_files, parse_skill_markdown

_PDF_SKILL = """---
name: pdf-tools
description: Fill, merge, split and OCR PDF files. Use whenever a .pdf is involved.
tags: [pdf, documents]
allowed-tools: Bash(pdftotext:*) Read
license: MIT
metadata:
  author: docs-team
---

# PDF tools

Use `pypdf` for merging and `pdfplumber` for text extraction.

## OCR

Run `ocrmypdf` first when the PDF is scanned.
"""


def _write_skill(root: Path, name: str, text: str) -> Path:
    directory = root / name
    directory.mkdir(parents=True)
    (directory / "SKILL.md").write_text(text, encoding="utf-8")
    return directory


def test_parse_frontmatter_and_body() -> None:
    frontmatter, body = parse_skill_markdown(_PDF_SKILL)
    assert frontmatter["name"] == "pdf-tools"
    assert frontmatter["metadata"] == {"author": "docs-team"}
    assert body.lstrip().startswith("# PDF tools")

    assert parse_skill_markdown("no frontmatter here") == ({}, "no frontmatter here")
    with pytest.raises(SkillParseError, match="invalid YAML"):
        parse_skill_markdown("---\nname: [unclosed\n---\nbody")
    with pytest.raises(SkillParseError, match="mapping"):
        parse_skill_markdown("---\n- just\n- a list\n---\nbody")


def test_skill_from_markdown_maps_the_agent_skills_fields() -> None:
    skill = skill_from_markdown(_PDF_SKILL, default_name="ignored", namespace="docs")
    assert skill.name == "pdf-tools"
    assert skill.namespace == "docs"
    assert skill.description.startswith("Fill, merge, split and OCR PDF files.")
    assert skill.content.startswith("# PDF tools")
    assert skill.category is SkillCategory.HOW_TO
    assert skill.tags == ["pdf", "documents"]
    # Claude Code's allowed-tools is the closest thing to related_tools
    assert skill.related_tools == ["Bash(pdftotext:*)", "Read"]
    assert skill.source == "skill_md"
    # Unknown frontmatter keys travel along in metadata
    assert skill.metadata["license"] == "MIT"
    assert skill.metadata["metadata"] == {"author": "docs-team"}
    assert skill.metadata["format"] == "agent-skills"
    # The embedding text carries the frontmatter, not the body
    assert "pypdf" not in skill.to_embedding_text()
    assert "pdf documents" in skill.to_embedding_text()


def test_skill_from_markdown_fills_gaps_and_respects_limits(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # No frontmatter at all: name from the directory, description from the body
    skill = skill_from_markdown(
        "# Title\n\nTurn CSV files into charts quickly.\n", default_name="csv-charts"
    )
    assert skill.name == "csv-charts"
    assert skill.description == "Turn CSV files into charts quickly."

    # A tiny description is padded rather than rejected
    skill = skill_from_markdown("---\nname: x\ndescription: Hi\n---\nbody text", default_name="x")
    assert len(skill.description) >= 10 and "Hi" in skill.description

    # Category names map onto SkillCategory; unknown ones fall back with a warning
    skill = skill_from_markdown(
        "---\nname: x\ndescription: A workflow for releasing\ncategory: WORKFLOW\n---\nbody",
        default_name="x",
    )
    assert skill.category is SkillCategory.WORKFLOW
    with caplog.at_level(logging.WARNING):
        skill = skill_from_markdown(
            "---\nname: x\ndescription: A skill with a bad category\ncategory: nope\n---\nbody",
            default_name="x",
        )
    assert skill.category is SkillCategory.HOW_TO
    assert "unknown category" in caplog.text

    # Over-long bodies are truncated to the schema ceiling with a warning
    with caplog.at_level(logging.WARNING):
        skill = skill_from_markdown(
            "---\nname: big\ndescription: A very large skill body\n---\n" + ("word " * 20000),
            default_name="big",
        )
    assert len(skill.content) <= 50000
    assert "truncated" in caplog.text


def test_load_skills_from_directory_walks_and_skips(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    _write_skill(tmp_path, "pdf-tools", _PDF_SKILL)
    _write_skill(
        tmp_path,
        "nested/xlsx",
        "---\nname: xlsx\ndescription: Build spreadsheets with formulas\n---\nUse openpyxl.",
    )
    _write_skill(
        tmp_path,
        ".hidden/secret",
        "---\nname: secret\ndescription: Should be skipped entirely\n---\nx",
    )
    _write_skill(tmp_path, "broken", "---\nname: [oops\n---\nx")
    (tmp_path / "README.md").write_text("not a skill")

    with caplog.at_level(logging.WARNING):
        skills = load_skills_from_directory(tmp_path, namespace="team")
    assert [s.name for s in skills] == ["xlsx", "pdf-tools"]
    assert all(s.namespace == "team" for s in skills)
    assert "Skipping skill" in caplog.text

    assert [s.name for s in load_skills_from_directory(tmp_path, recursive=False)] == ["pdf-tools"]
    with pytest.raises(SkillParseError, match="broken"):
        load_skills_from_directory(tmp_path, strict=True)
    with pytest.raises(FileNotFoundError):
        load_skills_from_directory(tmp_path / "missing")

    # A single skill directory (or its SKILL.md) works too
    assert load_skill(tmp_path / "pdf-tools").name == "pdf-tools"
    assert load_skill(tmp_path / "pdf-tools" / "SKILL.md").source_uri is not None
    assert [p.parent.name for p in iter_skill_files(tmp_path / "pdf-tools")] == ["pdf-tools"]
    with pytest.raises(FileNotFoundError):
        load_skill(tmp_path / "does-not-exist")


@pytest.mark.asyncio
async def test_gantry_add_skills_from_directory_retrieves_by_meaning(tmp_path: Path) -> None:
    _write_skill(tmp_path, "pdf-tools", _PDF_SKILL)
    _write_skill(
        tmp_path,
        "deploy",
        "---\nname: deploy\ndescription: Ship a release to production with the deploy pipeline\n"
        "tags: [release, deploy]\n---\nRun the release checklist, then `make deploy`.",
    )
    gantry = AgentGantry()
    assert await gantry.add_skills_from_directory(str(tmp_path)) == 2
    assert await gantry.count_skills() == 2

    prompt = await gantry.retrieve_skills_as_prompt("merge two pdf documents", limit=1)
    assert prompt.startswith("## Pdf Tools")
    assert "pypdf" in prompt  # the body is injected verbatim
    assert "make deploy" not in prompt

    prompt = await gantry.retrieve_skills_as_prompt("deploy the release to production", limit=1)
    assert "make deploy" in prompt


def test_bundled_skill_loads_through_the_loader() -> None:
    """The library's own SKILL.md is a valid Agent Skill for the loader."""
    skills = load_skills_from_directory(skill_path().parent)
    assert [s.name for s in skills] == ["agent-gantry"]
    assert "semantic" in skills[0].description.lower()
    assert skills[0].content.startswith("# Agent-Gantry")


def test_a_model_rejection_is_reported_as_a_skill_parse_error(tmp_path: Path) -> None:
    """A document the parser can read but the model rejects -- a frontmatter
    name past the length cap -- raised Pydantic's ValidationError. That is a
    ValueError, so it escaped as itself and a caller handling the documented
    SkillParseError missed it."""
    skill_dir = tmp_path / "toolong"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        "---\nname: " + "a" * 200 + "\ndescription: A skill whose name exceeds the limit.\n---\n\nBody.\n",
        encoding="utf-8",
    )

    with pytest.raises(SkillParseError):
        load_skill(skill_dir / "SKILL.md")
    with pytest.raises(SkillParseError):
        load_skills_from_directory(tmp_path, strict=True)

    # ...and the non-strict path still skips it rather than raising
    assert load_skills_from_directory(tmp_path, strict=False) == []


def test_unterminated_frontmatter_is_rejected(tmp_path: Path) -> None:
    """A document that opens ``---`` and never closes it fell through to the
    "no frontmatter" path, so the whole thing became the body: the name
    silently became the directory's and the description -- the text that gets
    embedded -- became the raw YAML. ``strict=True`` accepted it too."""
    skill_dir = tmp_path / "unterminated"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        "---\nname: my_skill\ndescription: A skill whose frontmatter never closes.\n"
        "\nBody text, with no closing delimiter.\n",
        encoding="utf-8",
    )

    with pytest.raises(SkillParseError, match="never closed"):
        load_skill(skill_dir / "SKILL.md")
    with pytest.raises(SkillParseError, match="never closed"):
        load_skills_from_directory(tmp_path, strict=True)
    assert load_skills_from_directory(tmp_path, strict=False) == []


def test_documents_without_frontmatter_are_still_accepted() -> None:
    """The guard keys off the *opening* delimiter at the start of the file, so
    ordinary Markdown -- including a horizontal rule further down -- is
    untouched."""
    assert parse_skill_markdown("# Title\n\nJust a body.\n")[0] == {}
    assert parse_skill_markdown("Some text\n\n---\n\nMore text\n")[0] == {}
    assert parse_skill_markdown("---\nname: ok\n---\n\nBody\n")[0] == {"name": "ok"}


def test_a_byte_order_mark_does_not_hide_the_frontmatter(tmp_path: Path) -> None:
    """Both frontmatter patterns anchor on the start of the document, so an
    editor's UTF-8 BOM made a perfectly good ``SKILL.md`` look like it had
    none: the name silently became the directory's and the raw YAML became the
    description -- the embedded text -- exactly what the unterminated-block
    guard exists to prevent."""
    from agent_gantry.skills.loader import load_skill

    directory = tmp_path / "bom_skill"
    directory.mkdir()
    (directory / "SKILL.md").write_text(
        "﻿---\nname: bom-skill\n"
        "description: A skill whose file was saved with a byte order mark.\n"
        "---\n\nThe body.\n",
        encoding="utf-8",
    )

    skill = load_skill(directory)
    assert skill.name == "bom-skill"
    assert skill.description == "A skill whose file was saved with a byte order mark."
    assert "---" not in skill.description
