"""
Load Agent Skills (``SKILL.md``) directories into Gantry's semantic skill store.

The `Agent Skills <https://agentskills.io>`_ format — used by Claude Code,
the Claude Agent SDK and a growing set of agent runtimes — packages a skill
as a directory holding a ``SKILL.md`` whose YAML frontmatter carries ``name``
and ``description`` and whose Markdown body carries the instructions. Those
runtimes load every skill's metadata into the prompt up front; Gantry instead
embeds each skill's metadata once and injects only the skills relevant to the
current prompt — the same top-k retrieval it applies to tools, so a library
of hundreds of skills costs a few hundred tokens per turn rather than all of
them.

.. code-block:: python

    from agent_gantry import AgentGantry

    gantry = AgentGantry()
    await gantry.add_skills_from_directory("~/.claude/skills")
    prompt_section = await gantry.retrieve_skills_as_prompt("convert this docx to pdf")

Only the frontmatter (name, description, tags, category) is embedded; the
Markdown body is stored as the skill's content and injected verbatim on
retrieval, exactly as the runtime would have done, so skill authors change
nothing.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from agent_gantry.schema.skill import Skill, SkillCategory

logger = logging.getLogger(__name__)

#: File name the Agent Skills format mandates (case-insensitive on disk, but
#: the spec spells it this way).
SKILL_FILE_NAME = "SKILL.md"

_FRONTMATTER = re.compile(r"\A---[ \t]*\r?\n(.*?)\r?\n---[ \t]*\r?\n?", re.DOTALL)
_DESCRIPTION_MIN = 10
_DESCRIPTION_MAX = 2000
_CONTENT_MAX = 50000
_SUMMARY_MAX = 500

#: Frontmatter keys consumed into typed fields; anything else lands in
#: ``Skill.metadata`` so runtime-specific keys (``allowed-tools``,
#: ``license``, ``compatibility``, ``metadata``) travel with the skill.
_KNOWN_KEYS = frozenset(
    {"name", "description", "tags", "category", "related_tools", "summary", "namespace"}
)


class SkillParseError(ValueError):
    """A ``SKILL.md`` file could not be turned into a :class:`Skill`."""


def parse_skill_markdown(text: str) -> tuple[dict[str, Any], str]:
    """Split a ``SKILL.md`` document into ``(frontmatter, body)``.

    Args:
        text: The file's contents.

    Returns:
        The parsed YAML frontmatter mapping (empty when absent) and the
        Markdown body with the frontmatter block removed.

    Raises:
        SkillParseError: When the frontmatter is not valid YAML or not a mapping.
    """
    match = _FRONTMATTER.match(text)
    if not match:
        return {}, text
    import yaml  # type: ignore[import-untyped]  # a core dependency, imported lazily

    try:
        data = yaml.safe_load(match.group(1))
    except yaml.YAMLError as exc:
        raise SkillParseError(f"invalid YAML frontmatter: {exc}") from exc
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise SkillParseError("frontmatter must be a YAML mapping")
    return data, text[match.end() :]


def _as_str_list(value: Any) -> list[str]:
    """Coerce a frontmatter list-ish value (list, or comma/space separated string)."""
    if value is None:
        return []
    if isinstance(value, str):
        parts = re.split(r"[,\s]+", value.strip())
        return [p for p in parts if p]
    if isinstance(value, (list, tuple, set)):
        return [str(v).strip() for v in value if str(v).strip()]
    return [str(value)]


def _first_paragraph(body: str) -> str:
    """The first non-heading paragraph of a Markdown body, whitespace-collapsed."""
    for block in re.split(r"\n\s*\n", body):
        lines = [ln.strip() for ln in block.splitlines()]
        lines = [ln for ln in lines if ln and not ln.startswith("#")]
        if lines:
            return " ".join(" ".join(lines).split())
    return ""


def _fit(text: str, limit: int) -> str:
    """Cut ``text`` to ``limit`` characters, at a word boundary where possible."""
    if len(text) <= limit:
        return text
    cut = text[: limit - 1]
    space = cut.rfind(" ")
    if space > limit // 2:
        cut = cut[:space]
    return cut.rstrip() + "…"


def skill_from_markdown(
    text: str,
    *,
    default_name: str,
    namespace: str = "default",
    source_uri: str | None = None,
    category: SkillCategory | str | None = None,
) -> Skill:
    """Build a :class:`Skill` from the contents of a ``SKILL.md`` file.

    Args:
        text: The file's contents (frontmatter + Markdown body).
        default_name: Name to use when the frontmatter has none (the
            directory name, by convention).
        namespace: Namespace for the skill; a frontmatter ``namespace`` wins.
        source_uri: Recorded on the skill for provenance (the file path).
        category: Category to assign when the frontmatter has none; defaults
            to :attr:`SkillCategory.HOW_TO`.

    Returns:
        The skill, ready for :meth:`AgentGantry.add_skills`.

    Raises:
        SkillParseError: When the document has no usable name or content.
    """
    frontmatter, body = parse_skill_markdown(text)
    body = body.strip()

    name = str(frontmatter.get("name") or default_name).strip()
    if not name:
        raise SkillParseError("skill has no name (frontmatter 'name' or directory name)")

    description = " ".join(str(frontmatter.get("description") or "").split())
    if len(description) < _DESCRIPTION_MIN:
        # The description is what gets embedded, so a missing one is filled
        # from the body's opening paragraph rather than left empty.
        fallback = _first_paragraph(body)
        description = description or fallback
        if len(description) < _DESCRIPTION_MIN:
            description = f"{description or name} (skill '{name}')"
    description = _fit(description, _DESCRIPTION_MAX)

    content = body or description
    if len(content) > _CONTENT_MAX:
        logger.warning(
            "Skill %r body is %d characters; truncated to %d for storage",
            name,
            len(content),
            _CONTENT_MAX,
        )
        content = _fit(content, _CONTENT_MAX)

    raw_category = frontmatter.get("category") or category or SkillCategory.HOW_TO
    try:
        resolved_category = (
            raw_category
            if isinstance(raw_category, SkillCategory)
            else SkillCategory(str(raw_category).strip().lower())
        )
    except ValueError:
        logger.warning("Skill %r has unknown category %r; using how_to", name, raw_category)
        resolved_category = SkillCategory.HOW_TO

    related_tools = _as_str_list(frontmatter.get("related_tools"))
    if not related_tools:
        # Claude Code's ``allowed-tools`` names the tools a skill drives —
        # the closest thing the format has to Gantry's related_tools.
        related_tools = _as_str_list(frontmatter.get("allowed-tools"))

    summary = frontmatter.get("summary")
    summary_text = _fit(" ".join(str(summary).split()), _SUMMARY_MAX) if summary else None

    metadata: dict[str, Any] = {
        key: value for key, value in frontmatter.items() if key not in _KNOWN_KEYS
    }
    metadata.setdefault("format", "agent-skills")

    try:
        return Skill(
            name=name,
            namespace=str(frontmatter.get("namespace") or namespace),
            description=description,
            content=content,
            summary=summary_text,
            category=resolved_category,
            tags=_as_str_list(frontmatter.get("tags")),
            related_tools=related_tools,
            source="skill_md",
            source_uri=source_uri,
            metadata=metadata,
        )
    except ValidationError as exc:
        # A document this function could read but the model rejects — a
        # frontmatter name past the length cap, say. Pydantic's error is a
        # ValueError, so it propagated as itself, and a caller handling the
        # documented SkillParseError missed it.
        raise SkillParseError(f"skill {name!r} is not valid: {exc}") from exc


def load_skill(path: str | Path, *, namespace: str = "default") -> Skill:
    """Load one skill from a ``SKILL.md`` file or the directory containing it.

    Args:
        path: The ``SKILL.md`` file, or its directory.
        namespace: Namespace for the skill (frontmatter ``namespace`` wins).

    Raises:
        FileNotFoundError: When no ``SKILL.md`` exists at ``path``.
        SkillParseError: When the file cannot be parsed into a skill.
    """
    file = Path(path).expanduser()
    if file.is_dir():
        file = file / SKILL_FILE_NAME
    if not file.is_file():
        raise FileNotFoundError(f"No {SKILL_FILE_NAME} at {file}")
    text = file.read_text(encoding="utf-8")
    try:
        return skill_from_markdown(
            text,
            default_name=file.parent.name,
            namespace=namespace,
            source_uri=file.resolve().as_uri(),
        )
    except SkillParseError as exc:
        raise SkillParseError(f"{file}: {exc}") from exc


def iter_skill_files(root: str | Path, *, recursive: bool = True) -> Iterator[Path]:
    """Yield every ``SKILL.md`` under ``root`` (``root`` itself may be a skill dir).

    Sorted for deterministic ordering. Hidden directories (``.git``,
    ``.venv``) are skipped.
    """
    base = Path(root).expanduser()
    if base.is_file():
        yield base
        return
    if (base / SKILL_FILE_NAME).is_file():
        yield base / SKILL_FILE_NAME
        if not recursive:
            return
    pattern = f"**/{SKILL_FILE_NAME}" if recursive else f"*/{SKILL_FILE_NAME}"
    for file in sorted(base.glob(pattern)):
        if file == base / SKILL_FILE_NAME:
            continue
        if any(part.startswith(".") for part in file.relative_to(base).parts[:-1]):
            continue
        yield file


def load_skills_from_directory(
    root: str | Path,
    *,
    namespace: str = "default",
    recursive: bool = True,
    strict: bool = False,
) -> list[Skill]:
    """Load every ``SKILL.md`` under ``root`` as a :class:`Skill`.

    Args:
        root: A skills directory (e.g. ``~/.claude/skills``), or a single
            skill's directory.
        namespace: Namespace for the loaded skills.
        recursive: Search nested directories (default) or only ``root``'s
            immediate children.
        strict: Raise on the first unparsable file instead of logging and
            skipping it.

    Returns:
        The loaded skills, in path order.

    Raises:
        FileNotFoundError: When ``root`` does not exist.
        SkillParseError: In ``strict`` mode, for the first bad file.
    """
    base = Path(root).expanduser()
    if not base.exists():
        raise FileNotFoundError(f"Skills directory not found: {base}")
    skills: list[Skill] = []
    for file in iter_skill_files(base, recursive=recursive):
        try:
            skills.append(load_skill(file, namespace=namespace))
        except (SkillParseError, ValueError) as exc:
            if strict:
                raise
            logger.warning("Skipping skill %s: %s", file, exc)
    return skills


__all__ = [
    "SKILL_FILE_NAME",
    "SkillParseError",
    "iter_skill_files",
    "load_skill",
    "load_skills_from_directory",
    "parse_skill_markdown",
    "skill_from_markdown",
]
