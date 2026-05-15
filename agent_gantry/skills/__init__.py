"""
Bundled Claude Skill describing how to use Agent-Gantry.

The skill itself lives at ``agent_gantry/skills/agent-gantry/SKILL.md`` and
is shipped inside the wheel so it is discoverable without extra downloads.
Two helpers are exposed for callers:

- :func:`skill_path` returns the absolute path to the bundled skill so it
  can be passed to ``SkillsProvider(skill_paths=[...])`` (Microsoft Agent
  Framework) or to Claude Code's ``--skill`` flag.
- :func:`install_to` copies the skill into a target directory — useful
  for projects that maintain a ``skills/`` folder in their own repo and
  want to vendor the Agent-Gantry skill alongside their own skills.

Both helpers fall back gracefully when the package was installed without
the skill files (e.g. someone vendored the source tree by hand and
trimmed it).
"""

from __future__ import annotations

import shutil
from pathlib import Path

_SKILL_DIR_NAME = "agent-gantry"


def skill_path() -> Path:
    """Return the absolute path to the bundled Agent-Gantry skill.

    Use to wire the skill into an agent without copying files:

    .. code-block:: python

        from agent_framework import SkillsProvider
        from agent_gantry.skills import skill_path

        provider = SkillsProvider(skill_paths=[str(skill_path().parent)])

    Returns:
        ``Path`` pointing at the ``agent-gantry/`` skill directory inside
        the installed package. The returned path is guaranteed to exist
        when the package was installed from the wheel — :exc:`FileNotFoundError`
        is raised otherwise so the caller can fall back to a downloaded copy.
    """
    path = Path(__file__).parent / _SKILL_DIR_NAME
    if not path.is_dir():
        raise FileNotFoundError(
            f"Agent-Gantry skill directory not found at {path!s}. "
            "This usually means the package was installed from a source "
            "tree that doesn't include skills/. Reinstall via "
            "'pip install agent-gantry' to get the bundled skill."
        )
    return path


def install_to(target: str | Path, *, overwrite: bool = False) -> Path:
    """Copy the bundled skill into ``target/agent-gantry/``.

    Args:
        target: Destination directory. Created if missing.
        overwrite: When True, replace an existing ``agent-gantry``
            directory inside ``target``. Defaults to False.

    Returns:
        Absolute path to the freshly installed skill directory.

    Raises:
        FileExistsError: When the destination already exists and
            ``overwrite=False``.
        FileNotFoundError: When the bundled skill is missing (see
            :func:`skill_path`).
    """
    src = skill_path()
    dst_root = Path(target).expanduser().resolve()
    dst_root.mkdir(parents=True, exist_ok=True)
    dst = dst_root / _SKILL_DIR_NAME
    if dst.exists():
        if not overwrite:
            raise FileExistsError(
                f"{dst!s} already exists; pass overwrite=True to replace it."
            )
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    return dst


__all__ = ["install_to", "skill_path"]
