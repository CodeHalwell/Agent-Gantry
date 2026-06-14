"""The package version must match across its two sources.

``agent_gantry.__version__`` and ``pyproject.toml``'s ``[project].version`` are
maintained by hand. The automated release workflow gates on this match (and
refuses to publish on a mismatch), so this test catches drift before CI/release.
"""

from __future__ import annotations

import re
from pathlib import Path

import agent_gantry

_ROOT = Path(__file__).resolve().parent.parent


def _pyproject_version() -> str:
    text = (_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'(?m)^version = "([^"]+)"', text)
    assert match, "could not find version in pyproject.toml"
    return match.group(1)


def test_version_matches_pyproject() -> None:
    assert agent_gantry.__version__ == _pyproject_version(), (
        f"agent_gantry.__version__ ({agent_gantry.__version__}) != "
        f"pyproject.toml version ({_pyproject_version()})"
    )
