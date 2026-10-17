"""Release validation must stop inconsistent packages before publication."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from scripts.check_release import check_git_tag, release_metadata

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def release_tree(tmp_path: Path) -> Path:
    for name in (
        "pyproject.toml",
        "agent_gantry/__init__.py",
        "package.json",
        "package-lock.json",
        "uv.lock",
        "README.md",
        "CHANGELOG.md",
    ):
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    return tmp_path


def test_release_metadata_is_consistent() -> None:
    version, notes = release_metadata(ROOT)
    assert notes.strip()
    assert release_metadata(ROOT, f"v{version}") == (version, notes)


@pytest.mark.parametrize(
    "name",
    [
        "agent_gantry/__init__.py",
        "package.json",
        "package-lock.json",
        "uv.lock",
        "README.md",
    ],
)
def test_rejects_stale_version(release_tree: Path, name: str) -> None:
    version, _ = release_metadata(release_tree)
    path = release_tree / name
    path.write_text(path.read_text().replace(version, "0.0.1"), encoding="utf-8")
    with pytest.raises(ValueError, match="version|README"):
        release_metadata(release_tree)


def test_rejects_wrong_release_tag() -> None:
    with pytest.raises(ValueError, match="Release tag"):
        release_metadata(ROOT, "v0.0.1")


def test_requires_current_dated_changelog(release_tree: Path) -> None:
    version, _ = release_metadata(release_tree)
    path = release_tree / "CHANGELOG.md"
    path.write_text(path.read_text().replace(f"[{version}]", "[0.0.1]"), encoding="utf-8")
    with pytest.raises(ValueError, match="latest dated"):
        release_metadata(release_tree)


def test_existing_tag_must_match_source_commit(tmp_path: Path) -> None:
    def git(*args: str) -> None:
        subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True)

    git("init")
    git(
        "-c",
        "user.name=Release Test",
        "-c",
        "user.email=test@example.com",
        "commit",
        "--allow-empty",
        "-m",
        "Initial",
    )
    check_git_tag(tmp_path, "0.20.0")  # No tag yet is valid for a manual publish.
    git("tag", "v0.20.0")
    check_git_tag(tmp_path, "0.20.0")
    git(
        "-c",
        "user.name=Release Test",
        "-c",
        "user.email=test@example.com",
        "commit",
        "--allow-empty",
        "-m",
        "Different source",
    )
    with pytest.raises(ValueError, match="different commit"):
        check_git_tag(tmp_path, "0.20.0")
