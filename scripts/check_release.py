"""Validate release metadata without importing optional dependencies.

Run ``python scripts/check_release.py --tag v0.20.0 --notes-file release-notes.md``.
Omit ``--tag`` for a branch build; ``--check-git-tag`` also checks an existing
version tag points to the checked-out commit before anything can be published.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
from datetime import date
from pathlib import Path


def release_metadata(root: Path, tag: str | None = None) -> tuple[str, str]:
    """Return the consistent version and its dated changelog entry, or fail."""
    project = (root / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'(?ms)^\[project\]\s*\n.*?^version = "([^"]+)"', project)
    if match is None:
        raise ValueError("Missing [project].version in pyproject.toml")
    version = match.group(1)
    if not re.fullmatch(r"\d+\.\d+\.\d+", version):
        raise ValueError(f"Expected a stable release version, got {version!r}")

    module = ast.parse((root / "agent_gantry/__init__.py").read_text(encoding="utf-8"))
    runtime_version = next(
        (
            ast.literal_eval(node.value)
            for node in module.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "__version__"
                for target in node.targets
            )
        ),
        None,
    )
    package = json.loads((root / "package.json").read_text(encoding="utf-8"))
    npm_lock = json.loads((root / "package-lock.json").read_text(encoding="utf-8"))
    uv_lock = (root / "uv.lock").read_text(encoding="utf-8")
    editable = re.search(
        r'(?m)^name = "agent-gantry"\nversion = "([^"]+)"\nsource = \{ editable = "\." \}',
        uv_lock,
    )
    versions = {
        "agent_gantry.__version__": runtime_version,
        "package.json": package.get("version"),
        "package-lock.json": npm_lock.get("version"),
        "package-lock.json root package": npm_lock.get("packages", {}).get("", {}).get("version"),
        "uv.lock editable project": editable.group(1) if editable else None,
    }
    for source, actual in versions.items():
        if actual != version:
            raise ValueError(f"{source} version {actual!r} differs from pyproject.toml {version!r}")
    if tag is not None and tag != f"v{version}":
        raise ValueError(f"Release tag {tag!r} must match package version v{version}")
    if f"**v{version}**" not in (root / "README.md").read_text(encoding="utf-8"):
        raise ValueError(f"README.md must describe v{version}")

    changelog = (root / "CHANGELOG.md").read_text(encoding="utf-8")
    entries = re.findall(
        r"(?ms)^## \[([^\]]+)\] - (\d{4}-\d{2}-\d{2})\s*\n(.*?)(?=^## |\Z)",
        changelog,
    )
    if not entries or entries[0][0] != version:
        raise ValueError(f"The latest dated CHANGELOG.md entry must be [{version}]")
    _, released, notes = entries[0]
    date.fromisoformat(released)
    if not notes.strip():
        raise ValueError(f"CHANGELOG.md entry [{version}] must contain release notes")
    return version, notes.strip() + "\n"


def check_git_tag(root: Path, version: str) -> None:
    """An existing release tag must identify the artifacts' source commit."""
    tagged = subprocess.run(
        ["git", "rev-parse", "--verify", f"refs/tags/v{version}^{{commit}}"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    if tagged.returncode:
        return  # A manual publish creates the tag after uploading to PyPI.
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    if tagged.stdout.strip() != head:
        raise ValueError(
            f"Existing tag v{version} points to a different commit; refusing to publish"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", help="GitHub Release tag; must equal v<package version>")
    parser.add_argument("--check-git-tag", action="store_true")
    parser.add_argument("--notes-file", type=Path, help="Write the dated changelog entry here")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    try:
        version, notes = release_metadata(root, args.tag)
        if args.check_git_tag:
            check_git_tag(root, version)
    except (ValueError, OSError) as exc:
        parser.exit(1, f"Release metadata error: {exc}\n")
    if args.notes_file:
        args.notes_file.write_text(notes, encoding="utf-8")
    print(f"Release metadata OK: v{version}")


if __name__ == "__main__":
    main()
