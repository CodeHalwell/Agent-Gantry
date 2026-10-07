"""The docs site's npm manifest must stay reproducible.

``package.json`` used to ask for ``"latest"`` for React, TypeScript and their types.
CI runs ``npm ci``, which honours the lock, but the documented setup command is
``npm install``, which resolves whatever the registry tags ``latest`` today and
rewrites the lock: a React or TypeScript major could arrive in an unrelated pull
request's lock diff and break ``astro check`` for reasons the change never touched.
"""

from __future__ import annotations

import json
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SECTIONS = ("dependencies", "devDependencies")
_FLOATING = {"", "*", "latest", "next", "x"}


def _load(name: str) -> dict:
    return json.loads((_ROOT / name).read_text(encoding="utf-8"))


def test_no_docs_dependency_floats_to_the_registrys_newest_release() -> None:
    manifest = _load("package.json")
    floating = {
        name: spec
        for section in _SECTIONS
        for name, spec in manifest.get(section, {}).items()
        if spec.strip() in _FLOATING
    }
    assert not floating, (
        f"package.json has unpinned dependencies {floating}; use a caret range taken from "
        "package-lock.json so `npm install` cannot jump a major version"
    )


def test_the_lock_records_the_manifests_specs() -> None:
    # What `npm ci` checks. Editing one file and not the other fails CI; failing here is quicker.
    manifest = _load("package.json")
    root = _load("package-lock.json")["packages"][""]
    for section in _SECTIONS:
        assert root.get(section, {}) == manifest.get(section, {}), (
            f"package-lock.json's root {section} differ from package.json's; "
            "run `npm install --package-lock-only`"
        )
