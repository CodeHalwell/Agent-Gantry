"""The package version must match everywhere it is written down.

``agent_gantry.__version__``, ``pyproject.toml``'s ``[project].version``, the
docs site's ``package.json`` (which the site reads for its "vX docs" label) and
the version in ``README.md``'s opening paragraph are maintained by hand. The
``lint`` job in ``ci.yml`` checks the first two before anything else runs; this
test also covers the other two, which a release had to remember separately and
which nothing checked, so the site and README could advertise a stale version.
"""

from __future__ import annotations

import re
from pathlib import Path

import agent_gantry

_ROOT = Path(__file__).resolve().parent.parent


def _pyproject_version() -> str:
    text = (_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    # Anchor to the [project] table so we read the package version, not some
    # other `version = "..."` line (e.g. inside a tool config or dependency).
    project = re.search(r"(?ms)^\[project\]\s*\n(.*?)(?=^\[)", text)
    assert project, "could not find [project] table in pyproject.toml"
    match = re.search(r'(?m)^version = "([^"]+)"', project.group(1))
    assert match, "could not find version in [project] table"
    return match.group(1)


def test_version_matches_pyproject() -> None:
    assert agent_gantry.__version__ == _pyproject_version(), (
        f"agent_gantry.__version__ ({agent_gantry.__version__}) != "
        f"pyproject.toml version ({_pyproject_version()})"
    )


def test_docs_site_version_matches_the_package() -> None:
    import json

    site = json.loads((_ROOT / "package.json").read_text(encoding="utf-8"))["version"]
    assert site == agent_gantry.__version__, (
        f"package.json version ({site}) != agent_gantry.__version__ ({agent_gantry.__version__}); "
        "the documentation site displays the former"
    )


def test_readme_names_the_current_version() -> None:
    readme = (_ROOT / "README.md").read_text(encoding="utf-8")
    assert f"**v{agent_gantry.__version__}**" in readme, (
        f"README.md's opening paragraph does not name v{agent_gantry.__version__}"
    )
