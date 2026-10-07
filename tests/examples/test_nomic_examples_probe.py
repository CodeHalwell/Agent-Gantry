"""Examples that build a ``NomicEmbedder`` must probe for sentence-transformers first.

``NomicEmbedder`` imports sentence-transformers lazily, on first use, so constructing
it never raises ``ImportError``. A ``try/except ImportError`` around the constructor
alone therefore never fires, and the example dies at ``sync()`` with a traceback
instead of falling back. Five examples did exactly that, against the examples
README's promise that none of them crashes without the optional extras.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_EXAMPLES = Path(__file__).resolve().parents[2] / "examples"

#: Builds a Nomic embedder only when the user asks for one by name (``--embedder
#: nomic``; the default is the hash embedder), and the embedder's own ImportError
#: then names the package to install.
_ON_REQUEST_ONLY = {"agent_framework_tui_demo.py"}


def _scan(path: Path) -> tuple[list[int], list[int]]:
    """Lines that call ``NomicEmbedder(...)`` and lines that import sentence_transformers."""
    builds: list[int] = []
    probes: list[int] = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Call):
            func = node.func
            name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
            if name == "NomicEmbedder":
                builds.append(node.lineno)
        elif isinstance(node, ast.Import):
            if any(alias.name.split(".")[0] == "sentence_transformers" for alias in node.names):
                probes.append(node.lineno)
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".")[0] == "sentence_transformers":
                probes.append(node.lineno)
    return builds, probes


_BUILDERS = sorted(
    path
    for path in _EXAMPLES.rglob("*.py")
    if path.name not in _ON_REQUEST_ONLY and _scan(path)[0]
)


def test_the_scan_finds_the_examples_it_is_meant_to_guard() -> None:
    # A glob or parser that silently matched nothing would make the test below pass vacuously.
    names = {path.name for path in _BUILDERS}
    assert {"llm_demo.py", "tools.py", "tools_persistent.py", "nomic_tool_demo.py"} <= names


@pytest.mark.parametrize("path", _BUILDERS, ids=lambda p: str(p.relative_to(_EXAMPLES)))
def test_nomic_examples_probe_sentence_transformers_before_building_it(path: Path) -> None:
    builds, probes = _scan(path)
    assert probes and min(probes) < min(builds), (
        f"{path.relative_to(_EXAMPLES)} builds a NomicEmbedder (line {min(builds)}) without first "
        "importing sentence_transformers; the constructor never raises, so the script would "
        "crash at sync() instead of falling back"
    )
