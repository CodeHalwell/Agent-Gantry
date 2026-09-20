"""
Source linter — flags Agent-Gantry usage mistakes in Python files.

:mod:`agent_gantry.utils.registry_linter` inspects a *loaded* registry. Two
mistakes cannot be seen from there because they live in the calling code
rather than in any tool definition, and both came up often enough in this
repository's own examples to be worth a rule (#434):

- **An inverted ``score_threshold``.** The field is an absolute cosine cutoff
  that defaults to ``0.0`` on ``retrieve_tools()`` and every framework
  adapter, so a non-zero value only ever *tightens* the filter. A comment
  beside it claiming it lowers, loosens or relaxes anything — "0.1 for
  SimpleEmbedder compatibility" was the recurring form — teaches the knob
  backwards, and a bare non-zero literal with no justification is usually the
  same mistake without the comment.
- **A gantry that is never closed.** A script that constructs an
  :class:`~agent_gantry.core.gantry.AgentGantry` under ``if __name__ ==
  "__main__"`` and never calls ``close()`` gets away with a warning at
  interpreter shutdown, which is why it goes unnoticed — and is exactly the
  wrong thing to copy into a long-lived service.

The CLI runs this as ``agent-gantry lint --source PATH``.
"""

from __future__ import annotations

import ast
import io
import re
import tokenize
from dataclasses import dataclass, field
from pathlib import Path

#: Comment text that claims a non-zero threshold makes retrieval *more*
#: permissive. Case-insensitive; matched against the comment on the same line
#: as the argument and the comment on the line above it.
_INVERTED_COMMENT = re.compile(r"compat|lower|loosen|relax|permissive|lenient", re.IGNORECASE)

#: Constructors whose result is a gantry that needs closing.
_GANTRY_CONSTRUCTORS = {"AgentGantry", "create_default_gantry"}
_GANTRY_CLASSMETHODS = {"quick_start", "from_config"}


@dataclass
class ThresholdFinding:
    """A non-zero ``score_threshold`` literal with no, or an inverted, justification."""

    path: str
    line: int
    value: float
    comment: str | None

    @property
    def inverted(self) -> bool:
        return self.comment is not None


@dataclass
class UnclosedGantryFinding:
    """A ``__main__`` script constructs a gantry and never calls ``close()``."""

    path: str
    line: int


@dataclass
class SourceAnalysis:
    """Aggregated findings across the files scanned."""

    thresholds: list[ThresholdFinding] = field(default_factory=list)
    unclosed: list[UnclosedGantryFinding] = field(default_factory=list)
    files_scanned: int = 0

    @property
    def empty(self) -> bool:
        return not self.thresholds and not self.unclosed

    def format_text(self) -> str:
        """Human-readable rendering for the CLI."""
        if self.empty:
            return f"No issues found in {self.files_scanned} file(s)."
        lines: list[str] = []
        if self.thresholds:
            lines.append(
                "Non-zero score_threshold overrides (an absolute cosine cutoff; "
                "non-zero only ever tightens the filter):"
            )
            for f in self.thresholds:
                why = (
                    f"comment claims it relaxes something: {f.comment!r}"
                    if f.inverted
                    else "no comment justifies it"
                )
                lines.append(f"  - {f.path}:{f.line}: score_threshold={f.value:g} — {why}")
        if self.unclosed:
            if lines:
                lines.append("")
            lines.append("Gantries constructed in a __main__ script and never closed:")
            for f in self.unclosed:
                lines.append(f"  - {f.path}:{f.line}")
        return "\n".join(lines)


def _comments_by_line(source: str) -> dict[int, str]:
    comments: dict[int, str] = {}
    try:
        for tok in tokenize.generate_tokens(io.StringIO(source).readline):
            if tok.type == tokenize.COMMENT:
                comments[tok.start[0]] = tok.string.lstrip("#").strip()
    except (tokenize.TokenError, SyntaxError):
        pass
    return comments


def _numeric_literal(node: ast.expr) -> float | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        inner = _numeric_literal(node.operand)
        return None if inner is None else -inner
    return None


def _threshold_findings(
    tree: ast.AST, comments: dict[int, str], path: str
) -> list[ThresholdFinding]:
    findings: list[ThresholdFinding] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        for kw in node.keywords:
            if kw.arg != "score_threshold":
                continue
            value = _numeric_literal(kw.value)
            if value is None or value == 0.0:
                continue
            line = kw.value.lineno
            nearby = [comments.get(line), comments.get(line - 1)]
            # ``ToolQuery(score_threshold=...)`` on its own line puts the
            # comment above the call rather than above the argument.
            if node.lineno != line:
                nearby.append(comments.get(node.lineno - 1))
            nearby = [c for c in nearby if c]
            inverted = next((c for c in nearby if _INVERTED_COMMENT.search(c)), None)
            if inverted is not None:
                findings.append(ThresholdFinding(path, line, value, inverted))
            elif not nearby:
                findings.append(ThresholdFinding(path, line, value, None))
    return findings


def _is_main_guard(node: ast.stmt) -> bool:
    if not isinstance(node, ast.If) or not isinstance(node.test, ast.Compare):
        return False
    left = node.test.left
    return (
        isinstance(left, ast.Name)
        and left.id == "__name__"
        and any(
            isinstance(c, ast.Constant) and c.value == "__main__" for c in node.test.comparators
        )
    )


def _constructs_gantry(node: ast.Call) -> bool:
    func = node.func
    if isinstance(func, ast.Name):
        return func.id in _GANTRY_CONSTRUCTORS
    if isinstance(func, ast.Attribute):
        if func.attr in _GANTRY_CONSTRUCTORS:
            return True
        return func.attr in _GANTRY_CLASSMETHODS and (
            isinstance(func.value, ast.Name) and func.value.id == "AgentGantry"
        )
    return False


def _unclosed_findings(tree: ast.Module, path: str) -> list[UnclosedGantryFinding]:
    if not any(_is_main_guard(stmt) for stmt in tree.body):
        return []
    closes = False
    constructions: list[int] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if _constructs_gantry(node):
                constructions.append(node.lineno)
            elif isinstance(node.func, ast.Attribute) and node.func.attr in {"close", "aclose"}:
                closes = True
        elif isinstance(node, (ast.AsyncWith, ast.With)):
            # ``async with AgentGantry() as gantry:`` closes on exit.
            for item in node.items:
                if isinstance(item.context_expr, ast.Call) and _constructs_gantry(
                    item.context_expr
                ):
                    closes = True
    if closes or not constructions:
        return []
    return [UnclosedGantryFinding(path, min(constructions))]


def analyze_source(text: str, path: str = "<string>") -> SourceAnalysis:
    """Analyse one file's source text.

    Args:
        text: Python source.
        path: Name reported in findings.

    Returns:
        A :class:`SourceAnalysis` for this one file (``files_scanned`` is 1).
    """
    analysis = SourceAnalysis(files_scanned=1)
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return analysis
    comments = _comments_by_line(text)
    analysis.thresholds = _threshold_findings(tree, comments, path)
    analysis.unclosed = _unclosed_findings(tree, path)
    return analysis


def _iter_python_files(paths: list[str | Path]) -> list[Path]:
    files: list[Path] = []
    for raw in paths:
        p = Path(raw)
        if p.is_dir():
            files.extend(
                f for f in sorted(p.rglob("*.py")) if "__pycache__" not in f.parts
            )
        elif p.suffix == ".py":
            files.append(p)
    return files


def analyze_paths(paths: list[str | Path]) -> SourceAnalysis:
    """Analyse every ``.py`` file under ``paths`` (files or directories).

    Args:
        paths: Files or directories to scan; directories are walked.

    Returns:
        A :class:`SourceAnalysis` aggregating every file's findings.
    """
    total = SourceAnalysis()
    for file in _iter_python_files(paths):
        try:
            text = file.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        one = analyze_source(text, str(file))
        total.files_scanned += 1
        total.thresholds.extend(one.thresholds)
        total.unclosed.extend(one.unclosed)
    return total


__all__ = [
    "SourceAnalysis",
    "ThresholdFinding",
    "UnclosedGantryFinding",
    "analyze_paths",
    "analyze_source",
]
