# AGENTS.md — Agent-Gantry

Instructions for any coding agent (Claude Code, Codex, Antigravity, Shelley)
working in this repository. Humans: see CONTRIBUTING.md for the long form.

## Commands (run exactly these)

```bash
uv sync --extra dev --extra nomic --extra lancedb --extra agent-frameworks --extra example-tools
uv run ruff check agent_gantry/            # lint (CI: lint job)
uv run ruff format --check agent_gantry/   # formatting
uvx ty check agent_gantry/ --warn all      # types (advisory in CI, --exit-zero)
uv run pytest --ignore=tests/test_phase5_mcp.py --ignore=tests/test_phase6_a2a.py -q
```

Run a single test file with `uv run pytest tests/test_tool.py -q`.
Pre-commit runs ruff on commit and the test suite on push: `uvx pre-commit install --hook-type pre-commit --hook-type pre-push`.

## Conventions

- Python 3.10+ compatible code; the CI matrix is 3.10 to 3.13. No 3.11-only syntax.
- Ruff config lives in `pyproject.toml` (line length 100, rules E F I N W UP). Do not add per-file ignores to silence real findings.
- Type hints on every public function. Prefer `from __future__ import annotations`.
- Async first: the core is asyncio; do not add blocking I/O in async paths (use `asyncio.to_thread`).
- Logging via the module logger, never `print`, in library code.
- Errors: raise the specific exception from `agent_gantry.exceptions`; do not swallow exceptions.
- British English in prose, comments and docs.
- Keep CHANGELOG.md updated under the Unreleased heading for user-visible changes.

## Definition of done

- All commands above pass locally.
- New behaviour has tests in `tests/`; bug fixes add a regression test.
- No secrets, tokens or real endpoints in code, tests or fixtures.
- Docs updated when a public API changes (`docs/` and README examples).

## Boundaries

- Never push to `main`. Work on a branch, open a pull request, let CI and review run.
- Never merge your own pull request.
- Never use production credentials. Tests must run with no network and no secrets.
- Do not edit `.github/workflows/`, `pyproject.toml` dependency pins or `uv.lock` unless the task is specifically about them.
- Do not delete or skip failing tests to make the suite green.
