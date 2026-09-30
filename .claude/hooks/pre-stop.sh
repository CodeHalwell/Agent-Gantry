#!/usr/bin/env bash
# Stop hook: Claude may not hand back while lint or the test suite fails.
set -u
input="$(cat)"
# Avoid an infinite loop: if this stop was already triggered by this hook, allow it.
if printf '%s' "$input" | jq -e '.stop_hook_active == true' >/dev/null 2>&1; then exit 0; fi
cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)" || exit 0
# Only gate when Python files changed in this working tree.
if [ -z "$(git status --porcelain -- '*.py' 2>/dev/null)" ]; then exit 0; fi
fail=""
lint="$(uv run ruff check agent_gantry/ 2>&1)" || fail="$fail\n--- ruff ---\n$(printf '%s' "$lint" | tail -30)"
tests="$(uv run pytest --ignore=tests/test_phase5_mcp.py --ignore=tests/test_phase6_a2a.py -q -x 2>&1)" || fail="$fail\n--- pytest ---\n$(printf '%s' "$tests" | tail -40)"
if [ -n "$fail" ]; then
  jq -n --arg r "Checks failed. Fix them before finishing (see AGENTS.md).$(printf "$fail")" \
    '{"decision":"block","reason":$r}'
fi
exit 0
