#!/usr/bin/env bash
# PostToolUse hook: lint and format a Python file Claude just edited.
set -u
file="$(jq -r '.tool_input.file_path // empty' 2>/dev/null)"
[ -n "$file" ] || exit 0
case "$file" in *.py) ;; *) exit 0 ;; esac
[ -f "$file" ] || exit 0
cd "$(git -C "$(dirname "$file")" rev-parse --show-toplevel 2>/dev/null || pwd)" || exit 0
out="$(uvx ruff check --fix "$file" 2>&1; uvx ruff format "$file" 2>&1)"
if uvx ruff check "$file" >/dev/null 2>&1; then exit 0; fi
# Exit 2 feeds stderr back to Claude as something to fix.
printf 'ruff still reports issues in %s:\n%s\n' "$file" "$(uvx ruff check "$file" 2>&1 | tail -20)" >&2
exit 2
