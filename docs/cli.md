# CLI usage

Agent-Gantry ships with a lightweight CLI for quick inspection and demo workflows. The entrypoint
is installed as `agent-gantry`.

## Commands

```
agent-gantry list [--namespace default]
agent-gantry search "<query>" [--limit 5]
```

### list

Lists registered tools in the current session (the CLI loads a small set of demo tools by
default).

```
agent-gantry list
# default.send_email: Send an email with a subject and body.
# default.generate_report: Generate a report for the given date range.
# default.process_refund: Process a refund for a given order.
```

### search

Performs semantic retrieval over the demo tools and prints the top matches with scores.

```
agent-gantry search "refund an order" --limit 3
# process_refund (0.74) - Process a refund for a given order.
# send_email (0.42) - Send an email with a subject and body.
```

## Notes

- The CLI uses in-memory defaults and a simple embedder; for production use, prefer the Python API
  with a configured `AgentGantryConfig`.
- Customize or extend CLI behaviour by importing `agent_gantry.cli.main` in your own entrypoint.
