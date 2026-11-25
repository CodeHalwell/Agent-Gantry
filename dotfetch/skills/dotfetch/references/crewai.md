# CrewAI

> Part of the `dotfetch` skill. Core concepts, debugging and common
> pitfalls are in [`SKILL.md`](../SKILL.md); read those first.

Install with `uv add "dotfetch[crewai]"` (pins `crewai>=1.15.22`, Python 3.10–3.13; tested on 1.15.22 and the latest). Not with `[openai-agents]` (openai 3 vs CrewAI's `openai<3`).

```python
from crewai import Agent, Crew, Task
from dotfetch.crewai import ApprovalRequest, CrewAIAdapter, current_crew_context

adapter = CrewAIAdapter(fetcher, approve=lambda req: ask_a_human(req))   # or approve="console"
agent = Agent(role="support", goal="...", backstory="...", tools=await adapter.select(task_text))
output = await Crew(agents=[agent], tasks=[Task(description=task_text, expected_output="...", agent=agent)]).kickoff_async()
output.tasks_output[0].tool_failures                  # dotfetch failures recorded as CrewAI ToolFailure
```

- **Selection**: per call (`select` per agent, or `agent_builder(...).build(query)` per task); CrewAI has no hook to swap tools between model turns.
- **Errors**: `_run` returns CrewAI's `ToolFailure` (`code` = error type, `retryable` for argument errors); `tool_failure_policy` on the agent/task decides whether to abort. `error_mode="raise"` restores `ToolExecutionError`.
- **Approval**: a `before_tool_call` hook (registered once per process) asks `approve(request) -> bool` for gated tools (`fetcher.requires_confirmation`); refused calls are blocked by CrewAI; no approver means "needs approval, not run".
- **Run context**: `current_crew_context()` returns CrewAI's `ToolCallHookContext` (`.agent`, `.task`, `.crew`).
