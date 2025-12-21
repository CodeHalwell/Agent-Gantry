---
name: "💾 State Management"
about: Feature requests for session memory, context persistence, and stateful tools
title: "[State] "
labels: ["enhancement", "state", "memory", "session"]
assignees: []
---

## Feature Area
<!-- Check the area(s) this feature request relates to -->
- [ ] Session Memory / Context Store
- [ ] Cross-Tool State Sharing
- [ ] Persistent State Storage
- [ ] State Scoping (user, session, global)
- [ ] Other State Management Feature

## Problem Statement
<!-- Describe the current limitation in state management this feature would address -->

**Current Behavior:**
<!-- How is state currently handled? -->

**Desired Behavior:**
<!-- What state management capabilities are needed? -->

## Proposed Solution

### Overview
<!-- High-level description of the state management solution -->

### State Model
<!-- Describe the state model, scope, and lifecycle -->

### Technical Details
<!-- Technical implementation details, if applicable -->

### Example Usage
<!-- Provide a code example showing how state would be accessed -->

```python
# Example usage
from agent_gantry import AgentGantry

gantry = AgentGantry()

@gantry.register
async def read_file(filename: str, ctx) -> str:
    """Read a file and remember it."""
    # Store in session memory
    content = read_from_disk(filename)
    ctx.memory["last_file"] = filename
    ctx.memory["file_content"] = content
    return content

@gantry.register
async def summarize_last_file(ctx) -> str:
    """Summarize the last file that was read."""
    # Access from session memory
    filename = ctx.memory.get("last_file")
    content = ctx.memory.get("file_content")
    return f"Summary of {filename}: {generate_summary(content)}"
```

## Use Case
<!-- Describe a real-world scenario where state management would be valuable -->

**Example Scenario:**


**Expected Benefit:**


## Context: Current State Handling

Currently, Agent-Gantry is largely stateless:
- ✅ Tools are pure functions or async callables
- ✅ Context is passed via LLM conversation history
- ⚠️ No built-in mechanism for tools to share state
- ⚠️ Tools must pass state via arguments every time

This feature request aims to add **session-scoped state management** for tools.

### Session Memory
- **Goal**: Tools can persist state within a session without external DBs
- **Example**: "What file was I reading?" without passing filename every time
- **Scope**: Per-session, per-user, or per-conversation
- **Storage**: In-memory by default, pluggable backends (Redis, etc.)

### Use Cases
1. **Multi-Step Workflows**: Tool A stores intermediate results for Tool B
2. **File Operations**: Remember "current working directory" or "active file"
3. **Conversation Context**: Track entities, IDs, or references mentioned in conversation
4. **Caching**: Store expensive computation results within a session
5. **User Preferences**: Remember user choices within a session

### State API Design Considerations
```python
# Option 1: Context parameter
@gantry.register
async def my_tool(arg: str, ctx: ExecutionContext) -> str:
    ctx.memory["key"] = "value"
    return ctx.memory.get("other_key", default="")

# Option 2: Session object
@gantry.register(stateful=True)
async def my_tool(arg: str, session: Session) -> str:
    await session.set("key", "value")
    return await session.get("other_key")

# Option 3: Decorator-based
@gantry.register
@gantry.with_memory(scope="session")
async def my_tool(arg: str, memory: dict) -> str:
    memory["key"] = "value"
    return memory.get("other_key", "")
```

## State Scoping
<!-- Define different scope levels for state -->
- [ ] **Session-scoped**: State persists for a single agent session
- [ ] **User-scoped**: State persists across sessions for a user
- [ ] **Global-scoped**: State shared across all users (with proper isolation)
- [ ] **Tool-scoped**: Each tool instance has its own state

## Persistence Strategy
<!-- How should state be persisted? -->
- [ ] In-memory only (lost on restart)
- [ ] Redis/Memcached
- [ ] Database (PostgreSQL, SQLite)
- [ ] File-based
- [ ] Pluggable backend

## Thread Safety & Concurrency
<!-- How should concurrent access be handled? -->

## TTL & Expiration
<!-- Should state expire automatically? -->

## Additional Context
<!-- Add any other context, architecture diagrams, or references -->

## Related Issues
<!-- Link to related issues or PRs -->

---
**Note**: This feature aligns with the strategic goal of supporting stateful tool patterns and multi-step agent workflows without requiring external state management.
