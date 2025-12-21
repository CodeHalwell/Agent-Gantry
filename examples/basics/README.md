# examples/basics

Introductory examples that show how to register tools, sync embeddings, and execute them.

## Files
- `tool_demo.py`: Small "hello world" walkthrough that registers a single tool, retrieves it, and executes it.
- `multi_tool_demo.py`: Registers a diverse set of tools (math, travel, support) to illustrate semantic routing across namespaces.
- `async_demo.py`: Demonstrates native async tool execution and awaits multiple tool calls.
- `tool_creation_patterns.py`: Side-by-side patterns for decorating functions, registering lambdas, and attaching metadata such as namespaces, tags, and capabilities.

## Run commands

```bash
python examples/basics/tool_demo.py
python examples/basics/multi_tool_demo.py
python examples/basics/async_demo.py
```

Each script uses the in-memory embedder/vector store, so no credentials are required. Use these as starting points when wiring Agent-Gantry into your own agents.
