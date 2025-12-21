# Agent Framework Integrations

This directory contains examples of how to integrate **Agent-Gantry** with popular agent frameworks.

## Core Concept

Agent-Gantry acts as a **Semantic Tool Router**. Instead of passing all your tools to an agent (which increases token costs and reduces accuracy), you use Agent-Gantry to retrieve only the most relevant tools for the current user query.

The general pattern is:
1. **Register** all your tools with `AgentGantry`.
2. **Retrieve** relevant tools using `gantry.retrieve_tools(query)`.
3. **Wrap** the retrieved tools in the framework's native tool format.
4. **Execute** the tools via `gantry.execute(ToolCall(...))`.

## Examples Included

- `agent_framework_example.py`: Microsoft Agent Framework integration using `ChatAgent`.
- `google_adk_example.py`: Google Agent Development Kit (ADK) integration using `Agent` + `Runner`.
- `langchain_example.py`: Using Gantry with LangChain 0.3+ `create_agent`.
- `langgraph_example.py`: Integrating Gantry into a LangGraph workflow.
- `crewai_example.py`: Using Gantry to provide dynamic tools to a CrewAI Agent (using `kickoff_async`).
- `autogen_example.py`: Registering Gantry tools for an AutoGen (AG2) `AssistantAgent`.
- `llamaindex_example.py`: Using Gantry with LlamaIndex's new `AgentWorkflow` based `ReActAgent`.
- `semantic_kernel_example.py`: Registering Gantry tools as Semantic Kernel plugins with auto-function calling.

## Installation Requirements

We recommend using `uv` for managing dependencies.

```bash
# Install all framework dependencies
uv add langchain langchain-openai langgraph crewai autogen-agentchat autogen-ext[openai] \
	llama-index-core llama-index-llms-openai semantic-kernel agent-framework google-adk
```

**Note on Python Version:** These examples are verified on **Python 3.13**, but should work on any supported Agent-Gantry version (**Python 3.10+**).

## Environment Variables

Ensure you have your API keys set in a `.env` file:

```env
OPENAI_API_KEY=sk-...
# Other keys as needed for specific providers
```
