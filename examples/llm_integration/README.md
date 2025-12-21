# examples/llm_integration

End-to-end examples that pair Agent-Gantry with LLM SDKs. Scripts retrieve tools, pass them to the
provider, and execute whichever tool the model selects.

## Files
- `llm_demo.py`: Provider-agnostic chat loop (retrieve -> chat -> execute).
- `decorator_demo.py`: Uses `with_semantic_tools` to inject top-k tools into SDK calls automatically.
- `token_savings_demo.py`: Benchmarks prompt-token savings when surfacing only the top 2 tools.
- `openai_demo.py`, `anthropic_demo.py`, `google_genai_demo.py`, `mistral_demo.py`, `groq_demo.py`:
  Provider-specific runs that highlight schema conversion and credential handling.

## Run commands

```bash
python examples/llm_integration/llm_demo.py
python examples/llm_integration/decorator_demo.py
python examples/llm_integration/token_savings_demo.py
```

Provider-specific scripts require credentials (e.g., `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`). Other
scripts can run offline using the simple embedder. Check the script headers for environment variable
hints before running.
