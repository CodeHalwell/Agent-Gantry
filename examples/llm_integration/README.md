# examples/llm_integration

End-to-end examples that pair Agent-Gantry with LLM SDKs. Scripts retrieve tools, pass them to the
provider, and execute whichever tool the model selects.

## Files

Run offline (no key needed; an optional `OPENAI_API_KEY` swaps the mock for the real model):
- `llm_demo.py`: Provider-agnostic chat loop (retrieve -> chat -> execute) with a mocked model fallback.
- `multi_turn_conversation.py`: Two-turn conversation that re-routes tools on every turn.
- `decorator_demo.py`: Uses `with_semantic_tools` to inject top-k tools into a (mock) SDK call, both with `set_default_gantry()` and with an explicit gantry.
- `token_savings_demo.py`: Counts prompt tokens for all 15 tool schemas vs. Gantry's top 2 (tiktoken if installed).

Require a provider key (each exits with a message naming the variable if it is missing):
- `openai_demo.py` (`OPENAI_API_KEY`): Responses API, Chat Completions, decorator, and the typed `OpenAIAdapter`.
- `anthropic_demo.py` (`ANTHROPIC_API_KEY`): `to_dialect("anthropic")`, decorator with `dialect="anthropic"`, and the typed `AnthropicAdapter`.
- `anthropic_skills_demo.py` (`ANTHROPIC_API_KEY`): `create_skills_client` — bundles Gantry tools into named skills whose instructions go into the system prompt.
- `anthropic_thinking_demo.py` (`ANTHROPIC_API_KEY`): Interleaved and extended thinking via `create_anthropic_client`, with and without Gantry tools.
- `google_genai_demo.py` (`GOOGLE_API_KEY`): `to_dialect("gemini")` with the function-call round trip, decorator, and the typed `GeminiAdapter`.
- `groq_demo.py` (`GROQ_API_KEY`): OpenAI-style schemas into Groq's SDK, decorator, and the typed `GroqAdapter`.
- `mistral_demo.py` (`MISTRAL_API_KEY`): Mistral through the `openai` SDK with `base_url="https://api.mistral.ai/v1"` (the `mistralai` package is quarantined on PyPI), plus the typed `MistralAdapter`.

## Run commands

```bash
python examples/llm_integration/llm_demo.py
python examples/llm_integration/multi_turn_conversation.py
python examples/llm_integration/decorator_demo.py
python examples/llm_integration/token_savings_demo.py
```

Install the provider extra for the script you want, e.g. `pip install "agent-gantry[openai]"`,
`[anthropic]`, `[google-genai]`, `[groq]`. `mistral_demo.py` uses the `openai` extra. Check the
script headers for environment variable hints before running.
