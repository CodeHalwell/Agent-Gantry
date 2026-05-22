# LLM SDK Compatibility Guide (May 2026)

This document details Agent-Gantry's compatibility with major Python LLM SDKs as of May 2026.

## Overview

Agent-Gantry is designed to work seamlessly with leading LLM providers. This guide covers:
- Installation and setup
- Client initialization patterns
- Key endpoint methods
- Integration examples with Agent-Gantry
- Known incompatibilities and workarounds

## Supported LLM Providers

| Provider | Package | Status | Notes |
|----------|---------|--------|-------|
| OpenAI | `openai` | ✅ Full Support | Primary integration |
| Azure OpenAI | `openai` | ✅ Full Support | Uses AzureOpenAI client |
| Anthropic | `anthropic` | ✅ Full Support | Claude models |
| Google GenAI | `google-genai` | ✅ Full Support | For prototyping |
| Google Vertex AI | `google-cloud-aiplatform` | ✅ Full Support | Production recommended |
| Mistral | `openai` | ⚠️ OpenAI-compatible endpoint | `mistralai` quarantined on PyPI (2026-05-12); use OpenAI SDK with `base_url="https://api.mistral.ai/v1"` |
| Groq | `groq` | ✅ Full Support | Fast inference |
| OpenRouter | `openai` | ✅ Full Support | Via base_url override |

## Installation

Install Agent-Gantry with LLM provider support:

```bash
# Install with all LLM providers
pip install agent-gantry[llm-providers]

# Or install specific providers
pip install agent-gantry[openai]
pip install agent-gantry[anthropic]
pip install agent-gantry[google-genai]
pip install agent-gantry[google-vertexai]
pip install agent-gantry[mistral]
pip install agent-gantry[groq]

# Install all dependencies including LLM providers
pip install agent-gantry[all]
```

---

## OpenAI

### Package
```bash
pip install "openai>=2.38.0"
```

### Client Initialization
```python
from openai import OpenAI

# Standard initialization
client = OpenAI(api_key="your-api-key")

# Or with environment variable (OPENAI_API_KEY)
client = OpenAI()
```

### Key Methods

#### Chat Completions
```python
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"}
    ]
)
print(response.choices[0].message.content)
```

#### Responses API (Modern)
```python
# The newer Responses API with simpler tool format.
# GPT-4.1 is the recommended model for agentic / Responses API workloads.
response = client.responses.create(
    model="gpt-4.1",
    instructions="You are a helpful assistant.",
    input="Hello!"
)
print(response.output_text)
```

#### Realtime API (Beta)
```python
# WebSocket-based realtime conversations
async with client.beta.realtime.connect(model="gpt-4o-realtime-preview") as conn:
    await conn.send({"type": "input_audio_buffer.append", "audio": audio_data})
    async for event in conn:
        if event.type == "response.audio.delta":
            # Handle audio response
            pass
```

#### Audio Transcriptions
```python
with open("audio.mp3", "rb") as audio_file:
    transcription = client.audio.transcriptions.create(
        model="whisper-1",
        file=audio_file
    )
print(transcription.text)
```

### Agent-Gantry Integration

```python
from openai import OpenAI
from agent_gantry import AgentGantry

# Initialize both
client = OpenAI()
gantry = AgentGantry()

@gantry.register(tags=["utility"])
def get_current_weather(location: str) -> str:
    """Get weather for a location."""
    return f"Sunny, 72°F in {location}"

await gantry.sync()

# Get tools in OpenAI Chat Completions format (default)
tools = await gantry.retrieve_tools("What's the weather?", limit=5)

# Use with OpenAI Chat Completions API
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "What's the weather in SF?"}],
    tools=tools
)

# Or get tools in OpenAI Responses API format
tools_responses = await gantry.retrieve_tools(
    "What's the weather?", 
    limit=5, 
    dialect="openai_responses"
)

# Use with OpenAI Responses API
response = client.responses.create(
    model="gpt-4.1",
    input="What's the weather in SF?",
    tools=tools_responses
)
```

---

## Azure OpenAI

### Package
```bash
pip install "openai>=2.38.0"
```

### Client Initialization
```python
from openai import AzureOpenAI

client = AzureOpenAI(
    api_key="your-azure-api-key",
    api_version="2024-10-21",
    azure_endpoint="https://your-resource.openai.azure.com"
)
```

### Key Methods

#### Chat Completions
```python
response = client.chat.completions.create(
    model="gpt-4o",  # Your deployment name
    messages=[
        {"role": "user", "content": "Hello!"}
    ]
)
```

#### Responses API
```python
# The Responses API uses a different format for tools and responses.
# Use gpt-4.1 for agentic / Responses API workloads (recommended over gpt-4o).
response = client.responses.create(
    model="gpt-4.1",  # Your deployment name; gpt-4.1 recommended for Responses API
    input="Analyse this document and extract key points",
    tools=[
        {
            "type": "function",
            "name": "extract_key_points",
            "description": "Extract key points from text",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"}
                },
                "required": ["text"]
            }
        }
    ]
)

# Handle function calls from Responses API
for output in response.output:
    if output.type == "function_call":
        print(f"Tool called: {output.name}")
        print(f"Arguments: {output.arguments}")
        # Execute tool and send result back
        result = client.responses.create(
            model="gpt-4.1",
            previous_response_id=response.id,
            input=[{
                "type": "function_call_output",
                "call_id": output.call_id,
                "output": "your tool result here"
            }]
        )
```

### Agent-Gantry Integration

```python
from openai import AzureOpenAI
from agent_gantry import AgentGantry

client = AzureOpenAI(
    api_key="your-key",
    api_version="2024-10-21",
    azure_endpoint="https://your-resource.openai.azure.com"
)
gantry = AgentGantry()

# Register tools and use with Azure OpenAI
tools = await gantry.retrieve_tools("your query")
response = client.chat.completions.create(
    model="your-deployment",
    messages=[{"role": "user", "content": "query"}],
    tools=tools
)
```

---

## Anthropic (Claude)

### Package
```bash
pip install "anthropic>=0.104.0"
```

### Client Initialization
```python
from anthropic import Anthropic

# Standard initialization
client = Anthropic(api_key="your-api-key")

# Or with environment variable (ANTHROPIC_API_KEY)
client = Anthropic()
```

### Key Methods

#### Messages
```python
response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "Hello, Claude!"}
    ]
)
print(response.content[0].text)
```

#### Prompt Caching
```python
# Cache long system prompts for efficiency using cache_control in the standard API.
# The beta.prompt_caching namespace is deprecated in favour of this approach.
response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=1024,
    system=[{
        "type": "text",
        "text": "Long system prompt...",
        "cache_control": {"type": "ephemeral"}
    }],
    messages=[{"role": "user", "content": "Question?"}]
)
```

### Agent-Gantry Integration

```python
from anthropic import Anthropic
from agent_gantry import AgentGantry
from agent_gantry.schema.query import ConversationContext, ToolQuery

client = Anthropic()
gantry = AgentGantry()

@gantry.register
def search_database(query: str) -> str:
    """Search the database."""
    return f"Results for: {query}"

await gantry.sync()

# Retrieve tools and convert to Anthropic format via to_dialect("anthropic").
# This avoids manual field-mapping and keeps conversion logic centralised.
retrieval = await gantry.retrieve(
    ToolQuery(context=ConversationContext(query="search for data"), limit=5)
)
anthropic_tools = [t.tool.to_dialect("anthropic") for t in retrieval.tools]

response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=1024,
    tools=anthropic_tools,
    messages=[{"role": "user", "content": "Search for user data"}]
)
```

---

## Google GenAI

> **Note**: For production workloads, use Vertex AI instead. Google GenAI is recommended for prototyping.
>
> **v2.0 compatibility note (May 2026)**: `google-genai` 2.0.0 (released 2026-05-07) introduced breaking changes *only* in the Interactions API (SSE event renames, `response_format` field restructuring). The `GenerateContent` and function-calling surfaces Gantry uses are entirely unaffected. However, `google-adk` (the Google Agent Development Kit, included in `agent-gantry[agent-frameworks]`) requires `google-genai<2.0.0` across all current versions. If you need `google-genai 2.x`, install without `google-adk` (`pip install agent-gantry[google-genai]`). See [official migration notes](https://ai.google.dev/gemini-api/docs/interactions-breaking-changes-may-2026).

### Package
```bash
pip install "google-genai>=1.75.0"
```

### Client Initialization
```python
from google import genai

# Standard initialization with API key
client = genai.Client(api_key="your-api-key")

# Or with environment variable (GOOGLE_API_KEY)
client = genai.Client()
```

### Key Methods

#### Generate Content
```python
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Hello, Gemini!"
)
print(response.text)
```

#### Streaming
```python
for chunk in client.models.generate_content_stream(
    model="gemini-2.5-flash",
    contents="Write a story about a robot."
):
    print(chunk.text, end="")
```

### Agent-Gantry Integration

```python
from google import genai
from agent_gantry import AgentGantry
from agent_gantry.schema.query import ConversationContext, ToolQuery

client = genai.Client()
gantry = AgentGantry()

@gantry.register
def calculate(a: float, b: float, operation: str) -> str:
    """Perform a math operation on two numbers."""
    if operation == "add":
        return str(a + b)
    elif operation == "subtract":
        return str(a - b)
    elif operation == "multiply":
        return str(a * b)
    elif operation == "divide":
        return str(a / b) if b != 0 else "Error: Division by zero"
    return "Error: Unknown operation"

await gantry.sync()

# Retrieve tools in Gemini format and wrap as FunctionDeclaration objects.
from google.genai import types

retrieval = await gantry.retrieve(
    ToolQuery(context=ConversationContext(query="calculate something"), limit=5)
)
gemini_funcs = [
    types.FunctionDeclaration(**t.tool.to_dialect("gemini"))
    for t in retrieval.tools
]
tool_config = types.GenerateContentConfig(tools=[types.Tool(function_declarations=gemini_funcs)])

response = await client.aio.models.generate_content(
    model="gemini-2.5-flash",
    contents="What is 2 + 2?",
    config=tool_config,
)
```

---

## Google Vertex AI

> **Recommended for production workloads.**

### Package
```bash
pip install google-cloud-aiplatform>=1.70.0
```

### Setup
```python
import vertexai
from vertexai.generative_models import GenerativeModel

# Initialize Vertex AI
vertexai.init(
    project="your-project-id",
    location="us-central1"
)
```

### Key Methods

#### Generate Content
```python
model = GenerativeModel("gemini-2.5-flash")

response = model.generate_content("Hello, Vertex AI!")
print(response.text)
```

#### Chat Sessions
```python
model = GenerativeModel("gemini-2.5-flash")
chat = model.start_chat()

response = chat.send_message("What's the weather like?")
print(response.text)

# Continue the conversation
response = chat.send_message("And tomorrow?")
print(response.text)
```

### Agent-Gantry Integration

```python
import vertexai
from vertexai.generative_models import GenerativeModel, Tool, FunctionDeclaration
from agent_gantry import AgentGantry

vertexai.init(project="your-project", location="us-central1")

gantry = AgentGantry()

@gantry.register
def get_stock_price(symbol: str) -> str:
    """Get current stock price."""
    return f"${symbol}: $150.00"

await gantry.sync()

# Retrieve tools in Gemini format and unpack directly into FunctionDeclaration.
from agent_gantry.schema.query import ConversationContext, ToolQuery

retrieval = await gantry.retrieve(
    ToolQuery(context=ConversationContext(query="stock price"), limit=5)
)
vertex_functions = [
    FunctionDeclaration(**t.tool.to_dialect("gemini"))
    for t in retrieval.tools
]

model = GenerativeModel(
    "gemini-2.5-flash",
    tools=[Tool(function_declarations=vertex_functions)]
)

response = model.generate_content("What's AAPL stock price?")
```

---

## Mistral

> **⚠️ Note (2026-05-12):** The `mistralai` package was quarantined on PyPI and is
> no longer installable. Mistral's chat endpoint is OpenAI-compatible; use the
> `openai` SDK with `base_url="https://api.mistral.ai/v1"` instead. All Gantry
> tool schemas work without modification.

### Package
```bash
pip install "agent-gantry[openai]"
# or: pip install openai
```

### Client Initialization
```python
from openai import AsyncOpenAI

# Mistral is OpenAI-compatible — point AsyncOpenAI at the Mistral base URL.
client = AsyncOpenAI(api_key="your-mistral-api-key", base_url="https://api.mistral.ai/v1")
```

### Key Methods

#### Chat Complete (async)
```python
from openai import AsyncOpenAI

client = AsyncOpenAI(api_key="your-mistral-api-key", base_url="https://api.mistral.ai/v1")
response = await client.chat.completions.create(
    model="mistral-large-latest",
    messages=[{"role": "user", "content": "Hello, Mistral!"}],
)
print(response.choices[0].message.content)
```

### Agent-Gantry Integration

```python
from openai import AsyncOpenAI
from agent_gantry import AgentGantry

gantry = AgentGantry()

@gantry.register
def send_notification(message: str, channel: str) -> str:
    """Send a notification."""
    return f"Sent '{message}' to {channel}"

await gantry.sync()

# Gantry's default dialect produces OpenAI-compatible function schemas.
tools = await gantry.retrieve_tools("send notification")

client = AsyncOpenAI(api_key="your-key", base_url="https://api.mistral.ai/v1")
response = await client.chat.completions.create(
    model="mistral-large-latest",
    messages=[{"role": "user", "content": "Notify the team"}],
    tools=tools,
)
```

---

## Groq

### Package
```bash
pip install groq>=1.2.0
```

### Client Initialization
```python
from groq import Groq

client = Groq(api_key="your-api-key")

# Or with environment variable (GROQ_API_KEY)
client = Groq()
```

### Key Methods

#### Chat Completions
```python
response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
        {"role": "user", "content": "Hello, Groq!"}
    ]
)
print(response.choices[0].message.content)
```

### Agent-Gantry Integration

```python
from groq import Groq
from agent_gantry import AgentGantry

client = Groq()
gantry = AgentGantry()

@gantry.register
def analyze_text(text: str) -> str:
    """Analyze text sentiment."""
    return "Positive sentiment detected"

await gantry.sync()

# Get tools in OpenAI-compatible format
tools = await gantry.retrieve_tools("analyze text")

# Use with Groq (fully OpenAI-compatible)
response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[{"role": "user", "content": "Analyze: I love this!"}],
    tools=tools
)
```

---

## OpenRouter (and OpenAI-Compatible APIs)

OpenRouter and other OpenAI-compatible providers (DeepSeek, Perplexity, Together AI, etc.) can be used via the standard OpenAI client with a custom `base_url`.

### Package
```bash
pip install "openai>=2.38.0"
```

### Client Initialization
```python
from openai import OpenAI

# OpenRouter
client = OpenAI(
    api_key="your-openrouter-key",
    base_url="https://openrouter.ai/api/v1"
)

# DeepSeek
client = OpenAI(
    api_key="your-deepseek-key",
    base_url="https://api.deepseek.com/v1"
)

# Perplexity
client = OpenAI(
    api_key="your-perplexity-key",
    base_url="https://api.perplexity.ai"
)
```

### Key Methods
```python
# Standard OpenAI-compatible chat completions
response = client.chat.completions.create(
    model="anthropic/claude-sonnet-4.6",  # OpenRouter model format
    messages=[
        {"role": "user", "content": "Hello!"}
    ]
)
```

### Agent-Gantry Integration

```python
from openai import OpenAI
from agent_gantry import AgentGantry

# OpenRouter client
client = OpenAI(
    api_key="your-openrouter-key",
    base_url="https://openrouter.ai/api/v1"
)
gantry = AgentGantry()

@gantry.register
def web_search(query: str) -> str:
    """Search the web."""
    return f"Results for: {query}"

await gantry.sync()

# Tools work with any OpenAI-compatible provider
tools = await gantry.retrieve_tools("search the web")

response = client.chat.completions.create(
    model="openai/gpt-4o",
    messages=[{"role": "user", "content": "Search for Python tutorials"}],
    tools=tools
)
```

---

## Tool Format Conversion

Agent-Gantry provides OpenAI Chat Completions compatible tool schemas by default. Here's how to convert them for other providers:

### OpenAI Chat Completions Format (Default)
```python
{
    "type": "function",
    "function": {
        "name": "my_tool",
        "description": "Tool description",
        "parameters": {
            "type": "object",
            "properties": {...},
            "required": [...]
        }
    }
}
```

### OpenAI Responses API Format
```python
# Use dialect="openai_responses" when retrieving tools
tools = await gantry.retrieve_tools("query", dialect="openai_responses")

# Format structure:
{
    "type": "function",
    "name": "my_tool",
    "description": "Tool description",
    "parameters": {
        "type": "object",
        "properties": {...},
        "required": [...]
    }
}

# Tool call format (from response.output):
{
    "type": "function_call",
    "call_id": "call_xxx",
    "name": "my_tool",
    "arguments": "{\"arg\": \"value\"}"
}

# Tool result format (to send back):
{
    "type": "function_call_output",
    "call_id": "call_xxx",
    "output": "result string"
}
```

### Anthropic Format
```python
# Use Agent-Gantry's built-in dialect converter — avoids manual field-mapping errors.
from agent_gantry.schema.query import ConversationContext, ToolQuery

retrieval = await gantry.retrieve(
    ToolQuery(context=ConversationContext(query="your query"), limit=5)
)
anthropic_tools = [t.tool.to_dialect("anthropic") for t in retrieval.tools]
# Each entry: {"name": "...", "description": "...", "input_schema": {...}}
```

### Vertex AI Format
```python
from vertexai.generative_models import FunctionDeclaration
from agent_gantry.schema.query import ConversationContext, ToolQuery

retrieval = await gantry.retrieve(
    ToolQuery(context=ConversationContext(query="your query"), limit=5)
)
# to_dialect("gemini") returns {"name": ..., "description": ..., "parameters": ...}
# which maps directly onto FunctionDeclaration's constructor.
vertex_functions = [
    FunctionDeclaration(**t.tool.to_dialect("gemini"))
    for t in retrieval.tools
]
```

---

## Known Incompatibilities and Workarounds

### 1. Google GenAI vs Legacy google-generativeai

**Issue**: The `google-generativeai` package is deprecated in favor of `google-genai`.

**Workaround**: Use `google-genai` for new projects:
```python
# Old (deprecated)
# import google.generativeai as genai

# New (recommended)
from google import genai
```

### 2. OpenAI Chat Completions vs Responses API

**Issue**: OpenAI has two APIs with different tool formats:
- **Chat Completions** (`client.chat.completions.create`): Traditional API with nested `function` key
- **Responses API** (`client.responses.create`): Newer API with flattened tool schema

**Workaround**: Agent-Gantry supports both via the `dialect` parameter:
```python
# For Chat Completions API (default)
tools = await gantry.retrieve_tools("query", dialect="openai")

# For Responses API
tools = await gantry.retrieve_tools("query", dialect="openai_responses")
```

### 3. Tool Schema Differences

**Issue**: Different providers have slightly different tool schema formats.

**Workaround**: Use Agent-Gantry's OpenAI-compatible output and transform as needed (see Tool Format Conversion section above).

### 4. Streaming Differences

**Issue**: Streaming implementations vary across providers.

**Workaround**: Normalize streaming handling in your application layer:
```python
# OpenAI/Azure/Groq/Mistral
for chunk in response:
    if chunk.choices[0].delta.content:
        yield chunk.choices[0].delta.content

# Anthropic
for chunk in response:
    if chunk.type == "content_block_delta":
        yield chunk.delta.text

# Google GenAI
for chunk in response:
    yield chunk.text
```

---

## Environment Variables

| Provider | Environment Variable |
|----------|---------------------|
| OpenAI | `OPENAI_API_KEY` |
| Azure OpenAI | `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT` |
| Anthropic | `ANTHROPIC_API_KEY` |
| Google GenAI | `GOOGLE_API_KEY` |
| Google Vertex AI | `GOOGLE_APPLICATION_CREDENTIALS` (service account) |
| Mistral | `MISTRAL_API_KEY` |
| Groq | `GROQ_API_KEY` |
| OpenRouter | `OPENROUTER_API_KEY` |

---

## Best Practices

1. **Use Agent-Gantry for Tool Management**: Let Agent-Gantry handle tool registration, semantic routing, and execution while using LLM providers for inference.

2. **Prefer OpenAI-Compatible Format**: Agent-Gantry outputs OpenAI-compatible tool schemas which work with most providers directly.

3. **Feature Detection**: Check for provider-specific features before using them to maintain portability.

4. **Error Handling**: Implement proper error handling for API calls, as error formats vary across providers.

5. **Rate Limiting**: Be aware of different rate limits across providers and implement appropriate backoff strategies.

---

## References

- [OpenAI Python SDK](https://github.com/openai/openai-python)
- [Anthropic Python SDK](https://github.com/anthropics/anthropic-sdk-python)
- [Google GenAI SDK](https://github.com/googleapis/python-genai)
- [Google Cloud AI Platform](https://cloud.google.com/python/docs/reference/aiplatform/latest)
- [Mistral AI API docs](https://docs.mistral.ai/) (use OpenAI SDK with `base_url="https://api.mistral.ai/v1"`; `mistralai` package is quarantined)
- [Groq SDK](https://github.com/groq/groq-python)
- [OpenRouter Documentation](https://openrouter.ai/docs)
