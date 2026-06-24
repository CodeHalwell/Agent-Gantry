const groups = [
  [
    "LLM SDKs",
    [
      "OpenAI / Azure / OpenRouter",
      "Anthropic Claude",
      "Google GenAI",
      "Google Vertex AI",
      "Groq",
      "Mistral via OpenAI-compatible endpoint",
    ],
  ],
  [
    "Frameworks",
    [
      "Microsoft Agent Framework",
      "LangChain",
      "LangGraph",
      "LlamaIndex",
      "CrewAI",
      "AutoGen",
      "Semantic Kernel",
      "Google ADK",
      "Pydantic AI",
      "OpenAI Agents SDK",
      "Smolagents",
      "Haystack",
      "Agno",
    ],
  ],
  [
    "Protocols and storage",
    [
      "MCP server/client routing",
      "A2A server/executor",
      "LanceDB persistence",
      "Qdrant / Chroma / pgvector adapters",
      "Nomic / OpenAI / sentence-transformers embeddings",
      "Cohere / cross-encoder rerankers",
    ],
  ],
];
export default function ProviderMatrix() {
  return (
    <div className="grid">
      {groups.map(([name, items]) => (
        <section className="card" key={name as string}>
          <h3>{name}</h3>
          <ul aria-label={name as string} className="pill-list" role="list">
            {(items as string[]).map((i) => (
              <li key={i}>
                <span className="pill">{i}</span>
              </li>
            ))}
          </ul>
        </section>
      ))}
    </div>
  );
}
