const groups = [
 ['LLM SDKs', ['OpenAI / Azure / OpenRouter', 'Anthropic Claude', 'Google GenAI', 'Google Vertex AI', 'Groq', 'Mistral via OpenAI-compatible endpoint']],
 ['Frameworks', ['Microsoft Agent Framework', 'LangChain', 'LangGraph', 'LlamaIndex', 'CrewAI', 'Google ADK', 'Pydantic AI', 'OpenAI Agents SDK', 'Haystack', 'Agno', 'Strands Agents', 'DSPy']],
 ['Protocols and storage', ['MCP server/client routing', 'A2A server/executor', 'LanceDB persistence', 'Qdrant / Chroma / pgvector adapters', 'Nomic / OpenAI / sentence-transformers embeddings', 'Cohere / cross-encoder rerankers']],
];
export default function ProviderMatrix(){return <div className="grid">{groups.map(([name,items])=>{const id = `provider-matrix-${(name as string).toLowerCase().replace(/\s+/g, '-')}`; return <section className="card" key={name as string} aria-labelledby={id}><h3 id={id}>{name}</h3><ul role="list" style={{ display: 'flex', flexWrap: 'wrap', padding: 0, margin: 0, listStyle: 'none' }}>{(items as string[]).map(i=><li key={i}><span className="pill">{i}</span></li>)}</ul></section>})}</div>}
