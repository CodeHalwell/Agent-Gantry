export type JourneyStageSource = {
  title: string;
  detail: string;
  code: string;
};

export const journeyStages: JourneyStageSource[] = [
  {
    title: 'Register',
    detail:
      'Decorate ordinary Python functions and let dotfetch infer schemas from type hints and docstrings.',
    code: `@fetcher.register(tags=["finance"])
def calculate_tax(amount: float) -> float:
    """Calculate tax."""
    return amount * 0.08`,
  },
  {
    title: 'Retrieve',
    detail:
      'Sync once, then retrieve only the tools that match a user request, conversation state, and policy constraints.',
    code: `tools = await fetcher.retrieve_tools(
    "What is tax on $100?",
    limit=5,
    dialect="openai",
)`,
  },
  {
    title: 'Execute',
    detail:
      'Execute via the built-in engine with async support, timeouts, retries, circuit breakers, callbacks, and telemetry.',
    code: `result = await fetcher.execute(ToolCall(
    tool_name="calculate_tax",
    arguments={"amount": 100},
))`,
  },
  {
    title: 'Operate',
    detail:
      'Persist embeddings, route to MCP/A2A, bridge agent frameworks, and instrument usage across production workflows.',
    code: `fetcher = DotFetch(config=DotFetchConfig(
    vector_store={"provider": "lancedb"},
    telemetry={"provider": "opentelemetry"},
))`,
  },
];
