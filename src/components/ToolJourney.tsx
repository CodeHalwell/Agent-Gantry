import { useMemo, useState } from 'react';

type Stage = { title: string; detail: string; code: string };
const stages: Stage[] = [
  { title: 'Register', detail: 'Decorate ordinary Python functions and let Gantry infer schemas from type hints and docstrings.', code: '@gantry.register(tags=["finance"] )\ndef calculate_tax(amount: float) -> float:\n    """Calculate tax."""\n    return amount * 0.08' },
  { title: 'Retrieve', detail: 'Sync once, then retrieve only the tools that match a user request, conversation state, and policy constraints.', code: 'tools = await gantry.retrieve_tools(\n    "What is tax on $100?",\n    limit=5,\n    dialect="openai",\n)' },
  { title: 'Execute', detail: 'Execute via the built-in engine with async support, timeouts, retries, circuit breakers, callbacks, and telemetry.', code: 'result = await gantry.execute(ToolCall(\n    tool_name="calculate_tax",\n    arguments={"amount": 100},\n))' },
  { title: 'Operate', detail: 'Persist embeddings, route to MCP/A2A, bridge agent frameworks, and instrument usage across production workflows.', code: 'gantry = AgentGantry(config=AgentGantryConfig(\n    vector_store={"provider": "lancedb"},\n    telemetry={"provider": "opentelemetry"},\n))' },
];
export default function ToolJourney(){
 const [active,setActive]=useState(0); const stage=useMemo(()=>stages[active], [active]);
 return <div className="card"><div className="grid">{stages.map((s,i)=><button key={s.title} onClick={()=>setActive(i)} style={{padding:'1rem',borderRadius:'1rem',border:'1px solid var(--line)',background:i===active?'linear-gradient(135deg,#6ee7f9,#95f985)':'#101827',color:i===active?'#071018':'var(--text)',fontWeight:800,cursor:'pointer'}}>{i+1}. {s.title}</button>)}</div><h3>{stage.title}</h3><p className="lead" style={{fontSize:'1rem'}}>{stage.detail}</p><pre><code>{stage.code}</code></pre></div>
}
