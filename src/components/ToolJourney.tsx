import { useMemo, useState } from 'react';

type Stage = { title: string; detail: string; codeHtml: string };

type Props = { stages: Stage[] };

export default function ToolJourney({ stages }: Props) {
  const [active, setActive] = useState(0);
  const stage = useMemo(() => stages[active], [active, stages]);

  return (
    <div className="card">
      <div className="grid">
        {stages.map((s, i) => (
          <button
            key={s.title}
            onClick={() => setActive(i)}
            style={{
              padding: '1rem',
              borderRadius: '1rem',
              border: '1px solid var(--line)',
              background:
                i === active
                  ? 'linear-gradient(135deg,#6ee7f9,#95f985)'
                  : '#101827',
              color: i === active ? '#071018' : 'var(--text)',
              fontWeight: 800,
              cursor: 'pointer',
            }}
          >
            {i + 1}. {s.title}
          </button>
        ))}
      </div>
      <h3>{stage.title}</h3>
      <p className="lead" style={{ fontSize: '1rem' }}>
        {stage.detail}
      </p>
      <figure
        className="code-block"
        dangerouslySetInnerHTML={{ __html: stage.codeHtml }}
      />
    </div>
  );
}
