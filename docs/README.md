# Agent-Gantry documentation

The documentation has been upgraded from a static Jekyll-style folder to an **Astro + React + TypeScript** site.

## Develop locally

```bash
npm install
npm run dev
```

## Build and verify

```bash
npm run build
npm run preview
```

The source pages live in `src/pages`, shared visual structure lives in `src/layouts`, interactive React components live in `src/components`, and global styling lives in `src/styles/global.css`.

Legacy markdown files remain in this folder as historical reference and audit material while the rich docs experience is served by Astro.
