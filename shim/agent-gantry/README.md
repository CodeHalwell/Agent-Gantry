# agent-gantry

agent-gantry is now dotfetch. This is the final release under the old name, and it installs dotfetch.

To get the latest version:

```bash
pip install dotfetch
```

## Migrating

0.27.0 is a clean break, so existing code needs updating. There are no compatibility aliases.

- `agent_gantry` becomes `dotfetch`
- `AgentGantry` becomes `DotFetch`
- `AgentGantryConfig` becomes `DotFetchConfig`
- Classes prefixed `Gantry` become `DotFetch`
- The CLI `agent-gantry` becomes `dotfetch`
- The logger name `agent_gantry` becomes `dotfetch`

For example:

```python
# before
from agent_gantry import AgentGantry

# after
from dotfetch import DotFetch
```

The project lives at https://github.com/CodeHalwell/dotfetch.

## Licence

MIT. See `LICENSE`.
