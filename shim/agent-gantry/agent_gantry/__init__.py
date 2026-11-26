"""Legacy shim for the old ``agent-gantry`` package name.

agent-gantry has been renamed to dotfetch. This final release exists only to
point existing installs at the new name, so importing it always fails.
"""

from __future__ import annotations

raise ImportError(
    "agent-gantry has been renamed to dotfetch. dotfetch is already installed "
    "alongside this package. Update your code: replace 'import agent_gantry' "
    "with 'import dotfetch', and 'from agent_gantry import AgentGantry' with "
    "'from dotfetch import DotFetch'. See "
    "https://github.com/CodeHalwell/dotfetch for the migration notes."
)
