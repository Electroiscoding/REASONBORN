# System-Prompt Semantics

ReasonBorn implements a strict Operator vs. User precedence system.
Operator configs (e.g., `research_mode.json`) override user requests.

Conflict resolution rules:
- Mode: Operator determines strictly.
- Outputs: Intersection (most restrictive wins).
- Safety: Maximum sensitivity wins.
- Resource Limits: Minimum constraints win.
