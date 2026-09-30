# Computer-Use Automation System

LLM-driven discovery of UI flows → typed, versioned capability artifacts → deterministic replay without the LLM,
with human-in-the-loop handoff and safety guardrails. Built for the interface.ai take-home (Assignment A).

> Work in progress — full setup, demo path, and design write-up (`REPORT.md`) land as milestones complete.

## Development

```bash
make setup     # create .venv and install
make check     # lint + mypy (strict) + unit tests + regression suite
make regress   # regression suite only (no API key required)
```
