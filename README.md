# Computer-Use Automation System

LLM-driven discovery of UI flows → typed, versioned capability artifacts → deterministic replay without the LLM,
with human-in-the-loop handoff and safety guardrails. Built for the interface.ai take-home (Assignment A).

> Work in progress — full setup, demo path, and design write-up (`REPORT.md`) land as milestones complete.

## Replay a capability (no LLM, no API key)

```bash
make app                                   # terminal 1: target app, tenant alpha, on :8001
export CU_CORE_USERNAME=teller1 CU_CORE_PASSWORD=demo-only-password
.venv/bin/python -m cua replay cu_core.member.get_share_balance \
    --base-url http://127.0.0.1:8001 --param member_id=10001
```

Output values are masked by default (`--reveal` prints them). Each run writes `runs/<run_id>/`
with `events.jsonl` (structured log), `result.json`, and a masked screenshot on failure.

## Development

```bash
make setup     # create .venv and install
make check     # lint + mypy (strict) + unit tests + regression suite
make regress   # regression suite only (no API key required)
```
