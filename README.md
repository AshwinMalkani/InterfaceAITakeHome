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

Irreversible capabilities (e.g. `cu_core.member.open_sub_account`) stop before the irreversible
step unless the capability's current content is approved **and** the run passes `--allow-irreversible`:

```bash
.venv/bin/python -m cua approve cu_core.member.open_sub_account --by "reviewer name"
.venv/bin/python -m cua replay cu_core.member.open_sub_account --base-url http://127.0.0.1:8001 \
    --param member_id=10002 --param "account_type=Money Market" --param "nickname=Rainy day" \
    --param opening_deposit=100.00 --allow-irreversible
```

Output values are masked by default (`--reveal` prints them). Each run writes `runs/<run_id>/`
with `events.jsonl` (structured log), `result.json`, and a masked screenshot on failure.

## Another tenant, same artifact

Capabilities are recorded once per vendor product. A tenant running the same product gets a small binding
(`config/tenants/beta.yaml`: relabelled fields, rare per-step patches) instead of a re-recording:

```bash
make app-beta                                         # tenant beta on :8002
.venv/bin/python -m cua replay cu_core.member.get_share_balance --base-url http://127.0.0.1:8002 \
    --param member_id=10001 --tenant beta
```

## Human handoff

Run with `--hitl` and failures a person can resolve pause the run on the same live browser instead of
failing. Examples: an unknown modal, a missing element, or an irreversible step without approval.

```bash
.venv/bin/python -m cua console                      # terminal 2: operator console, http://127.0.0.1:8090
.venv/bin/python -m cua replay cu_core.member.open_sub_account --base-url http://127.0.0.1:8001 \
    --param member_id=10002 --param "account_type=Money Market" --param "nickname=Rainy day" \
    --param opening_deposit=100.00 --hitl --headed     # stops before Confirm and asks for approval
```

In the console, **Take control**: the session's lease moves to you, and automation can't act while you
hold it. Work in the open browser window (or attach to `--cdp-port` via chrome://inspect). Then **Hand
back** with *resume at step*, *approve* (irreversible steps, this run only) or *abort*. Your clicks are
captured (never the values you type) and the run resumes on the same session.

## Development

```bash
make setup     # create .venv and install
make check     # lint + mypy (strict) + unit tests + regression suite
make regress   # regression suite only (no API key required)
```
