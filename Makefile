VENV ?= .venv
PY := $(VENV)/bin/python

.PHONY: setup lint typecheck test regress regress-update check app app-beta schema

setup:
	python3 -m venv $(VENV)
	$(PY) -m pip install -q --upgrade pip
	$(PY) -m pip install -q -e ".[dev]"

lint:
	$(PY) -m ruff check cua apps tests

typecheck:
	$(PY) -m mypy

# Fast unit tests only.
test:
	$(PY) -m pytest -m "not regression"

# Regression suite: guards + replay cases from tests/regression/cases.yaml. No API key needed.
regress:
	$(PY) -m pytest -m regression

# Regenerate golden results deliberately; review the diff like code.
regress-update:
	REGRESS_UPDATE=1 $(PY) -m pytest -m regression

# Regenerate the published JSON Schema for capability artifacts.
schema:
	$(PY) -c "from cua.artifact.store import write_json_schema; write_json_schema()"

# Everything CI runs.
check: lint typecheck test regress

# Target app, one tenant per port. Sign on as teller1 / $$CU_CORE_PASSWORD (default: demo-only-password).
app:
	$(PY) -m apps.cu_core --tenant alpha --port 8001

app-beta:
	$(PY) -m apps.cu_core --tenant beta --port 8002
