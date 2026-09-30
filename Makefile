VENV ?= .venv
PY := $(VENV)/bin/python

.PHONY: setup lint typecheck test regress regress-update check

setup:
	python3 -m venv $(VENV)
	$(PY) -m pip install -q --upgrade pip
	$(PY) -m pip install -q -e ".[dev]"

lint:
	$(PY) -m ruff check cua tests

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

# Everything CI runs.
check: lint typecheck test regress
