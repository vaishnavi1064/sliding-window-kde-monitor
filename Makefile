PYTHON := .venv/Scripts/python.exe

.PHONY: help venv test bench clean

help:
	@echo "venv   - create the 3.12 venv and install the package with dev deps"
	@echo "test   - run all validation tiers"
	@echo "bench  - benchmark sketch update/query throughput"
	@echo "clean  - remove caches and build artifacts"

venv:
	py -3.12 -m venv .venv
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e ".[dev]"

test:
	$(PYTHON) -m pytest -v

bench:
	$(PYTHON) -m scripts.benchmark

clean:
	rm -rf .pytest_cache sketch.egg-info
	find . -type d -name __pycache__ -not -path "./.venv/*" -exec rm -rf {} +
