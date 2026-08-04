PYTHON := .venv/Scripts/python.exe

# docker compose reads .env by itself; the Python entry points run on the host
# and do not, so load it for them explicitly. Absent .env is not an error --
# the compose defaults still apply and the code raises a clear message if a
# credential is genuinely missing.
LOAD_ENV := set -a; if [ -f .env ]; then . ./.env; fi; set +a;

.PHONY: help env venv native test bench bench-native tune memcheck memcheck-rows memcheck-crossover evaluate figures mlflow alerts mcp-verify mcp-serve data up down logs replay anomaly clean

help:
	@echo "env      - copy .env.example to .env (edit the placeholders before use)"
	@echo "venv     - create the 3.12 venv and install the package with dev+streaming deps"
	@echo "native   - compile the C++17 core (optional; the Python core is the default)"
	@echo "test     - run all validation tiers"
	@echo "bench    - benchmark sketch update/query throughput"
	@echo "bench-native - compare the native core against the Python core"
	@echo "tune     - sweep detector parameters against real data"
	@echo "memcheck - measure what cell reclamation buys"
	@echo "evaluate - full evaluation vs the four documented failures"
	@echo "figures  - evaluation figures + CSVs into docs/figures/"
	@echo "mlflow   - log operating points and artifacts to MLflow"
	@echo "alerts   - load detected episodes into Postgres for the MCP server"
	@echo "mcp-verify - exercise the three MCP tools end to end"
	@echo "mcp-serve  - run the MCP server on stdio"
	@echo "data     - download MetroPT-3 and convert to Parquet (~208 MB, not committed)"
	@echo "up       - start the monitoring stack (Kafka, consumer, Prometheus, Grafana, Alertmanager, Postgres)"
	@echo "replay   - stream MetroPT-3 into Kafka"
	@echo "anomaly  - replay with an injected synthetic fault, to demonstrate alerting"
	@echo "logs     - follow the consumer logs"
	@echo "down     - stop the stack"
	@echo ""
	@echo "Grafana http://localhost:3000 | Prometheus http://localhost:9090 | Alertmanager http://localhost:9093"

env:
	@if [ -f .env ]; then echo ".env already exists, leaving it alone"; 	else cp .env.example .env && echo "created .env from .env.example - edit the placeholders"; fi

venv:
	py -3.12 -m venv .venv
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e ".[dev,streaming]"

# Phase 5. Optional: every validation tier passes without it, and the Python
# core stays the default and the oracle. Needs a C++17 compiler -- on Windows the
# Visual Studio Build Tools, which scripts/build_native.py locates itself.
native:
	$(PYTHON) -m pip install -q "pybind11>=2.13" "setuptools>=68"
	$(PYTHON) -m scripts.build_native

test:
	$(PYTHON) -m pytest -v

bench:
	$(PYTHON) -m scripts.benchmark

# Native vs Python: throughput, per-update latency, and cell memory. Runs each
# measurement in its own process so the RSS figures mean something.
bench-native:
	$(PYTHON) -m scripts.benchmark_native

# Replays the real data through the real sketch in-process, so detector
# parameters can be chosen from measured separation in seconds rather than
# from minutes-long Docker round trips. Seeds the Phase 3 evaluation harness.
tune:
	$(PYTHON) -m scripts.tune_detector

# Quantifies cell reclamation: without it the sparse cell dictionary grows
# one entry per distinct cell ever visited, so memory tracks readings seen
# rather than window size.
memcheck:
	$(PYTHON) -m scripts.memory_check

# Is the sketch actually cheaper than storing the window? Depends on dimension.
memcheck-rows:
	$(PYTHON) -m scripts.memory_check --records 150000 --rows-sweep

# Maps the dimension above which sketching beats storing the window - the
# applicability boundary that is Phase 3's main reportable finding.
memcheck-crossover:
	$(PYTHON) -m scripts.memory_check --records 120000 --crossover

# Full-dataset evaluation against the four documented failures.
evaluate:
	$(PYTHON) -m scripts.run_evaluation --method swakde
	$(PYTHON) -m scripts.run_evaluation --method race
	$(PYTHON) -m scripts.run_evaluation --method exact
	$(PYTHON) -m scripts.compare_methods
	$(PYTHON) -m scripts.horizon_sensitivity
	$(PYTHON) -m scripts.diagnose_failures

# Figures for the evaluation report, plus a CSV beside each so the chart is
# never the only way to read a value.
figures:
	$(PYTHON) -m scripts.make_plots

# Experiment tracking: one run per (method, alarm budget), figures as artifacts.
mlflow:
	$(PYTHON) -m scripts.log_to_mlflow

# Phase 4: load detected episodes into Postgres, then exercise the MCP tools.
alerts:
	docker compose up -d postgres
	$(LOAD_ENV) $(PYTHON) -m scripts.load_alerts

mcp-verify:
	$(LOAD_ENV) $(PYTHON) -m scripts.verify_mcp

mcp-serve:
	$(LOAD_ENV) $(PYTHON) -m mcp_server.server

data:
	$(PYTHON) -m scripts.download_data

up:
	docker compose up -d --build
	@echo ""
	@echo "Grafana    http://localhost:3000  (anonymous access enabled)"
	@echo "Prometheus http://localhost:9090"
	@echo "Alerts     http://localhost:9093"
	@echo ""
	@echo "Then: make replay   (or 'make anomaly' to demonstrate the alert path)"

replay:
	docker compose --profile replay run --rm producer \
		python -m streaming.producer --bootstrap kafka:9094

# Injects a synthetic fault, then waits for the alert to fire. Timings are set
# by the pipeline's own stages, not picked arbitrarily:
#   readings     0- 2000  feature standardizer warmup
#                2000- 3600  sliding window filling (density not yet comparable)
#                3600- 5600  scorer accumulating its baseline
#                5600-14000  baseline of genuine normal operation
#               14000-26000  injected fault
# Replay runs at 60 readings/s so the elevated-score span (~3600 readings, i.e.
# until the injected regime refills the window) lasts ~60s of wall clock,
# comfortably longer than the alert's 30s `for`. Takes roughly 7 minutes.
anomaly:
	docker compose --profile replay run --rm producer \
		python -m streaming.producer --bootstrap kafka:9094 \
		--speedup 60 --max-records 26000 \
		--inject-anomaly-at 14000 --inject-duration 12000 --inject-scale 4

logs:
	docker compose logs -f consumer

down:
	docker compose down

clean:
	rm -rf .pytest_cache sketch.egg-info
	find . -type d -name __pycache__ -not -path "./.venv/*" -exec rm -rf {} +
