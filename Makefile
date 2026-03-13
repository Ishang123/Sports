.PHONY: backend-dev frontend-dev dev pipeline pipeline-live pipeline-sample backtest price-sync price-sync-loop test setup-venv install-backend install-frontend

ROOT := $(abspath .)
PYTHON := $(ROOT)/.venv/bin/python
PIP := $(ROOT)/.venv/bin/pip
INTEGRITY_DB_PATH ?= $(ROOT)/integrity.db

setup-venv:
	python3 -m venv .venv

install-backend: setup-venv
	"$(PIP)" install --upgrade pip
	"$(PIP)" install -r backend/requirements.txt

install-frontend:
	cd frontend && npm install

backend-dev:
	INTEGRITY_DB_PATH="$(INTEGRITY_DB_PATH)" "$(PYTHON)" -m uvicorn app.main:app --reload --port 8000 --app-dir backend

frontend-dev:
	cd frontend && npm run dev

pipeline:
	INTEGRITY_DB_PATH="$(INTEGRITY_DB_PATH)" "$(PYTHON)" backend/app/jobs/run_pipeline.py --mode live --platform kalshi --window 30d --backfill-buffer-days 7 --markets-limit 1000 --trades-limit 2000

pipeline-live:
	INTEGRITY_DB_PATH="$(INTEGRITY_DB_PATH)" "$(PYTHON)" backend/app/jobs/run_pipeline.py --mode live --platform kalshi --window 30d --backfill-buffer-days 7 --markets-limit 1000 --trades-limit 2000

pipeline-sample:
	INTEGRITY_DB_PATH="$(INTEGRITY_DB_PATH)" "$(PYTHON)" backend/app/jobs/run_pipeline.py --mode sample --platform polymarket_like --window 30d

backtest:
	INTEGRITY_DB_PATH="$(INTEGRITY_DB_PATH)" "$(PYTHON)" backend/app/jobs/run_backtest.py --platform kalshi --window 90d --top-n 20 --target-bets 5000 --cv-folds 5 --random-trials 500

price-sync:
	INTEGRITY_DB_PATH="$(INTEGRITY_DB_PATH)" "$(PYTHON)" backend/app/jobs/sync_prices.py --lookback-minutes 30 --trades-limit 2000

price-sync-loop:
	INTEGRITY_DB_PATH="$(INTEGRITY_DB_PATH)" "$(PYTHON)" backend/app/jobs/sync_prices.py --lookback-minutes 30 --trades-limit 2000 --interval-seconds 10

test:
	INTEGRITY_DB_PATH="$(INTEGRITY_DB_PATH)" "$(PYTHON)" -m pytest backend/tests -q

dev:
	@echo "Run in separate terminals:"
	@echo "  make backend-dev"
	@echo "  make frontend-dev"
