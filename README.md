# Prediction Market Integrity Dashboard (Autoencoder-Based)

Integrity-first dashboard for anomaly detection in prediction-market trading behavior.

This project intentionally excludes copy-trading/tailing/follow signals, trade recommendations, and automated trading features.

## Stack
- Backend: Python 3.11, FastAPI, SQLite, SQLAlchemy
- ML: PyTorch autoencoder, pandas/numpy, scikit-learn
- Frontend: Next.js App Router + TypeScript

## Repo Layout
- `backend/app`: API, DB, connectors, feature engineering, ML, pipeline job
- `backend/artifacts`: saved models/scalers by window and version
- `backend/tests`: unit tests
- `frontend/app`: leaderboard and entity pages
- `data`: sample JSON for markets/trades
- `docs`: ethics and data dictionary

## Quickstart (Live Polymarket Default)
1. Create and use project virtualenv at repo root:
```bash
python3 -m venv .venv
source .venv/bin/activate
```
2. Install backend + frontend dependencies:
```bash
make install-backend
make install-frontend
```
3. Run live pipeline (writes to one DB file):
```bash
export INTEGRITY_DB_PATH="$(pwd)/integrity.db"
make pipeline
```
4. Start backend and frontend:
```bash
make backend-dev
make frontend-dev
```
5. Open `http://localhost:3000`.

## One-command reliability checks
From repo root, after `make pipeline` and with backend running:
```bash
curl -s "http://127.0.0.1:8000/api/debug/db_counts?window=30d"
curl -s "http://127.0.0.1:8000/api/entities?window=30d&min_trades=0&sort=score_desc"
```
Expected:
- `db_counts` shows non-zero `trades_count`, `entities_count`, and typically non-zero `models_count`/`scores_count`.
- `/api/entities` returns a non-empty JSON array.

## Live Polymarket ingestion: setup + run
Environment variables:
- `INTEGRITY_DB_PATH` (recommended): absolute path to SQLite file used by both pipeline and API.
- `POLYMARKET_GAMMA_BASE_URL` (optional): defaults to `https://gamma-api.polymarket.com`.
- `POLYMARKET_CLOB_BASE_URL` (optional): defaults to `https://clob.polymarket.com`.
- `POLYMARKET_API_KEY` (optional): only used if your CLOB route requires it.

Run a capped live ingestion/training pass:
```bash
export INTEGRITY_DB_PATH="$(pwd)/integrity.db"
make pipeline-live
```

Then start API and verify:
```bash
make backend-dev
curl -s "http://127.0.0.1:8000/api/debug/db_counts?window=30d"
curl -s "http://127.0.0.1:8000/api/entities?window=30d&min_trades=0&sort=score_desc"
```

Notes on live connector behavior:
- Markets are fetched from Gamma markets API.
- Trades are attempted via CLOB `/data/trades` first; if unavailable/auth-gated, the connector falls back to public Data API `/trades`.
- Entity identity uses Polymarket trade field `proxyWallet` when present; otherwise `entity_id` is set to `"unknown"`.
- Timestamps are normalized to tz-naive UTC before DB insert.
- Trade dedupe is idempotent via `(platform, trade_id)` unique constraint.
- Leaderboard `market` column shows top-2 market titles by entity notional volume.

If Polymarket public APIs do NOT expose user-level trader identity, this project falls back to `entity_id="unknown"` and entity-level anomaly quality is limited. Next step in that case is authenticated ingestion with account-level permissions (or an internal identity mapping layer).

## API Endpoints
- `GET /api/entities?platform=&window=30d&min_trades=10&sort=score_desc&market_filter=all`
- `GET /api/entities/{entity_id}?window=30d`
- `GET /api/models/latest?window=30d`
- `GET /api/debug/db_counts?window=30d`

Leaderboard enhancements:
- `market_filter` supports `all`, `sports`, `non_sports`.
- `market` column uses lightweight NLP summarization: top market category + extracted keywords from top-2 markets by entity volume.
- `current_price` column shows latest observed trade price for the entity's top-volume market.
- `kalshi_price` column now uses explicit mapping table (`market_mappings`) from Polymarket market_id -> Kalshi ticker.
- `kalshi_american_odds` converts Kalshi implied probability to American odds.
- `/api/entities` returns only rows with mapped/non-null Kalshi prices.
- Frontend leaderboard auto-refreshes every ~2 seconds.

Live price updates without retraining:
- One-shot recent trade sync: `make price-sync`
- Continuous sync loop (every 10s): `make price-sync-loop`
- This updates `trades`/`entities`; `current_price` in leaderboard updates on the next frontend poll.

Market mapping APIs (for Kalshi price matching):
- List mappings:
```bash
curl -s "http://127.0.0.1:8000/api/mappings?source_platform=polymarket&target_platform=kalshi"
```
- Suggest mappings by title similarity (preview only):
```bash
curl -s "http://127.0.0.1:8000/api/mappings/suggest?limit=50&min_similarity=0.55&auto_apply=false"
```
- Suggest + auto-apply mappings:
```bash
curl -s "http://127.0.0.1:8000/api/mappings/suggest?limit=50&min_similarity=0.58&auto_apply=true"
```
- Upsert mapping:
```bash
curl -s -X POST "http://127.0.0.1:8000/api/mappings" \
  -H "Content-Type: application/json" \
  -d '{
    "source_platform":"polymarket",
    "source_market_id":"<polymarket_condition_id>",
    "target_platform":"kalshi",
    "target_market_id":"<kalshi_ticker>",
    "confidence":1.0,
    "method":"manual",
    "notes":"manual mapping"
  }'
```

## Sample Mode (Explicit Only)
`sample` mode reads from:
- `data/sample_markets.json`
- `data/sample_trades.json`

No external keys are required.

Run sample explicitly:
```bash
make pipeline-sample
```

## Pipeline behavior
Command:
```bash
python backend/app/jobs/run_pipeline.py --mode live --platform polymarket --window 30d
```
Steps:
1. ingest live markets/trades
2. upsert entities
3. compute entity window features
4. train AE (with early stopping)
5. score entities by reconstruction error
6. percentile-calibrate anomaly score to 0-100
7. generate per-entity explanations
8. persist model metadata and artifacts

Artifacts written to:
- `backend/artifacts/{window}/{model_version}/model.pt`
- `backend/artifacts/{window}/{model_version}/scaler.pkl`
- `backend/artifacts/{window}/{model_version}/imputer.pkl`
- `backend/artifacts/{window}/{model_version}/feature_columns.json`
- `backend/artifacts/{window}/{model_version}/training_metadata.json`

## Ethics Notice
The UI contains a persistent warning:
"Anomaly detection for integrity research. Scores are not proof of wrongdoing."
