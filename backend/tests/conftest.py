from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.connectors.base import TimeRange
from app.connectors.polymarket_like import PolymarketLikeConnector
from app.db import configure_engine, get_session, init_db
from app.jobs.run_pipeline import upsert_entity_from_trades, upsert_market, upsert_trade


@pytest.fixture()
def seeded_db(tmp_path: Path):
    db_path = tmp_path / "test.db"
    configure_engine(str(db_path))
    init_db()

    root = Path(__file__).resolve().parents[2]
    connector = PolymarketLikeConnector(
        markets_path=root / "data" / "sample_markets.json",
        trades_path=root / "data" / "sample_trades.json",
    )

    now = datetime.now(timezone.utc)
    tr = TimeRange(start=now - timedelta(days=180), end=now)
    markets = connector.fetch_markets(tr, limit=5000)
    trades = connector.fetch_trades(tr, market_ids=[m["market_id"] for m in markets], limit=None)

    with get_session() as session:
        for m in markets:
            upsert_market(session, connector.platform, m)
        for t in trades:
            upsert_trade(session, connector.platform, t)
        upsert_entity_from_trades(session)

    return {"db_path": db_path}
