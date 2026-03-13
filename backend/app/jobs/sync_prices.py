from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from sqlalchemy import func

ROOT = Path(__file__).resolve().parents[3]
BACKEND_ROOT = ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.connectors.base import TimeRange
from app.connectors.polymarket_real import PolymarketRealConnector
from app.db import get_session, init_db, resolve_db_path
from app.jobs.run_pipeline import upsert_entity_from_trades, upsert_trade
from app.models_db import Entity, Trade


def sync_once(lookback_minutes: int, trades_limit: int) -> None:
    init_db()
    connector = PolymarketRealConnector(
        gamma_base_url=os.getenv(
            "POLYMARKET_GAMMA_BASE_URL",
            os.getenv("POLYMARKET_BASE_URL", "https://gamma-api.polymarket.com"),
        ),
        clob_url=os.getenv(
            "POLYMARKET_CLOB_BASE_URL",
            os.getenv("POLYMARKET_CLOB_URL", "https://clob.polymarket.com"),
        ),
        api_key=os.getenv("POLYMARKET_API_KEY"),
    )
    end = datetime.now(timezone.utc)
    start = end - timedelta(minutes=lookback_minutes)
    tr = TimeRange(start=start, end=end)

    trades = connector.fetch_trades(tr, market_ids=None, limit=trades_limit)
    with get_session() as session:
        for t in trades:
            upsert_trade(session, connector.platform, t)
        upsert_entity_from_trades(session)
        session.commit()

        trades_count = session.query(func.count(Trade.id)).scalar() or 0
        entities_count = session.query(func.count(Entity.id)).scalar() or 0
        print(
            f"Price sync complete | fetched={len(trades)} "
            f"trades_in_db={trades_count} entities_in_db={entities_count}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lookback-minutes", type=int, default=30)
    parser.add_argument("--trades-limit", type=int, default=2000)
    parser.add_argument("--interval-seconds", type=int, default=0)
    args = parser.parse_args()

    os.chdir(ROOT)
    print(f"Price sync using SQLite DB at: {resolve_db_path()}")

    if args.interval_seconds <= 0:
        sync_once(args.lookback_minutes, args.trades_limit)
        return

    while True:
        try:
            sync_once(args.lookback_minutes, args.trades_limit)
        except Exception as e:
            print(f"Price sync error: {e}")
        time.sleep(args.interval_seconds)


if __name__ == "__main__":
    main()
