from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from dateutil.parser import isoparse

from app.connectors.base import BaseConnector, TimeRange


class PolymarketLikeConnector(BaseConnector):
    platform = "polymarket_like"

    def __init__(self, markets_path: Path, trades_path: Path):
        self.markets_path = markets_path
        self.trades_path = trades_path

    def _load_json(self, path: Path) -> list[dict]:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _in_range(self, ts: datetime, time_range: TimeRange) -> bool:
        return time_range.start <= ts <= time_range.end

    def fetch_markets(self, time_range: TimeRange, limit: int = 1000) -> list[dict]:
        markets = self._load_json(self.markets_path)
        filtered = []
        for market in markets:
            close_ts = isoparse(market["close_ts"])
            if self._in_range(close_ts, time_range):
                filtered.append(market)
            if len(filtered) >= limit:
                break
        return filtered

    def fetch_trades(
        self,
        time_range: TimeRange,
        market_ids: list[str] | None = None,
        limit: int | None = None,
    ) -> list[dict]:
        trades = self._load_json(self.trades_path)
        market_set = set(market_ids) if market_ids else None
        filtered = []
        for trade in trades:
            ts = isoparse(trade["ts"])
            if not self._in_range(ts, time_range):
                continue
            if market_set and trade["market_id"] not in market_set:
                continue
            filtered.append(trade)
            if limit is not None and len(filtered) >= limit:
                break
        return filtered
