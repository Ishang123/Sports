from __future__ import annotations

import random
import time
from datetime import datetime, timezone
from typing import Any

import requests

from app.connectors.base import BaseConnector, TimeRange


class KalshiConnector(BaseConnector):
    platform = "kalshi"
    def __init__(self, base_url: str = "https://api.elections.kalshi.com/trade-api/v2"):
        primary = base_url.rstrip("/")
        fallback = "https://api.kalshi.com/trade-api/v2"
        self.base_urls = [primary] if primary == fallback else [primary, fallback]

    def _request_json(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        last_err: Exception | None = None
        for base_url in self.base_urls:
            url = f"{base_url}{path}"
            for attempt in range(5):
                try:
                    resp = requests.get(url, params=params, timeout=20)
                    if resp.status_code in (429, 500, 502, 503, 504):
                        time.sleep((2**attempt) * 0.25 + random.uniform(0, 0.2))
                        continue
                    resp.raise_for_status()
                    data = resp.json()
                    return data if isinstance(data, dict) else {}
                except requests.RequestException as e:
                    last_err = e
                    time.sleep((2**attempt) * 0.25 + random.uniform(0, 0.2))
        if last_err:
            raise last_err
        return {}

    def _parse_ts(self, s: Any) -> datetime | None:
        if s is None:
            return None
        try:
            if isinstance(s, (int, float)):
                # Handle both epoch seconds and epoch milliseconds.
                ts = float(s)
                if ts > 1e12:
                    ts = ts / 1000.0
                return datetime.fromtimestamp(ts, tz=timezone.utc)
            s_str = str(s)
            if s_str.endswith("Z"):
                return datetime.fromisoformat(s_str.replace("Z", "+00:00")).astimezone(timezone.utc)
            return datetime.fromisoformat(s_str).astimezone(timezone.utc)
        except Exception:
            return None

    def list_raw_markets(self, limit: int = 2000, status: str = "open") -> list[dict]:
        if status == "":
            # Prioritize resolved/settled markets first so backtests have outcomes.
            merged: list[dict] = []
            seen: set[str] = set()
            for st in ("settled", "resolved", "closed", "open"):
                try:
                    rows = self._list_raw_markets_single(limit=limit, status=st)
                except Exception:
                    continue
                for row in rows:
                    ticker = str(row.get("ticker") or "").strip()
                    if not ticker or ticker in seen:
                        continue
                    seen.add(ticker)
                    merged.append(row)
                    if len(merged) >= limit:
                        return merged
            # Final fallback to API default behavior.
            try:
                rows = self._list_raw_markets_single(limit=limit, status="")
                for row in rows:
                    ticker = str(row.get("ticker") or "").strip()
                    if not ticker or ticker in seen:
                        continue
                    seen.add(ticker)
                    merged.append(row)
                    if len(merged) >= limit:
                        break
            except Exception:
                pass
            return merged
        return self._list_raw_markets_single(limit=limit, status=status)

    def _list_raw_markets_single(self, limit: int = 2000, status: str = "open") -> list[dict]:
        out: list[dict] = []
        cursor: str | None = None
        while len(out) < limit:
            params: dict[str, Any] = {"limit": min(1000, limit - len(out))}
            if cursor:
                params["cursor"] = cursor
            if status != "":
                params["status"] = status
            payload = self._request_json("/markets", params=params)
            rows = payload.get("markets", [])
            if not isinstance(rows, list) or not rows:
                break
            out.extend(rows)
            cursor = payload.get("cursor")
            if not cursor:
                break
        return out

    def fetch_markets(self, time_range: TimeRange, limit: int = 1000) -> list[dict]:
        rows = self.list_raw_markets(limit=limit, status="")
        out: list[dict] = []
        for row in rows:
            ticker = str(row.get("ticker") or "").strip()
            if not ticker:
                continue
            open_dt = self._parse_ts(row.get("open_time"))
            close_dt = self._parse_ts(row.get("close_time") or row.get("expiration_time"))
            resolve_dt = self._parse_ts(row.get("result_updated_time") or row.get("expiration_time"))
            cmp_dt = close_dt or open_dt or resolve_dt
            if cmp_dt and not (time_range.start <= cmp_dt <= time_range.end):
                continue
            outcome = self._extract_market_outcome(row)
            out.append(
                {
                    "market_id": ticker,
                    "title": str(row.get("title") or ticker),
                    "category": row.get("category"),
                    "open_ts": open_dt.isoformat().replace("+00:00", "Z") if open_dt else None,
                    "close_ts": close_dt.isoformat().replace("+00:00", "Z") if close_dt else None,
                    "resolve_ts": resolve_dt.isoformat().replace("+00:00", "Z") if resolve_dt else None,
                    "outcome": outcome,
                    "metadata": row,
                }
            )
            if len(out) >= limit:
                break
        return out

    def fetch_trades(
        self,
        time_range: TimeRange,
        market_ids: list[str] | None = None,
        limit: int | None = None,
    ) -> list[dict]:
        max_rows = limit or 5000
        out: list[dict] = []
        tickers = [str(x).strip() for x in (market_ids or []) if str(x).strip()]
        if not tickers:
            tickers = [None]  # global feed fallback

        seen_trade_ids: set[str] = set()
        for ticker in tickers:
            if len(out) >= max_rows:
                break
            cursor: str | None = None
            while len(out) < max_rows:
                params: dict[str, Any] = {
                    "limit": min(1000, max_rows - len(out)),
                    "min_ts": int(time_range.start.timestamp()),
                    "max_ts": int(time_range.end.timestamp()),
                }
                if cursor:
                    params["cursor"] = cursor
                if ticker:
                    params["ticker"] = ticker

                payload = self._request_json("/markets/trades", params=params)
                rows = payload.get("trades", [])
                if not isinstance(rows, list) or not rows:
                    break

                for row in rows:
                    ts = self._parse_ts(
                        row.get("created_time")
                        or row.get("createdTime")
                        or row.get("ts")
                        or row.get("timestamp")
                    )
                    if not ts:
                        continue

                    price = (
                        row.get("yes_price_dollars")
                        or row.get("price_dollars")
                        or row.get("yes_price")
                        or row.get("price")
                    )
                    qty = row.get("count_fp") or row.get("count") or row.get("size") or 0
                    try:
                        p = float(price)
                        q = float(qty)
                    except Exception:
                        continue

                    t_id = str(row.get("trade_id") or f"kalshi:{row.get('ticker')}:{int(ts.timestamp())}:{len(out)}")
                    if t_id in seen_trade_ids:
                        continue
                    seen_trade_ids.add(t_id)

                    entity = self._extract_entity_id(row)
                    if entity == "unknown":
                        ticker_val = str(row.get("ticker") or ticker or "").strip()
                        if ticker_val:
                            # Kalshi public trades often omit trader identity; bucket unknown flow by market.
                            entity = f"unknown:{ticker_val}"
                    out.append(
                        {
                            "trade_id": t_id,
                            "market_id": str(row.get("ticker") or ticker or ""),
                            "entity_id": entity,
                            "ts": ts.isoformat().replace("+00:00", "Z"),
                            "side": str(row.get("taker_side") or row.get("side") or "unknown"),
                            "price": p,
                            "quantity": q,
                            "notional_usd": p * q,
                            "raw_json": row,
                        }
                    )
                    if len(out) >= max_rows:
                        break

                cursor = payload.get("cursor")
                if not cursor:
                    break
        return out

    def _extract_entity_id(self, row: dict[str, Any]) -> str:
        candidates = (
            row.get("taker_id"),
            row.get("maker_id"),
            row.get("user_id"),
            row.get("account_id"),
            row.get("wallet"),
            row.get("wallet_address"),
            row.get("participant_id"),
            row.get("client_id"),
        )
        for value in candidates:
            if value is None:
                continue
            s = str(value).strip()
            if s:
                return s
        return "unknown"

    def _extract_market_outcome(self, row: dict[str, Any]) -> str | None:
        candidates = (
            row.get("result"),
            row.get("outcome"),
            row.get("settlement_value"),
            row.get("winning_side"),
            row.get("winner"),
        )
        for value in candidates:
            if value is None:
                continue
            s = str(value).strip().lower()
            if s in {"yes", "no"}:
                return s
            if s in {"true", "1", "1.0"}:
                return "yes"
            if s in {"false", "0", "0.0"}:
                return "no"
        return None
