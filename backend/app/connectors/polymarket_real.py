from __future__ import annotations

import hashlib
import json
import random
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import requests

from app.connectors.base import BaseConnector, TimeRange


def _iso_z(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_any_ts(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        v = float(value)
        if v > 1e12:
            v = v / 1000.0
        return datetime.fromtimestamp(v, tz=timezone.utc)
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        try:
            if s.endswith("Z"):
                return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(timezone.utc)
            return datetime.fromisoformat(s).astimezone(timezone.utc)
        except ValueError:
            try:
                v = float(s)
                if v > 1e12:
                    v = v / 1000.0
                return datetime.fromtimestamp(v, tz=timezone.utc)
            except ValueError:
                return None
    return None


@dataclass
class RetryConfig:
    max_retries: int = 5
    base_sleep_s: float = 0.5
    timeout_s: float = 20.0


class PolymarketRealConnector(BaseConnector):
    platform = "polymarket"

    def __init__(
        self,
        gamma_base_url: str = "https://gamma-api.polymarket.com",
        clob_url: str = "https://clob.polymarket.com",
        data_api_url: str = "https://data-api.polymarket.com",
        api_key: str | None = None,
    ):
        self.gamma_base_url = gamma_base_url.rstrip("/")
        self.clob_url = clob_url.rstrip("/")
        self.data_api_url = data_api_url.rstrip("/")
        self.api_key = api_key
        self.retry = RetryConfig()

    @staticmethod
    def _normalize_outcome(value: Any) -> str | None:
        if value is None:
            return None
        s = str(value).strip().lower()
        if s in {"yes", "true", "1", "1.0"}:
            return "yes"
        if s in {"no", "false", "0", "0.0"}:
            return "no"
        return None

    @staticmethod
    def _coerce_list(value: Any) -> list[Any]:
        if value is None:
            return []
        if isinstance(value, list):
            return value
        if isinstance(value, str):
            s = value.strip()
            if not s:
                return []
            try:
                parsed = json.loads(s)
                return parsed if isinstance(parsed, list) else []
            except Exception:
                return []
        return []

    @staticmethod
    def _is_resolved_market(row: dict[str, Any]) -> bool:
        status = str(row.get("umaResolutionStatus") or row.get("resolutionStatus") or "").strip().lower()
        if status in {"resolved", "finalized", "settled"}:
            return True
        # Gamma commonly marks closed markets with outcome data after settlement.
        closed = row.get("closed")
        return bool(closed)

    def _extract_market_outcome(self, row: dict[str, Any]) -> str | None:
        # Direct fields first.
        for key in ("outcome", "result", "marketOutcome", "resolvedOutcome", "winningOutcome"):
            out = self._normalize_outcome(row.get(key))
            if out:
                return out

        # Winner booleans / indexes when present.
        for idx_key in ("winningOutcomeIndex", "winnerIndex", "resolvedOutcomeIndex", "outcomeIndex"):
            raw_idx = row.get(idx_key)
            try:
                idx = int(raw_idx)
            except (TypeError, ValueError):
                continue

            outcomes = self._coerce_list(row.get("outcomes"))
            if outcomes and 0 <= idx < len(outcomes):
                out = self._normalize_outcome(outcomes[idx])
                if out:
                    return out

            if idx in {0, 1}:
                # Most binary markets encode first/second outcome.
                return "yes" if idx == 0 else "no"

        # Token-level winner info.
        tokens = self._coerce_list(row.get("tokens"))
        if tokens:
            for tok in tokens:
                if not isinstance(tok, dict):
                    continue
                if tok.get("winner") is True:
                    out = self._normalize_outcome(tok.get("outcome"))
                    if out:
                        return out

            # Fallback: settled winner token often has final price 1.0.
            for tok in tokens:
                if not isinstance(tok, dict):
                    continue
                try:
                    p = float(tok.get("price"))
                except (TypeError, ValueError):
                    p = None
                if p is not None and p >= 0.999:
                    out = self._normalize_outcome(tok.get("outcome"))
                    if out:
                        return out

        # Outcome labels + final prices.
        outcomes = self._coerce_list(row.get("outcomes"))
        prices = self._coerce_list(row.get("outcomePrices"))
        if len(outcomes) == 2 and len(prices) == 2:
            try:
                p0 = float(prices[0])
                p1 = float(prices[1])
            except (TypeError, ValueError):
                p0 = p1 = None
            if p0 is not None and p1 is not None:
                winner_label = outcomes[0] if p0 >= p1 else outcomes[1]
                out = self._normalize_outcome(winner_label)
                if out:
                    return out

        return None

    def _request_json(self, url: str, params: dict[str, Any] | None = None, headers: dict[str, str] | None = None) -> Any:
        headers = headers or {}
        for attempt in range(self.retry.max_retries + 1):
            try:
                resp = requests.get(url, params=params, headers=headers, timeout=self.retry.timeout_s)
                if resp.status_code in (429, 500, 502, 503, 504):
                    if attempt < self.retry.max_retries:
                        sleep_s = self.retry.base_sleep_s * (2**attempt) + random.uniform(0, 0.2)
                        time.sleep(sleep_s)
                        continue
                resp.raise_for_status()
                return resp.json()
            except requests.RequestException:
                if attempt < self.retry.max_retries:
                    sleep_s = self.retry.base_sleep_s * (2**attempt) + random.uniform(0, 0.2)
                    time.sleep(sleep_s)
                    continue
                raise
        raise RuntimeError("unreachable")

    def fetch_markets(self, time_range: TimeRange, limit: int = 1000) -> list[dict]:
        out: list[dict] = []
        offset = 0
        page_size = min(500, max(1, limit))

        while len(out) < limit:
            params = {"limit": page_size, "offset": offset, "order": "id", "ascending": "false"}
            rows = self._request_json(f"{self.gamma_base_url}/markets", params=params)
            if not isinstance(rows, list) or not rows:
                break

            for row in rows:
                market_id = str(row.get("conditionId") or row.get("id") or "")
                if not market_id:
                    continue

                open_dt = _parse_any_ts(row.get("startDate") or row.get("createdAt"))
                close_dt = _parse_any_ts(row.get("endDate") or row.get("closedTime") or row.get("umaEndDate"))
                resolve_dt = _parse_any_ts(row.get("resolutionTimestamp") or row.get("resolveDate") or row.get("updatedAt"))

                cmp_dt = close_dt or open_dt or resolve_dt
                if cmp_dt and not (time_range.start <= cmp_dt <= time_range.end):
                    continue

                extracted_outcome = self._extract_market_outcome(row)
                if extracted_outcome is None and self._is_resolved_market(row):
                    # Some Gamma list responses omit winner fields; detail endpoint is richer.
                    try:
                        detail = self._request_json(f"{self.gamma_base_url}/markets/{market_id}")
                        if isinstance(detail, dict):
                            extracted_outcome = self._extract_market_outcome(detail)
                            if extracted_outcome is not None:
                                row = detail
                    except Exception:
                        # Keep list-row data if detail fetch fails.
                        pass

                out.append(
                    {
                        "market_id": market_id,
                        "title": str(row.get("question") or row.get("title") or market_id),
                        "category": row.get("category"),
                        "open_ts": _iso_z(open_dt) if open_dt else None,
                        "close_ts": _iso_z(close_dt) if close_dt else None,
                        "resolve_ts": _iso_z(resolve_dt) if resolve_dt else None,
                        "outcome": extracted_outcome,
                        "metadata": row,
                    }
                )
                if len(out) >= limit:
                    break

            offset += len(rows)
            if len(rows) < page_size:
                break

        return out

    def _fetch_trades_clob(self, time_range: TimeRange, market_ids: list[str] | None = None, limit: int | None = None) -> list[dict]:
        # Prefer CLOB ledger endpoint, but this usually requires L2 auth headers.
        # We try once; on auth failure caller will use Data API fallback.
        params: dict[str, Any] = {
            "after": int(time_range.start.timestamp()),
            "before": int(time_range.end.timestamp()),
        }
        if market_ids:
            params["market"] = market_ids[0]
        if limit:
            params["limit"] = min(limit, 1000)

        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        rows = self._request_json(f"{self.clob_url}/data/trades", params=params, headers=headers)
        if not isinstance(rows, list):
            return []
        return rows

    @staticmethod
    def _stable_trade_id(trade: dict[str, Any], idx: int) -> str:
        if trade.get("id"):
            return str(trade["id"])
        if trade.get("transactionHash"):
            parts = [
                str(trade.get("transactionHash")),
                str(trade.get("conditionId", "")),
                str(trade.get("proxyWallet", "")),
                str(trade.get("timestamp", "")),
                str(trade.get("side", "")),
                str(trade.get("size", "")),
                str(trade.get("price", "")),
                str(idx),
            ]
            return ":".join(parts)
        payload = json.dumps(trade, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _fetch_trades_data_api(
        self,
        time_range: TimeRange,
        market_ids: list[str] | None = None,
        limit: int | None = None,
    ) -> list[dict]:
        out: list[dict] = []
        offset = 0
        max_rows = limit if limit is not None else 10_000
        page_size = min(500, max_rows)
        market_param: str | None = None
        if market_ids:
            # Avoid very long URLs causing 400s on data-api.
            capped = market_ids[:20]
            joined = ",".join(capped)
            if len(joined) <= 1500:
                market_param = joined

        while len(out) < max_rows:
            params: dict[str, Any] = {"limit": page_size, "offset": offset, "takerOnly": "false"}
            if market_param:
                params["market"] = market_param
            try:
                rows = self._request_json(f"{self.data_api_url}/trades", params=params)
            except requests.HTTPError as e:
                # If market filter request is rejected, retry without market param.
                if e.response is not None and e.response.status_code == 400 and "market" in params:
                    params.pop("market", None)
                    try:
                        rows = self._request_json(f"{self.data_api_url}/trades", params=params)
                    except requests.HTTPError as e2:
                        # Some data-api deployments reject deep offsets; stop safely with collected rows.
                        if e2.response is not None and e2.response.status_code == 400:
                            break
                        raise
                elif e.response is not None and e.response.status_code == 400:
                    # Deep offset or unsupported query shape; stop safely with collected rows.
                    break
                else:
                    raise
            if not isinstance(rows, list) or not rows:
                break

            for row in rows:
                ts = _parse_any_ts(row.get("timestamp"))
                if ts is None:
                    continue
                if not (time_range.start <= ts <= time_range.end):
                    continue

                side_raw = str(row.get("side", "")).lower()
                if side_raw in {"buy", "yes"}:
                    side = "yes"
                elif side_raw in {"sell", "no"}:
                    side = "no"
                else:
                    side = side_raw or "unknown"

                price = float(row.get("price") or 0.0)
                qty = float(row.get("size") or 0.0)
                notional = price * qty

                # Data API exposes proxyWallet for user-level identity.
                entity_id = str(row.get("proxyWallet") or "").strip() or "unknown"
                market_id = str(row.get("conditionId") or "")
                if not market_id:
                    continue

                normalized = {
                    "trade_id": self._stable_trade_id(row, idx=offset + len(out)),
                    "market_id": market_id,
                    "entity_id": entity_id,
                    "ts": _iso_z(ts),
                    "side": side,
                    "price": price,
                    "quantity": qty,
                    "notional_usd": notional,
                    "raw_json": row,
                }
                out.append(normalized)
                if len(out) >= max_rows:
                    break

            offset += len(rows)
            if len(rows) < page_size:
                break

        return out

    def fetch_trades(
        self,
        time_range: TimeRange,
        market_ids: list[str] | None = None,
        limit: int | None = None,
    ) -> list[dict]:
        try:
            clob_rows = self._fetch_trades_clob(time_range, market_ids=market_ids, limit=limit)
        except requests.HTTPError:
            clob_rows = []
        except Exception:
            clob_rows = []

        if clob_rows:
            out: list[dict] = []
            for idx, row in enumerate(clob_rows):
                ts = _parse_any_ts(row.get("match_time") or row.get("last_update"))
                if ts is None:
                    continue
                side_raw = str(row.get("side", "")).lower()
                side = "yes" if side_raw in {"buy", "yes"} else "no" if side_raw in {"sell", "no"} else side_raw
                price = float(row.get("price") or 0.0)
                qty = float(row.get("size") or 0.0)
                notional = price * qty
                entity_id = str(row.get("maker_address") or "").strip() or "unknown"
                market_id = str(row.get("market") or "")
                if not market_id:
                    continue
                out.append(
                    {
                        "trade_id": str(row.get("id") or f"clob:{market_id}:{int(ts.timestamp())}:{idx}"),
                        "market_id": market_id,
                        "entity_id": entity_id,
                        "ts": _iso_z(ts),
                        "side": side,
                        "price": price,
                        "quantity": qty,
                        "notional_usd": notional,
                        "raw_json": row,
                    }
                )
            return out

        return self._fetch_trades_data_api(time_range, market_ids=market_ids, limit=limit)
