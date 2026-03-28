"""
Polymarket public API client for Wallet Discovery.
Uses Data API + Gamma Markets API — no auth required.
"""
from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any

import requests

GAMMA_BASE = "https://gamma-api.polymarket.com"
DATA_BASE = "https://data-api.polymarket.com"

_session = requests.Session()
_session.headers.update({"User-Agent": "pm-integrity/1.0"})

_WALLET_CACHE: dict[str, Any] = {"ts": 0.0, "wallets": []}
_MARKET_CACHE: dict[str, Any] = {"ts": 0.0, "markets": []}


def _get(url: str, params: dict | None = None, timeout: float = 12.0) -> Any:
    resp = _session.get(url, params=params, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# Active markets
# ---------------------------------------------------------------------------

def get_active_markets(limit: int = 30, cache_ttl: float = 30.0) -> list[dict]:
    now = time.time()
    if now - float(_MARKET_CACHE["ts"]) < cache_ttl and _MARKET_CACHE["markets"]:
        return _MARKET_CACHE["markets"][:limit]
    try:
        rows = _get(
            f"{GAMMA_BASE}/markets",
            params={"limit": min(limit * 2, 100), "closed": "false", "order": "volume24hr", "ascending": "false"},
        )
    except Exception:
        return _MARKET_CACHE.get("markets", [])[:limit]
    if not isinstance(rows, list):
        return []
    out: list[dict] = []
    for r in rows:
        tokens = r.get("tokens") or []
        yes_price: float | None = None
        if isinstance(tokens, list):
            for tok in tokens:
                if isinstance(tok, dict) and str(tok.get("outcome", "")).lower() in ("yes", "1"):
                    try:
                        yes_price = float(tok["price"])
                    except Exception:
                        pass
                    break
            if yes_price is None and tokens and isinstance(tokens[0], dict):
                try:
                    yes_price = float(tokens[0].get("price") or 0)
                except Exception:
                    pass
        out.append({
            "market_id": str(r.get("conditionId") or r.get("id") or ""),
            "title": str(r.get("question") or r.get("title") or ""),
            "category": str(r.get("category") or ""),
            "volume_24h": float(r.get("volume24hr") or r.get("volume24h") or 0),
            "volume_total": float(r.get("volume") or 0),
            "liquidity": float(r.get("liquidity") or 0),
            "yes_price": yes_price,
            "end_date": str(r.get("endDate") or ""),
        })
    _MARKET_CACHE["ts"] = now
    _MARKET_CACHE["markets"] = out
    return out[:limit]


# ---------------------------------------------------------------------------
# Wallet leaderboard
# ---------------------------------------------------------------------------

def get_top_wallets(limit: int = 50, cache_ttl: float = 60.0) -> list[dict]:
    now = time.time()
    if now - float(_WALLET_CACHE["ts"]) < cache_ttl and _WALLET_CACHE["wallets"]:
        return _WALLET_CACHE["wallets"][:limit]

    wallets = _fetch_leaderboard(limit) or _aggregate_from_trades(limit)

    _WALLET_CACHE["ts"] = now
    _WALLET_CACHE["wallets"] = wallets
    return wallets[:limit]


def _fetch_leaderboard(limit: int) -> list[dict]:
    for url, params in [
        (f"{DATA_BASE}/leaderboard", {"limit": limit}),
        (f"{GAMMA_BASE}/leaderboard", {"limit": limit}),
    ]:
        try:
            data = _get(url, params=params)
            rows: list[Any] = []
            if isinstance(data, list):
                rows = data
            elif isinstance(data, dict):
                for key in ("data", "results", "leaderboard", "users", "accounts"):
                    if isinstance(data.get(key), list):
                        rows = data[key]
                        break
            if rows:
                return [r for r in (_normalize_leaderboard_row(x) for x in rows[:limit]) if r]
        except Exception:
            continue
    return []


def _normalize_leaderboard_row(r: dict) -> dict | None:
    addr = str(
        r.get("address") or r.get("user") or r.get("proxyWallet") or r.get("wallet") or ""
    ).strip()
    if not addr:
        return None
    return {
        "address": addr,
        "profit_loss": float(r.get("profitLoss") or r.get("profit") or r.get("pnl") or 0),
        "roi": float(r.get("roi") or r.get("ROI") or 0),
        "volume": float(r.get("volume") or r.get("totalVolume") or 0),
        "num_trades": int(r.get("tradesCount") or r.get("numTrades") or r.get("trades") or 0),
        "win_rate": float(r.get("winRate") or 0),
    }


def _aggregate_from_trades(limit: int) -> list[dict]:
    try:
        rows = _get(f"{DATA_BASE}/trades", params={"limit": 1000, "takerOnly": "false"})
    except Exception:
        return []
    if not isinstance(rows, list):
        return []

    wallets: dict[str, dict] = {}
    for r in rows:
        addr = str(r.get("proxyWallet") or "").strip()
        if not addr or addr == "unknown":
            continue
        price = float(r.get("price") or 0)
        size = float(r.get("size") or 0)
        notional = price * size
        side = str(r.get("side") or "").upper()
        market_id = str(r.get("conditionId") or "")

        if addr not in wallets:
            wallets[addr] = {
                "address": addr, "volume": 0.0, "num_trades": 0,
                "profit_loss": 0.0, "roi": 0.0, "win_rate": 0.0,
                "_buy": 0.0, "_sell": 0.0,
                "_mkts": {},
            }
        w = wallets[addr]
        w["volume"] += notional
        w["num_trades"] += 1
        if market_id not in w["_mkts"]:
            w["_mkts"][market_id] = {"buy": 0.0, "sell": 0.0}
        if side in ("BUY", "YES"):
            w["_buy"] += notional
            w["_mkts"][market_id]["buy"] += notional
        else:
            w["_sell"] += notional
            w["_mkts"][market_id]["sell"] += notional

    for w in wallets.values():
        realized = w["_sell"] - w["_buy"]
        w["profit_loss"] = round(realized, 2)
        w["roi"] = round(realized / w["_buy"], 4) if w["_buy"] > 0 else 0.0
        closed = [m for m in w["_mkts"].values() if m["sell"] > 0]
        w["win_rate"] = round(
            len([m for m in closed if m["sell"] > m["buy"]]) / len(closed), 4
        ) if closed else 0.0
        del w["_buy"], w["_sell"], w["_mkts"]

    return sorted(wallets.values(), key=lambda w: w["volume"], reverse=True)[:limit]


# ---------------------------------------------------------------------------
# Wallet profile (detail view)
# ---------------------------------------------------------------------------

def get_wallet_profile(address: str) -> dict:
    profile: dict[str, Any] = {
        "address": address,
        "profit_loss": 0.0, "roi": 0.0, "volume": 0.0,
        "num_trades": 0, "win_rate": 0.0, "positions_count": 0,
        "positions": [], "recent_trades": [],
    }

    # Portfolio stats
    try:
        p = _get(f"{DATA_BASE}/portfolio", params={"user": address})
        if isinstance(p, dict):
            profile["profit_loss"] = float(p.get("profitLoss") or 0)
            profile["roi"] = float(p.get("roi") or 0)
            profile["volume"] = float(p.get("volume") or 0)
            profile["num_trades"] = int(p.get("tradesCount") or 0)
            profile["win_rate"] = float(p.get("winRate") or 0)
            profile["positions_count"] = int(p.get("positionsCount") or 0)
    except Exception:
        pass

    # Open positions
    try:
        pos = _get(f"{DATA_BASE}/positions", params={"user": address, "sizeThreshold": "0.01"})
        if isinstance(pos, list):
            profile["positions"] = [
                {
                    "market_id": str(p.get("conditionId") or ""),
                    "market_title": str(p.get("title") or p.get("market") or ""),
                    "outcome": str(p.get("outcome") or "Yes"),
                    "size": float(p.get("size") or 0),
                    "avg_price": float(p.get("avgPrice") or 0),
                    "pnl": float(p.get("pnl") or p.get("unrealizedPnl") or 0),
                }
                for p in pos[:20]
                if float(p.get("size") or 0) > 0.001
            ]
    except Exception:
        pass

    # Recent activity
    try:
        acts = _get(f"{DATA_BASE}/activity", params={"user": address, "limit": 50})
        if isinstance(acts, list):
            profile["recent_trades"] = [
                {
                    "market_id": str(a.get("conditionId") or ""),
                    "market_title": str(a.get("title") or ""),
                    "side": str(a.get("side") or ""),
                    "price": float(a.get("price") or 0),
                    "size": float(a.get("size") or 0),
                    "notional_usd": float(a.get("price") or 0) * float(a.get("size") or 0),
                    "ts": str(a.get("timestamp") or ""),
                }
                for a in acts[:50]
            ]
    except Exception:
        pass

    return profile


# ---------------------------------------------------------------------------
# Alert polling — recent global trades filtered for tagged wallets
# ---------------------------------------------------------------------------

def get_recent_trades_for_addresses(addresses: list[str], since_ts: float = 0.0) -> list[dict]:
    if not addresses:
        return []
    addr_set = {a.lower() for a in addresses}
    try:
        rows = _get(f"{DATA_BASE}/trades", params={"limit": 500, "takerOnly": "false"})
    except Exception:
        return []
    if not isinstance(rows, list):
        return []

    out: list[dict] = []
    for r in rows:
        addr = str(r.get("proxyWallet") or "").strip()
        if addr.lower() not in addr_set:
            continue
        try:
            ts = float(r.get("timestamp") or 0)
            if ts > 1e12:
                ts /= 1000.0
        except Exception:
            ts = 0.0
        if since_ts and ts and ts <= since_ts:
            continue
        out.append({
            "address": addr,
            "market_id": str(r.get("conditionId") or ""),
            "market_title": str(r.get("title") or ""),
            "side": str(r.get("side") or ""),
            "price": float(r.get("price") or 0),
            "size": float(r.get("size") or 0),
            "notional_usd": float(r.get("price") or 0) * float(r.get("size") or 0),
            "ts": ts,
            "ts_iso": datetime.fromtimestamp(ts, tz=timezone.utc).isoformat() if ts else "",
        })
    return out
