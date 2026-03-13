from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import delete
from sqlalchemy.orm import Session

from app.models_db import EntityWindowFeature, Market, Trade


FEATURE_COLUMNS = [
    "num_trades",
    "num_markets",
    "total_notional_usd",
    "avg_notional_usd",
    "median_notional_usd",
    "max_notional_usd",
    "size_cv",
    "trades_within_1h_of_close_frac",
    "trades_within_6h_of_close_frac",
    "median_time_to_close_at_trade",
    "burst_10m_max",
    "burst_1h_max",
    "inter_trade_time_median",
    "inter_trade_time_p90",
    "top1_market_volume_share",
    "top3_market_volume_share",
    "market_hhi",
    "side_entropy",
    "side_imbalance",
    "resolved_trade_count",
    "short_horizon_win_rate",
    "overall_win_rate",
]


def parse_window(window: str) -> timedelta:
    if not window.endswith("d"):
        raise ValueError(f"Unsupported window format: {window}")
    days = int(window[:-1])
    return timedelta(days=days)


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def _gini(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    sorted_vals = np.sort(values)
    n = len(sorted_vals)
    cum = np.cumsum(sorted_vals)
    if cum[-1] == 0:
        return 0.0
    return float((n + 1 - 2 * np.sum(cum) / cum[-1]) / n)


def _entropy(p: np.ndarray) -> float:
    p = p[p > 0]
    if len(p) == 0:
        return 0.0
    return float(-(p * np.log2(p)).sum())


def _burst_max(ts_series: pd.Series, freq: str) -> int:
    if ts_series.empty:
        return 0
    s = pd.Series(1, index=pd.to_datetime(ts_series)).sort_index()
    return int(s.rolling(freq).sum().max())


def _inter_trade_stats(ts_series: pd.Series) -> tuple[float, float]:
    if len(ts_series) < 2:
        return 0.0, 0.0
    deltas = pd.to_datetime(ts_series).sort_values().diff().dt.total_seconds().dropna() / 60.0
    return float(deltas.median()), float(deltas.quantile(0.9))


def build_entity_window_features(session: Session, window: str, as_of_ts: datetime | None = None) -> pd.DataFrame:
    as_of_ts = as_of_ts or datetime.now(timezone.utc)
    as_of_pd = pd.to_datetime(as_of_ts, utc=True)
    start_pd = as_of_pd - parse_window(window)

    # Pull then filter in pandas to avoid SQLite timezone edge-cases.
    trades = session.query(Trade).all()
    if not trades:
        return pd.DataFrame(columns=["entity_id", "platform", *FEATURE_COLUMNS])

    markets = session.query(Market).all()
    market_df = pd.DataFrame(
        [
            {
                "market_id": m.market_id,
                "platform": m.platform,
                "close_ts": m.close_ts,
                "outcome": m.outcome,
            }
            for m in markets
        ]
    )
    if market_df.empty:
        market_df = pd.DataFrame(columns=["market_id", "platform", "close_ts", "outcome"])

    trades_df = pd.DataFrame(
        [
            {
                "trade_id": t.trade_id,
                "platform": t.platform,
                "market_id": t.market_id,
                "entity_id": t.entity_id,
                "ts": t.ts,
                "side": t.side.lower(),
                "price": t.price,
                "quantity": t.quantity,
                "notional_usd": t.notional_usd,
            }
            for t in trades
        ]
    )
    trades_df["ts"] = pd.to_datetime(trades_df["ts"], utc=True, errors="coerce")
    windowed_trades = trades_df[(trades_df["ts"] >= start_pd) & (trades_df["ts"] <= as_of_pd)]
    # Safety fallback for sample-mode robustness if window filtering yields nothing.
    trades_df = windowed_trades if not windowed_trades.empty else trades_df

    df = trades_df.merge(market_df, on=["market_id", "platform"], how="left")
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    df["close_ts"] = pd.to_datetime(df["close_ts"], utc=True, errors="coerce")

    df["time_to_close_m"] = (df["close_ts"] - df["ts"]).dt.total_seconds() / 60
    df["within_1h"] = (df["time_to_close_m"] >= 0) & (df["time_to_close_m"] <= 60)
    df["within_6h"] = (df["time_to_close_m"] >= 0) & (df["time_to_close_m"] <= 360)

    rows: list[dict[str, Any]] = []

    for (platform, entity_id), grp in df.groupby(["platform", "entity_id"]):
        notionals = grp["notional_usd"].astype(float).to_numpy()
        market_vol = grp.groupby("market_id")["notional_usd"].sum().sort_values(ascending=False)
        shares = market_vol / market_vol.sum() if market_vol.sum() else market_vol
        side_counts = grp["side"].value_counts(normalize=True)

        resolved = grp[grp["outcome"].notna()].copy()
        if not resolved.empty:
            resolved["is_win"] = (
                ((resolved["side"] == "yes") & (resolved["outcome"].str.lower() == "yes"))
                | ((resolved["side"] == "no") & (resolved["outcome"].str.lower() == "no"))
            ).astype(float)
            resolved["is_short"] = resolved["time_to_close_m"] <= (6 * 60)
            short = resolved[resolved["is_short"]]
            short_wr = float(short["is_win"].mean()) if not short.empty else np.nan
            overall_wr = float(resolved["is_win"].mean())
            resolved_count = int(len(resolved))
        else:
            short_wr = np.nan
            overall_wr = np.nan
            resolved_count = 0

        yes_frac = float(side_counts.get("yes", 0.0))
        inter_median, inter_p90 = _inter_trade_stats(grp["ts"])

        row = {
            "entity_id": entity_id,
            "platform": platform,
            "num_trades": int(len(grp)),
            "num_markets": int(grp["market_id"].nunique()),
            "total_notional_usd": float(notionals.sum()),
            "avg_notional_usd": float(np.mean(notionals)),
            "median_notional_usd": float(np.median(notionals)),
            "max_notional_usd": float(np.max(notionals)),
            "size_cv": _safe_div(float(np.std(notionals)), float(np.mean(notionals))),
            "size_gini": _gini(notionals),
            "trades_within_1h_of_close_frac": float(grp["within_1h"].mean()),
            "trades_within_6h_of_close_frac": float(grp["within_6h"].mean()),
            "median_time_to_close_at_trade": float(np.nanmedian(grp["time_to_close_m"].to_numpy())),
            "burst_10m_max": _burst_max(grp["ts"], "10min"),
            "burst_1h_max": _burst_max(grp["ts"], "1h"),
            "inter_trade_time_median": inter_median,
            "inter_trade_time_p90": inter_p90,
            "top1_market_volume_share": float(shares.iloc[0]) if len(shares) else 0.0,
            "top3_market_volume_share": float(shares.head(3).sum()) if len(shares) else 0.0,
            "market_hhi": float((shares**2).sum()) if len(shares) else 0.0,
            "side_entropy": _entropy(side_counts.to_numpy()),
            "side_imbalance": abs(yes_frac - 0.5),
            "resolved_trade_count": resolved_count,
            "short_horizon_win_rate": short_wr,
            "overall_win_rate": overall_wr,
        }
        rows.append(row)

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    session.execute(
        delete(EntityWindowFeature).where(
            EntityWindowFeature.window == window,
            EntityWindowFeature.as_of_ts == as_of_ts,
        )
    )

    persist_cols = ["entity_id", "platform", *FEATURE_COLUMNS]
    for _, row in out[persist_cols].iterrows():
        session.add(
            EntityWindowFeature(
                entity_id=row["entity_id"],
                platform=row["platform"],
                window=window,
                as_of_ts=as_of_ts,
                feature_json=json.dumps({k: (None if pd.isna(v) else float(v)) for k, v in row.items() if k not in {"entity_id", "platform"}}),
                num_trades=int(row["num_trades"]),
                total_notional_usd=float(row["total_notional_usd"]),
            )
        )

    return out[persist_cols]
