from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sqlalchemy import desc
from sqlalchemy.orm import Session

from app.models_db import EntityScore, Market, Trade


@dataclass
class BacktestResult:
    metrics: dict[str, Any]
    top_accounts: list[str]


def _trade_cost_and_pnl(side: str, price: float, quantity: float, outcome: str) -> tuple[float, float, int]:
    side_l = side.lower()
    out_l = outcome.lower()
    yes_price = float(price)
    no_price = 1.0 - yes_price

    if side_l == "yes":
        cost = quantity * yes_price
        pnl = quantity * (1.0 - yes_price) if out_l == "yes" else -cost
        hit = int(out_l == "yes")
    elif side_l == "no":
        cost = quantity * no_price
        pnl = quantity * yes_price if out_l == "no" else -cost
        hit = int(out_l == "no")
    else:
        return 0.0, 0.0, 0
    return float(cost), float(pnl), hit


def _safe_roi(pnl: float, cost: float) -> float:
    return float(pnl / cost) if cost > 0 else 0.0


def _safe_sharpe(returns: np.ndarray) -> float:
    if returns.size < 2:
        return 0.0
    std = float(np.std(returns, ddof=1))
    if std == 0:
        return 0.0
    mean = float(np.mean(returns))
    return float((mean / std) * math.sqrt(returns.size))


def _crowd_baseline(eval_df: pd.DataFrame) -> dict[str, float]:
    if eval_df.empty:
        return {"roi": 0.0, "hit_rate": 0.0}

    rows = []
    for market_id, grp in eval_df.groupby("market_id"):
        outcome = str(grp["outcome"].iloc[0]).lower()
        yes_vol = float(grp.loc[grp["side"] == "yes", "notional_usd"].sum())
        no_vol = float(grp.loc[grp["side"] == "no", "notional_usd"].sum())
        pick = "yes" if yes_vol >= no_vol else "no"

        yes_price = float(grp["price"].mean())
        no_price = 1.0 - yes_price
        if pick == "yes":
            cost = yes_price
            pnl = (1.0 - yes_price) if outcome == "yes" else -yes_price
            hit = int(outcome == "yes")
        else:
            cost = no_price
            pnl = yes_price if outcome == "no" else -no_price
            hit = int(outcome == "no")
        rows.append((market_id, cost, pnl, hit))

    base = pd.DataFrame(rows, columns=["market_id", "cost", "pnl", "hit"])
    roi = _safe_roi(float(base["pnl"].sum()), float(base["cost"].sum()))
    hit_rate = float(base["hit"].mean()) if not base.empty else 0.0
    return {"roi": roi, "hit_rate": hit_rate}


def run_backtest(
    session: Session,
    platform: str = "kalshi",
    window: str = "30d",
    top_n: int = 20,
    target_bets: int = 5000,
    cv_folds: int = 5,
    random_trials: int = 200,
    seed: int = 42,
) -> BacktestResult:
    trades = (
        session.query(
            Trade.entity_id,
            Trade.market_id,
            Trade.ts,
            Trade.side,
            Trade.price,
            Trade.quantity,
            Trade.notional_usd,
            Trade.raw_json,
        )
        .filter(Trade.platform == platform)
        .all()
    )

    if not trades:
        return BacktestResult(metrics={"error": "no_trades"}, top_accounts=[])

    market_rows = (
        session.query(Market.market_id, Market.title, Market.outcome, Market.metadata_json)
        .filter(Market.platform == platform)
        .all()
    )
    market_title_map = {str(m_id): str(title or m_id) for m_id, title, _, _ in market_rows}
    market_outcome_map = {}
    for m_id, _, outcome, metadata_json in market_rows:
        normalized = _normalize_outcome(outcome)
        if normalized is None:
            normalized = _extract_outcome_from_raw(metadata_json)
        market_outcome_map[str(m_id)] = normalized

    df = pd.DataFrame(
        [
            {
                "entity_id": t.entity_id,
                "market_id": t.market_id,
                "ts": t.ts,
                "side": t.side,
                "price": float(t.price),
                "quantity": float(t.quantity),
                "notional_usd": float(t.notional_usd),
                "raw_json": t.raw_json,
            }
            for t in trades
        ]
    )

    # Use market outcomes stored in raw JSON where available.
    df["outcome"] = df["raw_json"].apply(_extract_outcome_from_raw)
    df["outcome"] = df["outcome"].fillna(df["market_id"].map(market_outcome_map))
    df = df[df["outcome"].isin(["yes", "no"])].copy()
    if df.empty:
        unresolved_sample = int(len(trades))
        return BacktestResult(
            metrics={
                "error": "no_resolved_trades",
                "sample_size_total_bets": unresolved_sample,
                "framework": _framework_definitions(),
            },
            top_accounts=[],
        )

    df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
    df = df.dropna(subset=["ts"]).sort_values("ts")

    # Use latest model score for ranking if available.
    scored_rows = (
        session.query(EntityScore.entity_id, EntityScore.anomaly_score_0_100)
        .filter(EntityScore.platform == platform, EntityScore.window == window)
        .order_by(desc(EntityScore.created_ts))
        .all()
    )
    score_map: dict[str, float] = {}
    for entity_id, anomaly in scored_rows:
        if entity_id not in score_map:
            # Convert anomaly into "sharpness-like" signal (higher is better).
            score_map[entity_id] = 100.0 - float(anomaly)

    if not score_map:
        # Fallback ranking by historical ROI if model scores missing.
        tmp = _account_level_metrics(df)
        score_map = {k: float(v) for k, v in tmp["roi"].to_dict().items()}

    account_rank = sorted(score_map.items(), key=lambda x: x[1], reverse=True)
    selected_accounts = _select_accounts_for_target_bets(
        eval_df=df,
        ranked_accounts=[a for a, _ in account_rank],
        min_top_n=max(1, top_n),
        target_bets=max(1, target_bets),
    )

    eval_df = df.copy()
    top_df = eval_df[eval_df["entity_id"].isin(selected_accounts)].copy()
    if top_df.empty:
        return BacktestResult(metrics={"error": "no_trades_for_selected_accounts"}, top_accounts=selected_accounts)

    top_df[["cost", "pnl", "hit"]] = top_df.apply(
        lambda r: pd.Series(_trade_cost_and_pnl(str(r["side"]), float(r["price"]), float(r["quantity"]), str(r["outcome"]))),
        axis=1,
    )
    top_stats = _strategy_stats(top_df)
    top_roi = float(top_stats["cumulative_roi"])
    top_hit_rate = float(top_stats["hit_rate"])
    top_sharpe = float(top_stats["sharpe_ratio"])

    # Random baseline with same number of accounts.
    rng = np.random.default_rng(seed)
    all_accounts = np.array(sorted(eval_df["entity_id"].dropna().unique()))
    pick_n = min(len(selected_accounts), len(all_accounts))
    random_rois = []
    random_hits = []
    for _ in range(max(1, random_trials)):
        sampled = rng.choice(all_accounts, size=pick_n, replace=False)
        samp_df = eval_df[eval_df["entity_id"].isin(sampled)].copy()
        if samp_df.empty:
            continue
        samp_df[["cost", "pnl", "hit"]] = samp_df.apply(
            lambda r: pd.Series(_trade_cost_and_pnl(str(r["side"]), float(r["price"]), float(r["quantity"]), str(r["outcome"]))),
            axis=1,
        )
        random_rois.append(_safe_roi(float(samp_df["pnl"].sum()), float(samp_df["cost"].sum())))
        random_hits.append(float(samp_df["hit"].mean()) if not samp_df.empty else 0.0)
    random_roi = float(np.mean(random_rois)) if random_rois else 0.0
    random_hit = float(np.mean(random_hits)) if random_hits else 0.0

    crowd = _crowd_baseline(eval_df)

    # Account-level predictive metrics (profitability classification).
    acct = _account_level_metrics(eval_df)
    acct["score"] = acct.index.to_series().map(score_map)
    acct = acct.dropna(subset=["score"]).copy()
    acct["is_profitable"] = (acct["roi"] > 0).astype(int)

    if acct["is_profitable"].nunique() >= 2:
        auc = float(roc_auc_score(acct["is_profitable"], acct["score"]))
    else:
        auc = float("nan")

    k = min(max(1, top_n), len(acct))
    predicted_positive = set(acct.sort_values("score", ascending=False).head(k).index.tolist())
    actual_positive = set(acct[acct["is_profitable"] == 1].index.tolist())
    true_pos = len(predicted_positive & actual_positive)
    precision = float(true_pos / len(predicted_positive)) if predicted_positive else 0.0
    recall = float(true_pos / len(actual_positive)) if actual_positive else 0.0

    market_avg_hit_rate = float(
        eval_df.apply(
            lambda r: 1.0 if (str(r["side"]).lower() == str(r["outcome"]).lower()) else 0.0,
            axis=1,
        ).mean()
    )

    # Out-of-sample validation by single holdout split.
    split_ts = eval_df["ts"].quantile(0.8)
    train_df = eval_df[eval_df["ts"] <= split_ts].copy()
    test_df = eval_df[eval_df["ts"] > split_ts].copy()
    oos_holdout = _out_of_sample_metrics(
        train_df=train_df,
        test_df=test_df,
        score_map=score_map,
        top_n=max(1, len(selected_accounts)),
    )

    # Time-series cross-validation (expanding window).
    cv = _time_series_cv_metrics(
        eval_df=eval_df,
        score_map=score_map,
        min_top_n=max(1, top_n),
        target_bets=max(1, target_bets),
        cv_folds=max(2, cv_folds),
    )

    top_market_rows = (
        top_df.groupby("market_id", as_index=False)["notional_usd"]
        .sum()
        .sort_values("notional_usd", ascending=False)
        .head(10)
    )
    top_markets_readable = [
        {
            "market_id": str(r["market_id"]),
            "title": market_title_map.get(str(r["market_id"]), str(r["market_id"])),
            "label": _market_nlp_label(market_title_map.get(str(r["market_id"]), str(r["market_id"]))),
            "notional_usd": float(r["notional_usd"]),
        }
        for _, r in top_market_rows.iterrows()
    ]

    metrics = {
        "platform": platform,
        "window": window,
        "num_markets": int(eval_df["market_id"].nunique()),
        "num_resolved_trades": int(len(eval_df)),
        "num_accounts_scored": int(len(account_rank)),
        "top_n_accounts": int(len(selected_accounts)),
        "target_sample_size_bets": int(target_bets),
        "strategy": {
            **top_stats,
        },
        "baseline_random_accounts": {
            "cumulative_roi": random_roi,
            "hit_rate": random_hit,
            "alpha_vs_random_roi": top_roi - random_roi,
        },
        "baseline_crowd_consensus": {
            "cumulative_roi": float(crowd["roi"]),
            "hit_rate": float(crowd["hit_rate"]),
            "alpha_vs_crowd_roi": top_roi - float(crowd["roi"]),
        },
        "signal_quality": {
            "top_hit_rate": top_hit_rate,
            "market_avg_hit_rate": market_avg_hit_rate,
            "hit_rate_uplift": top_hit_rate - market_avg_hit_rate,
        },
        "model_metrics": {
            "auc_profitability": auc,
            "precision_top_n_profitable": precision,
            "recall_top_n_profitable": recall,
        },
        "out_of_sample_validation": oos_holdout,
        "cross_validation": cv,
        "top_markets_readable": top_markets_readable,
        "framework": _framework_definitions(),
        "notes": {
            "sharpness_score_definition": "100 - anomaly_score_0_100 from latest entity_scores rows",
            "evaluation_mode": "in-sample on resolved trades available in DB",
        },
    }
    return BacktestResult(metrics=metrics, top_accounts=selected_accounts)


def _account_level_metrics(eval_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for entity_id, grp in eval_df.groupby("entity_id"):
        costs = []
        pnls = []
        for _, r in grp.iterrows():
            cost, pnl, _ = _trade_cost_and_pnl(str(r["side"]), float(r["price"]), float(r["quantity"]), str(r["outcome"]))
            costs.append(cost)
            pnls.append(pnl)
        total_cost = float(np.sum(costs))
        total_pnl = float(np.sum(pnls))
        rows.append({"entity_id": entity_id, "pnl": total_pnl, "cost": total_cost, "roi": _safe_roi(total_pnl, total_cost)})
    return pd.DataFrame(rows).set_index("entity_id")


def _extract_outcome_from_raw(raw_json: Any) -> str | None:
    if raw_json is None:
        return None
    if isinstance(raw_json, dict):
        payload = raw_json
    else:
        try:
            import json

            payload = json.loads(raw_json)
        except Exception:
            return None

    for key in ("outcome", "result", "market_outcome"):
        val = payload.get(key)
        if val is None:
            continue
        s = str(val).strip().lower()
        if s in {"yes", "no"}:
            return s
        if s in {"true", "1"}:
            return "yes"
        if s in {"false", "0"}:
            return "no"
    return None


def _normalize_outcome(value: Any) -> str | None:
    if value is None:
        return None
    s = str(value).strip().lower()
    if s in {"yes", "no"}:
        return s
    if s in {"true", "1", "1.0"}:
        return "yes"
    if s in {"false", "0", "0.0"}:
        return "no"
    return None


def _strategy_stats(df: pd.DataFrame) -> dict[str, float]:
    if df.empty:
        return {
            "cumulative_roi": 0.0,
            "hit_rate": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "sample_size_bets": 0.0,
            "total_pnl_usd": 0.0,
            "total_stake_usd": 0.0,
        }
    total_cost = float(df["cost"].sum())
    total_pnl = float(df["pnl"].sum())
    ret = (df["pnl"] / df["cost"].replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
    max_dd = _max_drawdown(df)
    return {
        "cumulative_roi": _safe_roi(total_pnl, total_cost),
        "hit_rate": float(df["hit"].mean()),
        "sharpe_ratio": _safe_sharpe(ret),
        "max_drawdown": float(max_dd),
        "sample_size_bets": float(len(df)),
        "total_pnl_usd": total_pnl,
        "total_stake_usd": total_cost,
    }


def _max_drawdown(df: pd.DataFrame) -> float:
    ordered = df.sort_values("ts")
    if ordered.empty:
        return 0.0
    eq = ordered["pnl"].cumsum().to_numpy(dtype=float)
    running_peak = np.maximum.accumulate(eq)
    dd = eq - running_peak
    return float(np.min(dd)) if dd.size else 0.0


def _out_of_sample_metrics(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    score_map: dict[str, float],
    top_n: int,
) -> dict[str, Any]:
    if train_df.empty or test_df.empty:
        return {"available": False, "reason": "insufficient_time_split_data"}

    train_rank = _account_level_metrics(train_df)
    train_rank["score"] = train_rank.index.to_series().map(score_map)
    train_rank["score"] = train_rank["score"].fillna(train_rank["roi"])
    selected = train_rank.sort_values("score", ascending=False).head(max(1, top_n)).index.tolist()

    oos_df = test_df[test_df["entity_id"].isin(selected)].copy()
    if oos_df.empty:
        return {"available": False, "reason": "no_oos_trades_for_selected_accounts"}

    oos_df[["cost", "pnl", "hit"]] = oos_df.apply(
        lambda r: pd.Series(_trade_cost_and_pnl(str(r["side"]), float(r["price"]), float(r["quantity"]), str(r["outcome"]))),
        axis=1,
    )
    stats = _strategy_stats(oos_df)
    return {"available": True, **stats}


def _select_accounts_for_target_bets(
    eval_df: pd.DataFrame,
    ranked_accounts: list[str],
    min_top_n: int,
    target_bets: int,
) -> list[str]:
    if not ranked_accounts:
        return []
    trade_count_map = eval_df.groupby("entity_id").size().to_dict()
    selected: list[str] = []
    running = 0
    for acc in ranked_accounts:
        selected.append(acc)
        running += int(trade_count_map.get(acc, 0))
        if len(selected) >= min_top_n and running >= target_bets:
            break
    return selected


def _time_series_cv_metrics(
    eval_df: pd.DataFrame,
    score_map: dict[str, float],
    min_top_n: int,
    target_bets: int,
    cv_folds: int,
) -> dict[str, Any]:
    df = eval_df.sort_values("ts").reset_index(drop=True)
    n = len(df)
    if n < (cv_folds + 1) * 20:
        return {"available": False, "reason": "insufficient_rows_for_cv", "folds_requested": cv_folds, "folds_used": 0}

    fold_edges = np.linspace(0, n, cv_folds + 2, dtype=int)
    fold_results: list[dict[str, Any]] = []

    for i in range(1, cv_folds + 1):
        train_end = fold_edges[i]
        test_end = fold_edges[i + 1]
        train_df = df.iloc[:train_end].copy()
        test_df = df.iloc[train_end:test_end].copy()
        if train_df.empty or test_df.empty:
            continue

        # Rank accounts with train data only; use model score where present, then train ROI fallback.
        train_rank = _account_level_metrics(train_df)
        train_rank["score"] = train_rank.index.to_series().map(score_map)
        train_rank["score"] = train_rank["score"].fillna(train_rank["roi"])
        ranked_accounts = train_rank.sort_values("score", ascending=False).index.tolist()
        selected = _select_accounts_for_target_bets(
            eval_df=train_df,
            ranked_accounts=ranked_accounts,
            min_top_n=min_top_n,
            target_bets=target_bets,
        )
        if not selected:
            continue

        oos_df = test_df[test_df["entity_id"].isin(selected)].copy()
        if oos_df.empty:
            continue

        oos_df[["cost", "pnl", "hit"]] = oos_df.apply(
            lambda r: pd.Series(_trade_cost_and_pnl(str(r["side"]), float(r["price"]), float(r["quantity"]), str(r["outcome"]))),
            axis=1,
        )
        stats = _strategy_stats(oos_df)
        stats["fold"] = i
        stats["accounts_selected"] = len(selected)
        fold_results.append(stats)

    if not fold_results:
        return {"available": False, "reason": "no_valid_cv_folds", "folds_requested": cv_folds, "folds_used": 0}

    frame = pd.DataFrame(fold_results)
    return {
        "available": True,
        "folds_requested": cv_folds,
        "folds_used": int(len(fold_results)),
        "mean_cumulative_roi": float(frame["cumulative_roi"].mean()),
        "mean_sharpe_ratio": float(frame["sharpe_ratio"].mean()),
        "mean_max_drawdown": float(frame["max_drawdown"].mean()),
        "mean_sample_size_bets": float(frame["sample_size_bets"].mean()),
        "folds": fold_results,
    }


def _market_nlp_label(title: str) -> str:
    words = [w.lower() for w in "".join(ch if ch.isalnum() or ch.isspace() else " " for ch in title).split()]
    stop = {"the", "and", "for", "with", "will", "is", "to", "of", "in", "on", "a", "an"}
    keep = [w for w in words if len(w) > 2 and w not in stop]
    if not keep:
        return title[:80]
    top = pd.Series(keep).value_counts().head(4).index.tolist()
    return ", ".join(top)


def _framework_definitions() -> dict[str, str]:
    return {
        "roi": "Cumulative ROI of top-ranked accounts over resolved trades.",
        "sharpe": "Mean trade return divided by std dev of trade return, scaled by sqrt(N).",
        "max_drawdown": "Worst peak-to-trough cumulative PnL decline.",
        "sample_size": "Number of bets/trades included in evaluation.",
        "out_of_sample_validation": "Train on earlier 80% time slice, evaluate selected accounts on final 20%.",
        "cross_validation": "Expanding-window time-series CV across multiple chronological folds.",
    }
