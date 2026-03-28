from __future__ import annotations

import json
import logging
import re
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from sqlalchemy import desc, func, select

from app.db import get_session, init_db, resolve_db_path
from app.models_db import (
    Entity,
    EntityScore,
    EntityWindowFeature,
    Market,
    ModelRegistry,
    Trade,
    TrackedWallet,
    WalletAlert,
)
from app.schemas import EntityDetail, EntityLeaderboardRow, ModelLatest

app = FastAPI(title="Prediction Market Integrity Dashboard API")
logger = logging.getLogger(__name__)

FEATURE_LABELS: dict[str, str] = {
    "num_trades": "Trade count",
    "num_markets": "Market breadth",
    "total_notional_usd": "Total traded volume",
    "avg_notional_usd": "Average trade size",
    "median_notional_usd": "Median trade size",
    "max_notional_usd": "Largest trade size",
    "size_cv": "Trade size variability",
    "trades_within_1h_of_close_frac": "Late trading within 1h of close",
    "trades_within_6h_of_close_frac": "Late trading within 6h of close",
    "median_time_to_close_at_trade": "Typical time to market close",
    "burst_10m_max": "10-minute burstiness",
    "burst_1h_max": "1-hour burstiness",
    "inter_trade_time_median": "Median gap between trades",
    "inter_trade_time_p90": "Long-tail gap between trades",
    "top1_market_volume_share": "Concentration in top market",
    "top3_market_volume_share": "Concentration in top 3 markets",
    "market_hhi": "Market concentration (HHI)",
    "side_entropy": "Direction diversity",
    "side_imbalance": "Directional imbalance",
    "resolved_trade_count": "Resolved trade coverage",
    "short_horizon_win_rate": "Short-horizon win rate",
    "overall_win_rate": "Overall win rate",
}

_STATIC_DIR = Path(__file__).resolve().parents[2] / "frontend"
_ALERT_LAST_FETCH: float = 0.0


@app.on_event("startup")
def on_startup() -> None:
    init_db()
    logger.warning("Integrity API using SQLite DB at: %s", resolve_db_path())


# ---------------------------------------------------------------------------
# Health / debug
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> dict:
    return {"ok": True}


@app.get("/api/debug/db_counts")
def debug_db_counts(window: str = Query(default="30d")) -> dict:
    with get_session() as session:
        latest_model = (
            session.query(ModelRegistry)
            .filter(ModelRegistry.window == window)
            .order_by(desc(ModelRegistry.created_ts))
            .first()
        )
        latest_version = latest_model.model_version if latest_model else None
        min_trade_ts = session.query(func.min(Trade.ts)).scalar()
        max_trade_ts = session.query(func.max(Trade.ts)).scalar()
        result = {
            "db_path": resolve_db_path(),
            "window": window,
            "latest_model_version": latest_version,
            "trades_count": session.query(func.count(Trade.id)).scalar() or 0,
            "entities_count": session.query(func.count(Entity.id)).scalar() or 0,
            "features_count": session.query(func.count(EntityWindowFeature.id)).scalar() or 0,
            "markets_count": session.query(func.count(Market.id)).scalar() or 0,
            "scores_count": session.query(func.count(EntityScore.id)).scalar() or 0,
            "min_trade_ts": min_trade_ts.isoformat() if min_trade_ts else None,
            "max_trade_ts": max_trade_ts.isoformat() if max_trade_ts else None,
        }
        return result


# ---------------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------------

@app.get("/api/backtest/latest")
def backtest_latest() -> dict:
    root = Path(__file__).resolve().parents[2]
    candidates = [
        root / "backend" / "artifacts" / "backtest_latest.json",
        root / "artifacts" / "backtest_latest.json",
    ]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        raise HTTPException(status_code=404, detail="Backtest report not found. Run make backtest first.")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to parse backtest report: {exc}") from exc


# ---------------------------------------------------------------------------
# Sharp Trades — entity leaderboard
# ---------------------------------------------------------------------------

@app.get("/api/entities", response_model=list[EntityLeaderboardRow])
def list_entities(
    platform: str | None = Query(default=None),
    window: str = Query(default="30d"),
    min_trades: int = Query(default=10),
    sort: str = Query(default="score_desc"),
    market_filter: str = Query(default="all"),
):
    with get_session() as session:
        latest_model_version = _latest_model_version_for_platform(session, window=window, platform=platform)
        if not latest_model_version:
            return _fallback_entities(session, window=window, platform=platform,
                                      min_trades=min_trades, sort=sort, market_filter=market_filter)

        q = (
            session.query(EntityScore, Entity, EntityWindowFeature)
            .join(Entity, (Entity.entity_id == EntityScore.entity_id) & (Entity.platform == EntityScore.platform))
            .outerjoin(
                EntityWindowFeature,
                (EntityWindowFeature.entity_id == EntityScore.entity_id)
                & (EntityWindowFeature.platform == EntityScore.platform)
                & (EntityWindowFeature.window == EntityScore.window),
            )
            .filter(EntityScore.model_version == latest_model_version, EntityScore.window == window)
        )
        if platform:
            q = q.filter(EntityScore.platform == platform)
        if min_trades > 0:
            q = q.filter(func.coalesce(EntityWindowFeature.num_trades, 0) >= min_trades)
        if sort == "score_desc":
            q = q.order_by(desc(EntityScore.anomaly_score_0_100))

        dedup: dict[tuple[str, str], tuple] = {}
        for score, entity, feature in q.all():
            k = (entity.platform, entity.entity_id)
            prev = dedup.get(k)
            if prev is None:
                dedup[k] = (score, entity, feature)
                continue
            prev_ts = prev[2].as_of_ts if prev[2] is not None else None
            curr_ts = feature.as_of_ts if feature is not None else None
            if prev_ts is None or (curr_ts is not None and curr_ts > prev_ts):
                dedup[k] = (score, entity, feature)

        rows = []
        for score, entity, feature in dedup.values():
            market_profile = _entity_market_profile(session, entity.platform, entity.entity_id)
            if not _market_filter_pass(market_profile["is_sports"], market_filter):
                continue
            raw_explanations = json.loads(score.top_explanations_json)[:3]
            current_price = _current_price_for_entity(
                session, entity.platform, entity.entity_id, market_profile.get("top_market_id")
            )
            rows.append(
                EntityLeaderboardRow(
                    entity_id=entity.entity_id,
                    platform=entity.platform,
                    anomaly_score_0_100=score.anomaly_score_0_100,
                    market=market_profile["label"],
                    current_price=current_price,
                    current_american_odds=_prob_to_american_odds(current_price),
                    top_explanations=[_nlp_reason_text(x) for x in raw_explanations],
                    num_trades=(feature.num_trades if feature else 0),
                    total_notional_usd=(feature.total_notional_usd if feature else 0.0),
                    first_seen_ts=entity.first_seen_ts.isoformat(),
                    last_seen_ts=entity.last_seen_ts.isoformat(),
                )
            )
        if sort == "score_desc":
            rows.sort(key=lambda r: r.anomaly_score_0_100, reverse=True)
        if rows:
            return rows
        return _fallback_entities(session, window=window, platform=platform,
                                  min_trades=min_trades, sort=sort, market_filter=market_filter)


@app.get("/api/entities/{entity_id}", response_model=EntityDetail)
def get_entity(entity_id: str, window: str = Query(default="30d"), platform: str | None = Query(default=None)):
    with get_session() as session:
        latest_model_version = _latest_model_version_for_platform(session, window=window, platform=platform)
        if not latest_model_version:
            raise HTTPException(status_code=404, detail="Model not found")

        score_q = session.query(EntityScore).filter(
            EntityScore.entity_id == entity_id,
            EntityScore.window == window,
            EntityScore.model_version == latest_model_version,
        )
        if platform:
            score_q = score_q.filter(EntityScore.platform == platform)
        score = score_q.first()
        if not score:
            raise HTTPException(status_code=404, detail="Entity score not found")

        entity = session.query(Entity).filter(
            Entity.entity_id == entity_id, Entity.platform == score.platform
        ).first()
        feature_row = (
            session.query(EntityWindowFeature)
            .filter(
                EntityWindowFeature.entity_id == entity_id,
                EntityWindowFeature.platform == score.platform,
                EntityWindowFeature.window == window,
            )
            .order_by(desc(EntityWindowFeature.as_of_ts))
            .first()
        )
        trades = (
            session.query(Trade)
            .filter(Trade.entity_id == entity_id, Trade.platform == score.platform)
            .order_by(desc(Trade.ts))
            .limit(50)
            .all()
        )
        top_markets_stmt = (
            select(Trade.market_id, func.sum(Trade.notional_usd).label("market_notional"))
            .where(Trade.entity_id == entity_id, Trade.platform == score.platform)
            .group_by(Trade.market_id)
            .order_by(desc("market_notional"))
            .limit(10)
        )
        top_markets = [
            {"market_id": m_id, "notional_usd": float(notional)}
            for m_id, notional in session.execute(top_markets_stmt).all()
        ]
        return EntityDetail(
            entity_id=entity_id,
            platform=score.platform,
            window=window,
            anomaly_score_0_100=score.anomaly_score_0_100,
            score_raw=score.score_raw,
            top_explanations=[_nlp_reason_text(x) for x in json.loads(score.top_explanations_json)],
            feature_snapshot=json.loads(feature_row.feature_json) if feature_row else {},
            recent_trades=[
                {
                    "trade_id": t.trade_id,
                    "market_id": t.market_id,
                    "ts": t.ts.isoformat(),
                    "side": t.side,
                    "price": t.price,
                    "quantity": t.quantity,
                    "notional_usd": t.notional_usd,
                }
                for t in trades
            ],
            top_markets=top_markets,
        )


@app.get("/api/models/latest", response_model=ModelLatest)
def model_latest(window: str = Query(default="30d")):
    with get_session() as session:
        model = (
            session.query(ModelRegistry)
            .filter(ModelRegistry.window == window)
            .order_by(desc(ModelRegistry.created_ts))
            .first()
        )
        if not model:
            raise HTTPException(status_code=404, detail="Model not found")
        return ModelLatest(
            model_version=model.model_version,
            window=model.window,
            metrics_summary=json.loads(model.metrics_json),
        )


# ---------------------------------------------------------------------------
# Wallet Discovery
# ---------------------------------------------------------------------------

@app.get("/api/wallets/leaderboard")
def wallet_leaderboard(limit: int = Query(default=50, le=200)):
    from app.connectors.polymarket_api import get_top_wallets
    wallets = get_top_wallets(limit=limit)
    with get_session() as session:
        tagged = {w.address for w in session.query(TrackedWallet).all()}
    for w in wallets:
        w["tagged"] = w["address"] in tagged
    return wallets


@app.get("/api/wallets/tagged")
def list_tagged_wallets():
    with get_session() as session:
        rows = session.query(TrackedWallet).order_by(desc(TrackedWallet.tagged_ts)).all()
        return [
            {"address": r.address, "label": r.label, "tagged_ts": r.tagged_ts.isoformat()}
            for r in rows
        ]


@app.post("/api/wallets/{address}/tag")
def tag_wallet(address: str, label: str | None = Query(default=None)):
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    with get_session() as session:
        existing = session.query(TrackedWallet).filter(TrackedWallet.address == address).first()
        if existing is None:
            session.add(TrackedWallet(address=address, label=label, tagged_ts=now))
    return {"ok": True, "address": address}


@app.delete("/api/wallets/{address}/tag")
def untag_wallet(address: str):
    with get_session() as session:
        session.query(TrackedWallet).filter(TrackedWallet.address == address).delete()
    return {"ok": True, "address": address}


@app.get("/api/wallets/{address}")
def get_wallet(address: str):
    from app.connectors.polymarket_api import get_wallet_profile
    profile = get_wallet_profile(address)
    with get_session() as session:
        tagged = session.query(TrackedWallet).filter(TrackedWallet.address == address).first()
        profile["tagged"] = tagged is not None
    return profile


@app.get("/api/alerts")
def get_alerts(limit: int = Query(default=50, le=200)):
    global _ALERT_LAST_FETCH
    with get_session() as session:
        tagged = [w.address for w in session.query(TrackedWallet).all()]

        # Refresh from Polymarket at most every 10 seconds
        if tagged and time.time() - _ALERT_LAST_FETCH > 10.0:
            _ALERT_LAST_FETCH = time.time()
            from app.connectors.polymarket_api import get_recent_trades_for_addresses
            new_trades = get_recent_trades_for_addresses(tagged, since_ts=0.0)
            for t in new_trades:
                if not t.get("ts"):
                    continue
                ts_float = float(t["ts"])
                ts_dt = datetime.fromtimestamp(ts_float, tz=timezone.utc).replace(tzinfo=None)
                try:
                    session.add(WalletAlert(
                        address=t["address"],
                        market_id=t["market_id"],
                        market_title=t.get("market_title") or "",
                        side=t.get("side") or "",
                        price=float(t.get("price") or 0),
                        notional_usd=float(t.get("notional_usd") or 0),
                        trade_ts=ts_float,
                        ts=ts_dt,
                    ))
                    session.flush()
                except Exception:
                    session.rollback()

        alerts = (
            session.query(WalletAlert)
            .filter(WalletAlert.address.in_(tagged) if tagged else False)
            .order_by(desc(WalletAlert.ts))
            .limit(limit)
            .all()
        ) if tagged else []

        return [
            {
                "address": a.address,
                "market_id": a.market_id,
                "market_title": a.market_title,
                "side": a.side,
                "price": a.price,
                "notional_usd": a.notional_usd,
                "ts": a.ts.isoformat(),
            }
            for a in alerts
        ]


@app.get("/api/markets/active")
def get_active_markets_endpoint(limit: int = Query(default=30, le=100)):
    from app.connectors.polymarket_api import get_active_markets
    return get_active_markets(limit=limit)


# ---------------------------------------------------------------------------
# Helper functions — entity leaderboard
# ---------------------------------------------------------------------------

def _fallback_entities(
    session,
    window: str,
    platform: str | None,
    min_trades: int,
    sort: str,
    market_filter: str,
) -> list[EntityLeaderboardRow]:
    entity_count = session.query(func.count(Entity.id)).scalar() or 0
    if entity_count == 0:
        return _fallback_entities_from_trades(
            session=session, platform=platform, min_trades=min_trades,
            sort=sort, market_filter=market_filter,
        )

    entity_q = session.query(Entity)
    if platform:
        entity_q = entity_q.filter(Entity.platform == platform)
    if sort == "score_desc":
        entity_q = entity_q.order_by(desc(Entity.last_seen_ts))

    rows: list[EntityLeaderboardRow] = []
    for entity in entity_q.limit(1000).all():
        feature = (
            session.query(EntityWindowFeature)
            .filter(
                EntityWindowFeature.entity_id == entity.entity_id,
                EntityWindowFeature.platform == entity.platform,
                EntityWindowFeature.window == window,
            )
            .order_by(desc(EntityWindowFeature.as_of_ts))
            .first()
        )
        num_trades = feature.num_trades if feature else 0
        total_notional = feature.total_notional_usd if feature else 0.0
        if num_trades < min_trades:
            continue
        market_profile = _entity_market_profile(session, entity.platform, entity.entity_id)
        if not _market_filter_pass(market_profile["is_sports"], market_filter):
            continue
        current_price = _current_price_for_entity(
            session, entity.platform, entity.entity_id, market_profile.get("top_market_id")
        )
        rows.append(
            EntityLeaderboardRow(
                entity_id=entity.entity_id,
                platform=entity.platform,
                anomaly_score_0_100=0.0,
                market=market_profile["label"],
                current_price=current_price,
                current_american_odds=_prob_to_american_odds(current_price),
                top_explanations=["No model scores available for this window yet."],
                num_trades=num_trades,
                total_notional_usd=total_notional,
                first_seen_ts=entity.first_seen_ts.isoformat(),
                last_seen_ts=entity.last_seen_ts.isoformat(),
            )
        )
    return rows


def _fallback_entities_from_trades(
    session,
    platform: str | None,
    min_trades: int,
    sort: str,
    market_filter: str,
) -> list[EntityLeaderboardRow]:
    q = (
        session.query(
            Trade.platform,
            Trade.entity_id,
            func.count(Trade.id).label("num_trades"),
            func.sum(Trade.notional_usd).label("total_notional_usd"),
            func.min(Trade.ts).label("first_seen_ts"),
            func.max(Trade.ts).label("last_seen_ts"),
        )
        .filter(Trade.entity_id.isnot(None), Trade.entity_id != "")
        .group_by(Trade.platform, Trade.entity_id)
    )
    if platform:
        q = q.filter(Trade.platform == platform)
    if min_trades > 0:
        q = q.having(func.count(Trade.id) >= min_trades)
    if sort == "score_desc":
        q = q.order_by(desc("last_seen_ts"))

    rows: list[EntityLeaderboardRow] = []
    for platform_val, entity_id, num_trades, total_notional, first_seen, last_seen in q.limit(1000).all():
        market_profile = _entity_market_profile(session, platform_val, entity_id)
        if not _market_filter_pass(market_profile["is_sports"], market_filter):
            continue
        current_price = _current_price_for_entity(session, platform_val, entity_id, market_profile.get("top_market_id"))
        rows.append(
            EntityLeaderboardRow(
                entity_id=entity_id,
                platform=platform_val,
                anomaly_score_0_100=0.0,
                market=market_profile["label"],
                current_price=current_price,
                current_american_odds=_prob_to_american_odds(current_price),
                top_explanations=["No model scores available for this window yet."],
                num_trades=int(num_trades or 0),
                total_notional_usd=float(total_notional or 0.0),
                first_seen_ts=first_seen.isoformat() if first_seen else "",
                last_seen_ts=last_seen.isoformat() if last_seen else "",
            )
        )
    return rows


def _entity_market_profile(session, platform: str, entity_id: str) -> dict:
    rows = (
        session.query(Market.market_id, Market.title, func.sum(Trade.notional_usd).label("vol"))
        .join(Market, (Market.platform == Trade.platform) & (Market.market_id == Trade.market_id))
        .filter(Trade.platform == platform, Trade.entity_id == entity_id)
        .group_by(Market.market_id, Market.title)
        .order_by(desc("vol"))
        .limit(2)
        .all()
    )
    category_rows = (
        session.query(Market.category, func.sum(Trade.notional_usd).label("vol"))
        .join(Market, (Market.platform == Trade.platform) & (Market.market_id == Trade.market_id))
        .filter(Trade.platform == platform, Trade.entity_id == entity_id)
        .group_by(Market.category)
        .order_by(desc("vol"))
        .all()
    )
    titles = [str(title) for _, title, _ in rows if title]
    categories = [str(cat) for cat, _ in category_rows if cat]
    is_sports = _is_sports_market(categories, titles)
    return {
        "label": _nlp_market_summary(titles, categories, is_sports=is_sports),
        "is_sports": is_sports,
        "top_market_id": (rows[0][0] if rows else None),
        "top_market_title": (rows[0][1] if rows else None),
    }


def _market_filter_pass(is_sports: bool, market_filter: str) -> bool:
    mode = (market_filter or "all").lower()
    if mode == "sports":
        return is_sports
    if mode in {"non_sports", "non-sports"}:
        return not is_sports
    return True


def _is_sports_market(categories: list[str], titles: list[str]) -> bool:
    cat_text = " ".join(categories).lower()
    if any(k in cat_text for k in {"sport", "nba", "nfl", "mlb", "nhl", "ncaa", "soccer", "football", "tennis"}):
        return True
    title_text = " ".join(titles).lower()
    sports_terms = {
        "vs", "spread", "o/u", "over", "under", "touchdown", "quarterback",
        "basketball", "football", "soccer", "baseball", "hockey", "tennis",
        "ncaa", "nfl", "nba", "mlb", "nhl",
    }
    return any(term in title_text for term in sports_terms)


def _nlp_market_summary(titles: list[str], categories: list[str], is_sports: bool) -> str:
    if not titles:
        return "Unknown market"
    stopwords = {"the", "and", "for", "with", "over", "under", "spread", "total",
                 "market", "versus", "vs", "will", "are", "from", "into"}
    tokens: list[str] = []
    for title in titles:
        tokens.extend([t.lower() for t in re.findall(r"[A-Za-z]{3,}", title)])
    keywords = [t for t in tokens if t not in stopwords]
    top_keywords = [kw for kw, _ in Counter(keywords).most_common(4)]
    primary_cat = _canonical_category(categories, titles, is_sports)
    if not top_keywords:
        return f"{primary_cat}: {titles[0][:80]}"
    return f"{primary_cat}: {', '.join(top_keywords)}"


def _canonical_category(categories: list[str], titles: list[str], is_sports: bool) -> str:
    if is_sports:
        return "sports"
    text = " ".join([*(c for c in categories if c), *(t for t in titles if t)]).lower()
    rules: list[tuple[str, tuple[str, ...]]] = [
        ("crypto", ("crypto", "bitcoin", "btc", "ethereum", "eth", "solana", "sol", "xrp", "doge")),
        ("politics", ("election", "president", "senate", "house", "democrat", "republican", "trump", "biden")),
        ("macro", ("fed", "fomc", "inflation", "cpi", "gdp", "recession", "rates", "yield")),
        ("tech", ("openai", "ai", "llm", "gpu", "nvidia", "tesla", "apple", "microsoft", "google")),
        ("geopolitics", ("war", "ceasefire", "ukraine", "russia", "china", "taiwan", "israel", "iran")),
        ("culture", ("movie", "oscar", "grammy", "music", "tv", "celebrity")),
    ]
    for label, keys in rules:
        if any(k in text for k in keys):
            return label
    return "other"


def _nlp_reason_text(raw_reason: str) -> str:
    pattern = r"^([a-zA-Z0-9_]+):\s*([\-0-9\.]+)\s*\(([\-0-9\.]+)th pct\);\s*high reconstruction error$"
    m = re.match(pattern, raw_reason.strip())
    if not m:
        return raw_reason
    feature, value_str, pct_str = m.groups()
    label = FEATURE_LABELS.get(feature, feature.replace("_", " ").title())
    try:
        value = float(value_str)
        pct = float(pct_str)
    except ValueError:
        return f"{label} deviates from baseline behavior."
    if pct >= 97:
        level = "extremely unusual"
    elif pct >= 90:
        level = "very unusual"
    elif pct <= 3:
        level = "extremely low vs peers"
    elif pct <= 10:
        level = "unusually low vs peers"
    else:
        level = "unusual"
    value_fmt = f"{value:.4f}".rstrip("0").rstrip(".")
    return f"{label} is {level} ({pct:.1f}th percentile, value {value_fmt})."


def _current_price_for_entity(session, platform: str, entity_id: str, market_id: str | None) -> float | None:
    if market_id:
        trade = (
            session.query(Trade)
            .filter(Trade.platform == platform, Trade.entity_id == entity_id, Trade.market_id == market_id)
            .order_by(desc(Trade.ts))
            .first()
        )
        if trade is not None:
            return float(trade.price)
    trade = (
        session.query(Trade)
        .filter(Trade.platform == platform, Trade.entity_id == entity_id)
        .order_by(desc(Trade.ts))
        .first()
    )
    return float(trade.price) if trade is not None else None


def _latest_model_version_for_platform(session, window: str, platform: str | None) -> str | None:
    q = session.query(EntityScore.model_version).filter(EntityScore.window == window)
    if platform:
        q = q.filter(EntityScore.platform == platform)
    row = q.order_by(desc(EntityScore.created_ts)).first()
    return str(row[0]) if row else None


def _prob_to_american_odds(price: object | None) -> int | None:
    if price is None:
        return None
    try:
        p = float(price)
    except Exception:
        return None
    if p > 1.0 and p <= 100.0:
        p = p / 100.0
    if p <= 0.0 or p >= 1.0:
        return None
    if p >= 0.5:
        odds = -100.0 * p / (1.0 - p)
    else:
        odds = 100.0 * (1.0 - p) / p
    return int(round(odds))


# ---------------------------------------------------------------------------
# Serve single-file SPA
# ---------------------------------------------------------------------------

@app.get("/{full_path:path}", include_in_schema=False)
def serve_spa(full_path: str) -> FileResponse:
    index = _STATIC_DIR / "index.html"
    if index.is_file():
        return FileResponse(index)
    raise HTTPException(status_code=404, detail="Frontend not built. Create frontend/index.html.")
