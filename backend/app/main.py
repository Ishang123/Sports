from __future__ import annotations

import json
import logging
import re
import time
from collections import Counter
from difflib import SequenceMatcher
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import desc, func, select

from app.connectors.kalshi import KalshiConnector
from app.db import get_session, init_db, resolve_db_path
from app.models_db import Entity, EntityScore, EntityWindowFeature, Market, MarketMapping, ModelRegistry, Trade
from app.schemas import (
    EntityDetail,
    EntityLeaderboardRow,
    MarketMappingIn,
    MarketMappingOut,
    MarketMappingSuggestion,
    ModelLatest,
)

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
_KALSHI_CACHE: dict[str, object] = {"ts": 0.0, "markets": []}
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup() -> None:
    init_db()
    logger.warning("Integrity API using SQLite DB at: %s", resolve_db_path())


@app.get("/health")
def health() -> dict:
    return {"ok": True}


@app.get("/api/backtest/latest")
def backtest_latest() -> dict:
    root = Path(__file__).resolve().parents[2]
    candidates = [
        root / "backend" / "artifacts" / "backtest_latest.json",
        root / "artifacts" / "backtest_latest.json",  # legacy fallback
    ]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        raise HTTPException(status_code=404, detail="Backtest report not found. Run make backtest first.")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to parse backtest report: {exc}") from exc


@app.get("/api/entities", response_model=list[EntityLeaderboardRow])
def list_entities(
    platform: str | None = Query(default=None),
    window: str = Query(default="30d"),
    min_trades: int = Query(default=10),
    sort: str = Query(default="score_desc"),
    market_filter: str = Query(default="all"),
    kalshi_equivalent_only: bool = Query(default=False),
    bankroll_usd: float = Query(default=500.0, ge=1.0),
    kelly_k: float = Query(default=0.015, ge=0.0, le=0.2),
    kelly_cap_pct: float = Query(default=0.01, ge=0.0, le=0.25),
):
    with get_session() as session:
        latest_model_version = _latest_model_version_for_platform(session, window=window, platform=platform)
        if not latest_model_version:
            return _fallback_entities(
                session,
                window=window,
                platform=platform,
                min_trades=min_trades,
                sort=sort,
                market_filter=market_filter,
                kalshi_equivalent_only=kalshi_equivalent_only,
                bankroll_usd=bankroll_usd,
                kelly_k=kelly_k,
                kelly_cap_pct=kelly_cap_pct,
            )

        q = session.query(EntityScore, Entity, EntityWindowFeature).join(
            Entity,
            (Entity.entity_id == EntityScore.entity_id) & (Entity.platform == EntityScore.platform),
        ).outerjoin(
            EntityWindowFeature,
            (EntityWindowFeature.entity_id == EntityScore.entity_id)
            & (EntityWindowFeature.platform == EntityScore.platform)
            & (EntityWindowFeature.window == EntityScore.window),
        ).filter(
            EntityScore.model_version == latest_model_version,
            EntityScore.window == window,
            EntityScore.platform == platform,
        )
        if min_trades > 0:
            q = q.filter(func.coalesce(EntityWindowFeature.num_trades, 0) >= min_trades)

        if platform:
            q = q.filter(EntityScore.platform == platform)

        if sort == "score_desc":
            q = q.order_by(desc(EntityScore.anomaly_score_0_100))

        dedup: dict[tuple[str, str], tuple[EntityScore, Entity, EntityWindowFeature | None]] = {}
        for score, entity, feature in q.all():
            k = (entity.platform, entity.entity_id)
            prev = dedup.get(k)
            if prev is None:
                dedup[k] = (score, entity, feature)
                continue
            prev_feature = prev[2]
            prev_ts = prev_feature.as_of_ts if prev_feature is not None else None
            curr_ts = feature.as_of_ts if feature is not None else None
            if prev_ts is None or (curr_ts is not None and curr_ts > prev_ts):
                dedup[k] = (score, entity, feature)

        rows = []
        for score, entity, feature in dedup.values():
            market_profile = _entity_market_profile(session, entity.platform, entity.entity_id)
            if not _market_filter_pass(market_profile["is_sports"], market_filter):
                continue
            raw_explanations = json.loads(score.top_explanations_json)[:3]
            current_price = _current_price_for_entity(session, entity.platform, entity.entity_id, market_profile.get("top_market_id"))
            kalshi = _kalshi_price_context_for_row(session, entity.platform, market_profile.get("top_market_id"))
            if entity.platform == "kalshi" and kalshi["price"] is None:
                kalshi = {"price": current_price, "title": kalshi.get("title")}
            if kalshi_equivalent_only and kalshi["price"] is None:
                continue
            rows.append(
                EntityLeaderboardRow(
                    entity_id=entity.entity_id,
                    platform=entity.platform,
                    anomaly_score_0_100=score.anomaly_score_0_100,
                    market=market_profile["label"],
                    current_price=current_price,
                    current_american_odds=_prob_to_american_odds(current_price),
                    kalshi_price=kalshi["price"],
                    kalshi_american_odds=_prob_to_american_odds(kalshi["price"]),
                    kalshi_market=kalshi["title"],
                    quarter_kelly_fraction=_quarter_kelly_fraction(
                        kalshi_price=kalshi["price"],
                        sharp_score_0_100=score.anomaly_score_0_100,
                        kelly_k=kelly_k,
                    ),
                    quarter_kelly_stake_usd=_quarter_kelly_stake_usd(
                        kalshi_price=kalshi["price"],
                        sharp_score_0_100=score.anomaly_score_0_100,
                        bankroll_usd=bankroll_usd,
                        kelly_k=kelly_k,
                        kelly_cap_pct=kelly_cap_pct,
                    ),
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
        return _fallback_entities(
            session,
            window=window,
            platform=platform,
            min_trades=min_trades,
            sort=sort,
            market_filter=market_filter,
            kalshi_equivalent_only=kalshi_equivalent_only,
            bankroll_usd=bankroll_usd,
            kelly_k=kelly_k,
            kelly_cap_pct=kelly_cap_pct,
        )


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
            "distinct_trade_entity_ids": (
                session.query(func.count(func.distinct(Trade.entity_id)))
                .filter(Trade.entity_id.isnot(None), Trade.entity_id != "")
                .scalar()
                or 0
            ),
            "entities_count": session.query(func.count(Entity.id)).scalar() or 0,
            "features_count": session.query(func.count(EntityWindowFeature.id)).scalar() or 0,
            "markets_count": session.query(func.count(Market.id)).scalar() or 0,
            "scores_count": session.query(func.count(EntityScore.id)).scalar() or 0,
            "models_count": session.query(func.count(ModelRegistry.id)).scalar() or 0,
            "min_trade_ts": min_trade_ts.isoformat() if min_trade_ts is not None else None,
            "max_trade_ts": max_trade_ts.isoformat() if max_trade_ts is not None else None,
            "scores_for_window": session.query(func.count(EntityScore.id)).filter(EntityScore.window == window).scalar() or 0,
            "features_for_window": session.query(func.count(EntityWindowFeature.id))
            .filter(EntityWindowFeature.window == window)
            .scalar()
            or 0,
        }
        if latest_version:
            result["scores_for_latest_model"] = (
                session.query(func.count(EntityScore.id))
                .filter(EntityScore.window == window, EntityScore.model_version == latest_version)
                .scalar()
                or 0
            )
        else:
            result["scores_for_latest_model"] = 0
        return result


def _fallback_entities(
    session,
    window: str,
    platform: str | None,
    min_trades: int,
    sort: str,
    market_filter: str,
    kalshi_equivalent_only: bool,
    bankroll_usd: float,
    kelly_k: float,
    kelly_cap_pct: float,
) -> list[EntityLeaderboardRow]:
    entity_count = session.query(func.count(Entity.id)).scalar() or 0
    if entity_count == 0:
        return _fallback_entities_from_trades(
            session=session,
            platform=platform,
            min_trades=min_trades,
            sort=sort,
            market_filter=market_filter,
            kalshi_equivalent_only=kalshi_equivalent_only,
            bankroll_usd=bankroll_usd,
            kelly_k=kelly_k,
            kelly_cap_pct=kelly_cap_pct,
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
        kalshi = _kalshi_price_context_for_row(session, entity.platform, market_profile.get("top_market_id"))
        if entity.platform == "kalshi" and kalshi["price"] is None:
            current_price = _current_price_for_entity(session, entity.platform, entity.entity_id, market_profile.get("top_market_id"))
            kalshi = {"price": current_price, "title": kalshi.get("title")}
        else:
            current_price = _current_price_for_entity(session, entity.platform, entity.entity_id, market_profile.get("top_market_id"))
        if kalshi_equivalent_only and kalshi["price"] is None:
            continue

        rows.append(
            EntityLeaderboardRow(
                entity_id=entity.entity_id,
                platform=entity.platform,
                anomaly_score_0_100=0.0,
                market=market_profile["label"],
                current_price=current_price,
                current_american_odds=_prob_to_american_odds(current_price),
                kalshi_price=kalshi["price"],
                kalshi_american_odds=_prob_to_american_odds(kalshi["price"]),
                kalshi_market=kalshi["title"],
                quarter_kelly_fraction=_quarter_kelly_fraction(
                    kalshi_price=kalshi["price"],
                    sharp_score_0_100=0.0,
                    kelly_k=kelly_k,
                ),
                quarter_kelly_stake_usd=_quarter_kelly_stake_usd(
                    kalshi_price=kalshi["price"],
                    sharp_score_0_100=0.0,
                    bankroll_usd=bankroll_usd,
                    kelly_k=kelly_k,
                    kelly_cap_pct=kelly_cap_pct,
                ),
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
    kalshi_equivalent_only: bool,
    bankroll_usd: float,
    kelly_k: float,
    kelly_cap_pct: float,
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
        kalshi = _kalshi_price_context_for_row(session, platform_val, market_profile.get("top_market_id"))
        current_price = _current_price_for_entity(session, platform_val, entity_id, market_profile.get("top_market_id"))
        if platform_val == "kalshi" and kalshi["price"] is None:
            kalshi = {"price": current_price, "title": kalshi.get("title")}
        if kalshi_equivalent_only and kalshi["price"] is None:
            continue
        rows.append(
            EntityLeaderboardRow(
                entity_id=entity_id,
                platform=platform_val,
                anomaly_score_0_100=0.0,
                market=market_profile["label"],
                current_price=current_price,
                current_american_odds=_prob_to_american_odds(current_price),
                kalshi_price=kalshi["price"],
                kalshi_american_odds=_prob_to_american_odds(kalshi["price"]),
                kalshi_market=kalshi["title"],
                quarter_kelly_fraction=_quarter_kelly_fraction(
                    kalshi_price=kalshi["price"],
                    sharp_score_0_100=0.0,
                    kelly_k=kelly_k,
                ),
                quarter_kelly_stake_usd=_quarter_kelly_stake_usd(
                    kalshi_price=kalshi["price"],
                    sharp_score_0_100=0.0,
                    bankroll_usd=bankroll_usd,
                    kelly_k=kelly_k,
                    kelly_cap_pct=kelly_cap_pct,
                ),
                top_explanations=["No model scores available for this window yet."],
                num_trades=int(num_trades or 0),
                total_notional_usd=float(total_notional or 0.0),
                first_seen_ts=first_seen.isoformat() if first_seen else "",
                last_seen_ts=last_seen.isoformat() if last_seen else "",
            )
        )
    return rows


def _entity_market_profile(session, platform: str, entity_id: str) -> dict:
    # Leaderboard market uses top-2 market titles by entity notional.
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
        "ncaa", "nfl", "nba", "mlb", "nhl", "falcons", "eagles", "wolfpack",
        "tar", "heels", "chippewas", "seminoles", "billikens", "rams",
    }
    return any(term in title_text for term in sports_terms)


def _nlp_market_summary(titles: list[str], categories: list[str], is_sports: bool) -> str:
    if not titles:
        return "Unknown market"

    stopwords = {
        "the", "and", "for", "with", "over", "under", "spread", "total",
        "market", "versus", "vs", "will", "are", "from", "into",
    }
    sports_noise = {
        "falcons", "eagles", "wolfpack", "chippewas", "seminoles", "billikens",
        "rams", "tar", "heels", "michigan", "carolina", "boston", "saint",
    }
    tokens: list[str] = []
    for title in titles:
        tokens.extend([t.lower() for t in re.findall(r"[A-Za-z]{3,}", title)])
    keywords = [t for t in tokens if t not in stopwords and (not is_sports or t not in sports_noise)]
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
        ("crypto", ("crypto", "bitcoin", "btc", "ethereum", "eth", "solana", "sol", "xrp", "doge", "token")),
        ("politics", ("election", "president", "senate", "house", "democrat", "republican", "trump", "biden", "gop")),
        ("macro", ("fed", "fomc", "inflation", "cpi", "gdp", "recession", "rates", "yield", "economy")),
        ("tech", ("openai", "ai", "llm", "gpu", "nvidia", "tesla", "apple", "microsoft", "google", "meta")),
        ("geopolitics", ("war", "ceasefire", "ukraine", "russia", "china", "taiwan", "israel", "gaza", "iran")),
        ("culture", ("movie", "oscar", "grammy", "music", "tv", "celebrity", "box office")),
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


def _mapped_kalshi_price_for_market(session, source_market_id: str | None) -> dict[str, object | None]:
    if not source_market_id:
        return {"price": None, "title": None}

    mapping = (
        session.query(MarketMapping)
        .filter(
            MarketMapping.source_platform == "polymarket",
            MarketMapping.source_market_id == source_market_id,
            MarketMapping.target_platform == "kalshi",
        )
        .order_by(desc(MarketMapping.updated_ts))
        .first()
    )
    if mapping is None:
        return {"price": None, "title": None}

    markets = _kalshi_markets_cached()
    if not markets:
        return {"price": None, "title": None}

    for m in markets:
        if str(m.get("ticker") or "") == mapping.target_market_id:
            return {"price": _kalshi_price_from_market(m), "title": str(m.get("title") or "")}
    return {"price": None, "title": None}


def _kalshi_price_for_market_id(market_id: str | None) -> dict[str, object | None]:
    if not market_id:
        return {"price": None, "title": None}

    markets = _kalshi_markets_cached()
    if not markets:
        return {"price": None, "title": None}

    for m in markets:
        if str(m.get("ticker") or "") == str(market_id):
            return {"price": _kalshi_price_from_market(m), "title": str(m.get("title") or "")}
    return {"price": None, "title": None}


def _kalshi_price_context_for_row(session, platform: str, market_id: str | None) -> dict[str, object | None]:
    if platform == "kalshi":
        return _kalshi_price_for_market_id(market_id)
    if platform == "polymarket":
        return _mapped_kalshi_price_for_market(session, market_id)
    return {"price": None, "title": None}


def _latest_model_version_for_platform(session, window: str, platform: str | None) -> str | None:
    q = session.query(EntityScore.model_version).filter(EntityScore.window == window)
    if platform:
        q = q.filter(EntityScore.platform == platform)
    row = q.order_by(desc(EntityScore.created_ts)).first()
    return str(row[0]) if row else None


def _quarter_kelly_fraction(kalshi_price: object | None, sharp_score_0_100: float, kelly_k: float) -> float | None:
    if kalshi_price is None:
        return None
    try:
        p_imp = float(kalshi_price)
    except Exception:
        return None
    if p_imp <= 0.0 or p_imp >= 1.0:
        return None

    s = max(0.0, min(1.0, float(sharp_score_0_100) / 100.0))
    p_win = max(0.01, min(0.99, p_imp + kelly_k * (s - 0.5)))
    b = (1.0 - p_imp) / p_imp
    if b <= 0:
        return None
    f = ((b * p_win) - (1.0 - p_win)) / b
    fq = max(0.0, f / 4.0)
    return fq


def _quarter_kelly_stake_usd(
    kalshi_price: object | None,
    sharp_score_0_100: float,
    bankroll_usd: float,
    kelly_k: float,
    kelly_cap_pct: float,
) -> float | None:
    fq = _quarter_kelly_fraction(kalshi_price=kalshi_price, sharp_score_0_100=sharp_score_0_100, kelly_k=kelly_k)
    if fq is None:
        return None
    capped_fraction = min(fq, max(0.0, kelly_cap_pct))
    stake = float(bankroll_usd) * capped_fraction
    return round(stake, 2)


def _kalshi_markets_cached(ttl_seconds: int = 20) -> list[dict]:
    now = time.time()
    if now - float(_KALSHI_CACHE["ts"]) < ttl_seconds and _KALSHI_CACHE["markets"]:
        return _KALSHI_CACHE["markets"]  # type: ignore[return-value]
    try:
        connector = KalshiConnector()
        markets = connector.list_raw_markets(limit=5000, status="")
        _KALSHI_CACHE["ts"] = now
        _KALSHI_CACHE["markets"] = markets
        return markets
    except Exception:
        return _KALSHI_CACHE.get("markets", []) if isinstance(_KALSHI_CACHE.get("markets", []), list) else []


def _kalshi_price_from_market(market: dict) -> float | None:
    fields = [
        market.get("last_price_dollars"),
        market.get("yes_price_dollars"),
        market.get("yes_bid_dollars"),
        market.get("yes_ask_dollars"),
        market.get("last_price"),
        market.get("yes_price"),
    ]
    for v in fields:
        if v is None:
            continue
        try:
            return float(v)
        except Exception:
            continue
    bid = market.get("yes_bid_dollars")
    ask = market.get("yes_ask_dollars")
    if bid is not None and ask is not None:
        try:
            return (float(bid) + float(ask)) / 2.0
        except Exception:
            return None
    return None


def _prob_to_american_odds(price: object | None) -> int | None:
    if price is None:
        return None
    try:
        p = float(price)
    except Exception:
        return None

    # Support both [0,1] dollars and [0,100] cents-like inputs.
    if p > 1.0 and p <= 100.0:
        p = p / 100.0
    if p <= 0.0 or p >= 1.0:
        return None

    if p >= 0.5:
        odds = -100.0 * p / (1.0 - p)
    else:
        odds = 100.0 * (1.0 - p) / p
    return int(round(odds))


def _normalize_title(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _title_similarity(a: str, b: str) -> float:
    a_n = _normalize_title(a)
    b_n = _normalize_title(b)
    if not a_n or not b_n:
        return 0.0
    seq = SequenceMatcher(None, a_n, b_n).ratio()
    a_tokens = set(a_n.split())
    b_tokens = set(b_n.split())
    inter = len(a_tokens & b_tokens)
    union = len(a_tokens | b_tokens) or 1
    jacc = inter / union
    return 0.65 * seq + 0.35 * jacc


@app.get("/api/mappings", response_model=list[MarketMappingOut])
def list_mappings(
    source_platform: str = Query(default="polymarket"),
    target_platform: str = Query(default="kalshi"),
    limit: int = Query(default=200),
):
    with get_session() as session:
        rows = (
            session.query(MarketMapping)
            .filter(
                MarketMapping.source_platform == source_platform,
                MarketMapping.target_platform == target_platform,
            )
            .order_by(desc(MarketMapping.updated_ts))
            .limit(limit)
            .all()
        )
        return [
            MarketMappingOut(
                source_platform=r.source_platform,
                source_market_id=r.source_market_id,
                target_platform=r.target_platform,
                target_market_id=r.target_market_id,
                confidence=r.confidence,
                method=r.method,
                notes=r.notes,
                updated_ts=r.updated_ts.isoformat(),
            )
            for r in rows
        ]


@app.get("/api/mappings/suggest", response_model=list[MarketMappingSuggestion])
def suggest_mappings(
    limit: int = Query(default=50),
    min_similarity: float = Query(default=0.52),
    auto_apply: bool = Query(default=False),
):
    with get_session() as session:
        poly_rows = (
            session.query(
                Market.market_id,
                Market.title,
                func.sum(Trade.notional_usd).label("vol"),
            )
            .join(Trade, (Trade.platform == Market.platform) & (Trade.market_id == Market.market_id))
            .filter(Market.platform == "polymarket")
            .group_by(Market.market_id, Market.title)
            .order_by(desc("vol"))
            .limit(max(limit * 4, 200))
            .all()
        )
        if not poly_rows:
            return []

        existing = {
            (m.source_market_id, m.target_market_id)
            for m in session.query(MarketMapping)
            .filter(MarketMapping.source_platform == "polymarket", MarketMapping.target_platform == "kalshi")
            .all()
        }

        kalshi = _kalshi_markets_cached()
        if not kalshi:
            return []
        kalshi_pairs = [
            (str(m.get("ticker") or ""), str(m.get("title") or ""))
            for m in kalshi
            if m.get("ticker") and m.get("title")
        ]

        suggestions: list[MarketMappingSuggestion] = []
        for src_id, src_title, _ in poly_rows:
            best_ticker = ""
            best_title = ""
            best_sim = 0.0
            for ticker, title in kalshi_pairs:
                sim = _title_similarity(str(src_title), title)
                if sim > best_sim:
                    best_sim = sim
                    best_ticker = ticker
                    best_title = title
            if best_sim < min_similarity or not best_ticker:
                continue
            if (str(src_id), best_ticker) in existing:
                continue

            applied = False
            if auto_apply:
                row = (
                    session.query(MarketMapping)
                    .filter(
                        MarketMapping.source_platform == "polymarket",
                        MarketMapping.source_market_id == str(src_id),
                        MarketMapping.target_platform == "kalshi",
                    )
                    .first()
                )
                now = datetime.now(timezone.utc).replace(tzinfo=None)
                if row is None:
                    row = MarketMapping(
                        source_platform="polymarket",
                        source_market_id=str(src_id),
                        target_platform="kalshi",
                        target_market_id=best_ticker,
                        confidence=float(best_sim),
                        method="auto_suggest",
                        notes="auto-suggested by title similarity",
                        updated_ts=now,
                    )
                    session.add(row)
                else:
                    row.target_market_id = best_ticker
                    row.confidence = float(best_sim)
                    row.method = "auto_suggest"
                    row.notes = "auto-suggested by title similarity"
                    row.updated_ts = now
                applied = True

            suggestions.append(
                MarketMappingSuggestion(
                    source_market_id=str(src_id),
                    source_title=str(src_title),
                    target_market_id=best_ticker,
                    target_title=best_title,
                    similarity=float(round(best_sim, 4)),
                    applied=applied,
                )
            )
            if len(suggestions) >= limit:
                break
        return suggestions


@app.post("/api/mappings", response_model=MarketMappingOut)
def upsert_mapping(payload: MarketMappingIn):
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    with get_session() as session:
        row = (
            session.query(MarketMapping)
            .filter(
                MarketMapping.source_platform == payload.source_platform,
                MarketMapping.source_market_id == payload.source_market_id,
                MarketMapping.target_platform == payload.target_platform,
            )
            .first()
        )
        if row is None:
            row = MarketMapping(
                source_platform=payload.source_platform,
                source_market_id=payload.source_market_id,
                target_platform=payload.target_platform,
                target_market_id=payload.target_market_id,
                confidence=payload.confidence,
                method=payload.method,
                notes=payload.notes,
                updated_ts=now,
            )
            session.add(row)
        else:
            row.target_market_id = payload.target_market_id
            row.confidence = payload.confidence
            row.method = payload.method
            row.notes = payload.notes
            row.updated_ts = now

        return MarketMappingOut(
            source_platform=row.source_platform,
            source_market_id=row.source_market_id,
            target_platform=row.target_platform,
            target_market_id=row.target_market_id,
            confidence=row.confidence,
            method=row.method,
            notes=row.notes,
            updated_ts=row.updated_ts.isoformat(),
        )


@app.get("/api/entities/{entity_id}", response_model=EntityDetail)
def get_entity(entity_id: str, window: str = Query(default="30d"), platform: str | None = Query(default=None)):
    with get_session() as session:
        latest_model_version = _latest_model_version_for_platform(session, window=window, platform=platform)
        if not latest_model_version:
            raise HTTPException(status_code=404, detail="Model not found")

        score_q = (
            session.query(EntityScore)
            .filter(
                EntityScore.entity_id == entity_id,
                EntityScore.window == window,
                EntityScore.model_version == latest_model_version,
            )
        )
        if platform:
            score_q = score_q.filter(EntityScore.platform == platform)
        score = score_q.first()
        if not score:
            raise HTTPException(status_code=404, detail="Entity score not found")

        entity = (
            session.query(Entity)
            .filter(Entity.entity_id == entity_id, Entity.platform == score.platform)
            .first()
        )
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
            {"market_id": m_id, "notional_usd": float(notional)} for m_id, notional in session.execute(top_markets_stmt).all()
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
