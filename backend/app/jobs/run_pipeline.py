from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from sqlalchemy import func, select

ROOT = Path(__file__).resolve().parents[3]
BACKEND_ROOT = ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.connectors.base import TimeRange
from app.connectors.kalshi import KalshiConnector
from app.connectors.polymarket_like import PolymarketLikeConnector
from app.connectors.polymarket_real import PolymarketRealConnector
from app.db import get_session, init_db, resolve_db_path
from app.features.build_features import FEATURE_COLUMNS, build_entity_window_features, parse_window
from app.ml.score_ae import score_entities
from app.ml.train_ae import train_autoencoder
from app.models_db import Entity, EntityScore, EntityWindowFeature, Market, ModelRegistry, Trade


def _safe_json_dumps(payload: object) -> str:
    return json.dumps(payload, default=str, ensure_ascii=False)


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _to_naive_utc(dt: datetime) -> datetime:
    """Convert aware->naive UTC; keep naive as-is."""
    if dt.tzinfo is None:
        return dt
    return dt.astimezone(timezone.utc).replace(tzinfo=None)


def _parse_iso_to_naive(value: str | None) -> datetime | None:
    if not value:
        return None
    return _to_naive_utc(datetime.fromisoformat(value.replace("Z", "+00:00")))


def upsert_market(session, platform: str, market: dict) -> None:
    market_id = str(market.get("market_id") or "").strip()
    if not market_id:
        return

    existing = session.execute(
        select(Market).where(Market.platform == platform, Market.market_id == market_id)
    ).scalar_one_or_none()

    payload = {
        "platform": platform,
        "market_id": market_id,
        "title": str(market.get("title") or market_id)[:512],
        "category": (str(market.get("category"))[:128] if market.get("category") is not None else None),
        "open_ts": _parse_iso_to_naive(market.get("open_ts")),
        "close_ts": _parse_iso_to_naive(market.get("close_ts")),
        "resolve_ts": _parse_iso_to_naive(market.get("resolve_ts")),
        "outcome": market.get("outcome"),
        "metadata_json": _safe_json_dumps(market.get("metadata", {})),
    }

    if existing:
        for k, v in payload.items():
            setattr(existing, k, v)
    else:
        session.add(Market(**payload))


def upsert_trade(session, platform: str, trade: dict) -> None:
    trade_id = str(trade.get("trade_id") or "").strip()
    market_id = str(trade.get("market_id") or "").strip()
    entity_id = str(trade.get("entity_id") or "").strip() or "unknown"
    if not trade_id or not market_id:
        return

    existing = session.execute(
        select(Trade).where(Trade.platform == platform, Trade.trade_id == trade_id)
    ).scalar_one_or_none()

    payload = {
        "platform": platform,
        "trade_id": trade_id,
        "market_id": market_id,
        "entity_id": entity_id,
        "ts": _to_naive_utc(datetime.fromisoformat(trade["ts"].replace("Z", "+00:00"))),
        "side": trade["side"],
        "price": float(trade["price"]),
        "quantity": float(trade["quantity"]),
        "notional_usd": float(trade["notional_usd"]),
        "raw_json": _safe_json_dumps(trade.get("raw_json", trade)),
    }

    if existing:
        for k, v in payload.items():
            setattr(existing, k, v)
    else:
        session.add(Trade(**payload))


def upsert_entity_from_trades(session) -> None:
    # Required because SessionLocal has autoflush=False.
    # Without this, selecting trades here can miss just-inserted rows.
    session.flush()

    trades = session.execute(select(Trade)).scalars().all()
    grouped: dict[tuple[str, str], list[datetime]] = {}
    for t in trades:
        if not t.entity_id:
            continue
        grouped.setdefault((t.platform, t.entity_id), []).append(t.ts)

    for (platform, entity_id), tss in grouped.items():
        existing = session.execute(
            select(Entity).where(Entity.platform == platform, Entity.entity_id == entity_id)
        ).scalar_one_or_none()

        payload = {
            "platform": platform,
            "entity_id": entity_id,
            "first_seen_ts": min(tss),
            "last_seen_ts": max(tss),
            "metadata_json": json.dumps({}),
        }

        if existing:
            existing.first_seen_ts = payload["first_seen_ts"]
            existing.last_seen_ts = payload["last_seen_ts"]
        else:
            session.add(Entity(**payload))

    session.flush()


def run(
    mode: str,
    window: str,
    platform: str,
    buffer_days: int,
    markets_limit: int,
    trades_limit: int | None,
) -> None:
    if mode not in {"sample", "live"}:
        raise ValueError("Mode must be one of: sample, live")
    init_db()
    print(f"Pipeline using SQLite DB at: {resolve_db_path()}")

    # Use tz-aware time ONLY for connector fetch filtering.
    now_aware = datetime.now(timezone.utc)
    if mode == "sample":
        if platform != "polymarket_like":
            raise ValueError("Sample mode currently supports --platform polymarket_like only")
        horizon = max(parse_window(window), timedelta(days=120))
        tr_aware = TimeRange(start=now_aware - horizon, end=now_aware)
        connector = PolymarketLikeConnector(
            markets_path=ROOT / "data" / "sample_markets.json",
            trades_path=ROOT / "data" / "sample_trades.json",
        )
    else:
        horizon = parse_window(window) + timedelta(days=buffer_days)
        tr_aware = TimeRange(start=now_aware - horizon, end=now_aware)
        if platform == "kalshi":
            connector = KalshiConnector(
                base_url=os.getenv("KALSHI_BASE_URL", "https://api.elections.kalshi.com/trade-api/v2")
            )
        elif platform == "polymarket":
            connector = PolymarketRealConnector(
                gamma_base_url=os.getenv("POLYMARKET_GAMMA_BASE_URL", "https://gamma-api.polymarket.com"),
                clob_url=os.getenv("POLYMARKET_CLOB_BASE_URL", "https://clob.polymarket.com"),
                data_api_url=os.getenv("POLYMARKET_DATA_API_URL", "https://data-api.polymarket.com"),
                api_key=os.getenv("POLYMARKET_API_KEY"),
            )
        else:
            raise ValueError("Live mode supports --platform kalshi or --platform polymarket")

    with get_session() as session:
        # --- Fetch (tz-aware) ---
        markets = connector.fetch_markets(tr_aware, limit=markets_limit)
        if not markets:
            wide_tr = TimeRange(
                start=datetime(1970, 1, 1, tzinfo=timezone.utc),
                end=now_aware + timedelta(days=365),
            )
            markets = connector.fetch_markets(wide_tr, limit=markets_limit)

        market_ids = [m["market_id"] for m in markets]
        trades = connector.fetch_trades(tr_aware, market_ids=market_ids or None, limit=trades_limit)
        if not trades and mode == "sample":
            wide_tr = TimeRange(
                start=datetime(1970, 1, 1, tzinfo=timezone.utc),
                end=now_aware + timedelta(days=365),
            )
            trades = connector.fetch_trades(wide_tr, market_ids=market_ids or None, limit=trades_limit)
        if not trades:
            raise RuntimeError("No trades fetched for the selected mode/time range")

        # --- Ingest into DB (tz-naive) ---
        # Dedupe connector payloads before upsert to avoid pending-row UNIQUE conflicts.
        dedup_markets: dict[str, dict] = {}
        for m in markets:
            m_id = str(m.get("market_id") or "").strip()
            if not m_id:
                continue
            dedup_markets[m_id] = m

        dedup_trades: dict[str, dict] = {}
        for t in trades:
            t_id = str(t.get("trade_id") or "").strip()
            if not t_id:
                continue
            dedup_trades[t_id] = t

        print(
            "Fetch dedupe debug | "
            f"markets_raw={len(markets)} markets_unique={len(dedup_markets)} "
            f"trades_raw={len(trades)} trades_unique={len(dedup_trades)}"
        )

        for m in dedup_markets.values():
            upsert_market(session, connector.platform, m)
        session.flush()

        for t in dedup_trades.values():
            upsert_trade(session, connector.platform, t)
        session.flush()

        upsert_entity_from_trades(session)

        distinct_trade_entities = (
            session.query(func.count(func.distinct(Trade.entity_id)))
            .filter(Trade.entity_id.isnot(None), Trade.entity_id != "")
            .scalar()
            or 0
        )
        entity_rows_after_upsert = session.query(func.count(Entity.id)).scalar() or 0
        print(
            "Entity upsert debug | "
            f"distinct_trade_entity_ids={distinct_trade_entities} "
            f"entities_in_table={entity_rows_after_upsert}"
        )

        # IMPORTANT: commit ingestion so rollback later won't wipe it
        session.commit()

        # --- Feature build (tz-naive, based on latest ingested trade) ---
        latest_trade_ts = session.execute(
            select(Trade.ts).order_by(Trade.ts.desc()).limit(1)
        ).scalar_one_or_none()

        # latest_trade_ts is stored tz-naive in DB
        as_of_ts = latest_trade_ts or now_aware.replace(tzinfo=None)

        feat_df = build_entity_window_features(session, window=window, as_of_ts=as_of_ts)
        if feat_df.empty:
            # Debug info that won't destroy ingestion now that we committed
            cnt = session.execute(select(Trade.id).limit(1)).scalar_one_or_none()
            minmax = session.execute(select(Trade.ts).order_by(Trade.ts.asc()).limit(1)).scalar_one_or_none(), \
                     session.execute(select(Trade.ts).order_by(Trade.ts.desc()).limit(1)).scalar_one_or_none()
            raise RuntimeError(
                "No features produced. "
                f"Trades exist? {'yes' if cnt is not None else 'no'}. "
                f"DB ts range: {minmax[0]} .. {minmax[1]}. "
                f"as_of_ts={as_of_ts} window={window}"
            )

        model_version = now_aware.strftime("%Y%m%d%H%M%S")
        artifact_root = ROOT / "backend" / "artifacts" / window
        params = {
            "seed": _env_int("AE_SEED", 42),
            "epochs": _env_int("AE_EPOCHS", 80),
            "batch_size": _env_int("AE_BATCH_SIZE", 32),
            "latent_dim": _env_int("AE_LATENT_DIM", 8),
            "lr": _env_float("AE_LR", 1e-3),
            "dropout": _env_float("AE_DROPOUT", 0.1),
            "weight_decay": _env_float("AE_WEIGHT_DECAY", 1e-5),
            "patience": _env_int("AE_PATIENCE", 10),
        }

        artifacts = train_autoencoder(
            df=feat_df,
            feature_columns=FEATURE_COLUMNS,
            artifact_root=artifact_root,
            model_version=model_version,
            params=params,
        )

        scored = score_entities(
            session=session,
            df=feat_df,
            artifact_dir=artifacts.model_dir,
            model_version=model_version,
            window=window,
        )

        metrics = {
            **artifacts.metrics,
            "score_mean": float(scored["anomaly_score_0_100"].mean()),
            "score_std": float(scored["anomaly_score_0_100"].std(ddof=0)),
            "score_p95": float(scored["anomaly_score_0_100"].quantile(0.95)),
        }
        session.add(
            ModelRegistry(
                model_version=model_version,
                window=window,
                params_json=json.dumps(params),
                metrics_json=json.dumps(metrics),
                created_ts=now_aware.replace(tzinfo=None),
            )
        )
        session.commit()

        trades_count = session.query(func.count(Trade.id)).scalar() or 0
        entities_count = session.query(func.count(Entity.id)).scalar() or 0
        features_count = session.query(func.count(EntityWindowFeature.id)).scalar() or 0
        scores_count = session.query(func.count(EntityScore.id)).scalar() or 0
        print(
            "Post-pipeline row counts | "
            f"trades={trades_count} entities={entities_count} "
            f"features={features_count} scores={scores_count} "
            f"model_version={model_version}"
        )

    print(f"Pipeline complete | mode={mode} window={window} model_version={model_version}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="live", choices=["sample", "live"])
    parser.add_argument("--platform", default="kalshi", choices=["polymarket_like", "kalshi", "polymarket"])
    parser.add_argument("--window", default="30d", choices=["7d", "30d", "90d"])
    parser.add_argument("--backfill-buffer-days", type=int, default=7)
    parser.add_argument("--markets-limit", type=int, default=2000)
    parser.add_argument("--trades-limit", type=int, default=None)
    args = parser.parse_args()

    os.chdir(ROOT)
    run(
        mode=args.mode,
        window=args.window,
        platform=args.platform,
        buffer_days=args.backfill_buffer_days,
        markets_limit=args.markets_limit,
        trades_limit=args.trades_limit,
    )


if __name__ == "__main__":
    main()
