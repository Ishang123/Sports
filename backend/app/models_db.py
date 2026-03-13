from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Float, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from app.db import Base


class Market(Base):
    __tablename__ = "markets"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    platform: Mapped[str] = mapped_column(String(64), index=True)
    market_id: Mapped[str] = mapped_column(String(128), index=True)
    title: Mapped[str] = mapped_column(String(512))
    category: Mapped[str | None] = mapped_column(String(128), nullable=True)
    open_ts: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    close_ts: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    resolve_ts: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    outcome: Mapped[str | None] = mapped_column(String(64), nullable=True)
    metadata_json: Mapped[str | None] = mapped_column(Text, nullable=True)

    __table_args__ = (UniqueConstraint("platform", "market_id", name="uq_market_platform_id"),)


class Trade(Base):
    __tablename__ = "trades"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    platform: Mapped[str] = mapped_column(String(64), index=True)
    trade_id: Mapped[str] = mapped_column(String(128), index=True)
    market_id: Mapped[str] = mapped_column(String(128), index=True)
    entity_id: Mapped[str] = mapped_column(String(128), index=True)
    ts: Mapped[datetime] = mapped_column(DateTime, index=True)
    side: Mapped[str] = mapped_column(String(16))
    price: Mapped[float] = mapped_column(Float)
    quantity: Mapped[float] = mapped_column(Float)
    notional_usd: Mapped[float] = mapped_column(Float, index=True)
    raw_json: Mapped[str | None] = mapped_column(Text, nullable=True)

    __table_args__ = (UniqueConstraint("platform", "trade_id", name="uq_trade_platform_id"),)


class Entity(Base):
    __tablename__ = "entities"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    platform: Mapped[str] = mapped_column(String(64), index=True)
    entity_id: Mapped[str] = mapped_column(String(128), index=True)
    first_seen_ts: Mapped[datetime] = mapped_column(DateTime, index=True)
    last_seen_ts: Mapped[datetime] = mapped_column(DateTime, index=True)
    metadata_json: Mapped[str | None] = mapped_column(Text, nullable=True)

    __table_args__ = (UniqueConstraint("platform", "entity_id", name="uq_entity_platform_id"),)


class EntityWindowFeature(Base):
    __tablename__ = "entity_window_features"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    entity_id: Mapped[str] = mapped_column(String(128), index=True)
    platform: Mapped[str] = mapped_column(String(64), index=True)
    window: Mapped[str] = mapped_column(String(8), index=True)
    as_of_ts: Mapped[datetime] = mapped_column(DateTime, index=True)
    feature_json: Mapped[str] = mapped_column(Text)
    num_trades: Mapped[int] = mapped_column(Integer)
    total_notional_usd: Mapped[float] = mapped_column(Float)

    __table_args__ = (UniqueConstraint("entity_id", "platform", "window", "as_of_ts", name="uq_feat_entity_window"),)


class EntityScore(Base):
    __tablename__ = "entity_scores"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    entity_id: Mapped[str] = mapped_column(String(128), index=True)
    platform: Mapped[str] = mapped_column(String(64), index=True)
    window: Mapped[str] = mapped_column(String(8), index=True)
    model_version: Mapped[str] = mapped_column(String(64), index=True)
    score_raw: Mapped[float] = mapped_column(Float)
    anomaly_score_0_100: Mapped[float] = mapped_column(Float, index=True)
    top_explanations_json: Mapped[str] = mapped_column(Text)
    created_ts: Mapped[datetime] = mapped_column(DateTime, index=True)


class ModelRegistry(Base):
    __tablename__ = "model_registry"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    model_version: Mapped[str] = mapped_column(String(64), index=True)
    window: Mapped[str] = mapped_column(String(8), index=True)
    params_json: Mapped[str] = mapped_column(Text)
    metrics_json: Mapped[str] = mapped_column(Text)
    created_ts: Mapped[datetime] = mapped_column(DateTime, index=True)

    __table_args__ = (UniqueConstraint("model_version", "window", name="uq_model_version_window"),)


class MarketMapping(Base):
    __tablename__ = "market_mappings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    source_platform: Mapped[str] = mapped_column(String(64), index=True)
    source_market_id: Mapped[str] = mapped_column(String(128), index=True)
    target_platform: Mapped[str] = mapped_column(String(64), index=True)
    target_market_id: Mapped[str] = mapped_column(String(128), index=True)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    method: Mapped[str | None] = mapped_column(String(32), nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    updated_ts: Mapped[datetime] = mapped_column(DateTime, index=True)

    __table_args__ = (
        UniqueConstraint(
            "source_platform",
            "source_market_id",
            "target_platform",
            name="uq_market_mapping_source_target",
        ),
    )
