from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class EntityLeaderboardRow(BaseModel):
    entity_id: str
    platform: str
    anomaly_score_0_100: float
    market: str
    current_price: float | None = None
    current_american_odds: int | None = None
    kalshi_price: float | None = None
    kalshi_american_odds: int | None = None
    kalshi_market: str | None = None
    quarter_kelly_fraction: float | None = None
    quarter_kelly_stake_usd: float | None = None
    top_explanations: list[str]
    num_trades: int
    total_notional_usd: float
    first_seen_ts: str
    last_seen_ts: str


class EntityDetail(BaseModel):
    entity_id: str
    platform: str
    window: str
    anomaly_score_0_100: float
    score_raw: float
    top_explanations: list[str]
    feature_snapshot: dict[str, Any]
    recent_trades: list[dict[str, Any]]
    top_markets: list[dict[str, Any]]


class ModelLatest(BaseModel):
    model_version: str
    window: str
    metrics_summary: dict[str, Any]


class MarketMappingIn(BaseModel):
    source_platform: str = "polymarket"
    source_market_id: str
    target_platform: str = "kalshi"
    target_market_id: str
    confidence: float | None = None
    method: str | None = "manual"
    notes: str | None = None


class MarketMappingOut(BaseModel):
    source_platform: str
    source_market_id: str
    target_platform: str
    target_market_id: str
    confidence: float | None = None
    method: str | None = None
    notes: str | None = None
    updated_ts: str


class MarketMappingSuggestion(BaseModel):
    source_market_id: str
    source_title: str
    target_market_id: str
    target_title: str
    similarity: float
    applied: bool = False
