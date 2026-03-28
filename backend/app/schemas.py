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
