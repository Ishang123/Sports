from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime


@dataclass
class TimeRange:
    start: datetime
    end: datetime


class BaseConnector(ABC):
    platform: str

    @abstractmethod
    def fetch_markets(self, time_range: TimeRange, limit: int = 1000) -> list[dict]:
        raise NotImplementedError

    @abstractmethod
    def fetch_trades(
        self,
        time_range: TimeRange,
        market_ids: list[str] | None = None,
        limit: int | None = None,
    ) -> list[dict]:
        raise NotImplementedError

    def fetch_outcomes(self, time_range: TimeRange) -> list[dict]:
        return []
