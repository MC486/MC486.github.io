from __future__ import annotations

from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class Bar:
    """A single OHLCV price bar for one symbol."""
    symbol: str
    day: date
    open: float
    high: float
    low: float
    close: float
    volume: float
