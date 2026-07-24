from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from datetime import date, datetime, timedelta
from typing import Dict, List

import numpy as np

from .bar import Bar

logger = logging.getLogger(__name__)


class DataSource(ABC):
    """Abstract source of historical OHLCV bars."""

    @abstractmethod
    def get_bars(self) -> List[Bar]:
        """Return bars sorted ascending by day."""
        raise NotImplementedError


class SyntheticDataSource(DataSource):
    """Deterministic geometric-Brownian-motion price series.

    Used for offline, reproducible runs and tests. No network required.
    """

    def __init__(
        self,
        symbol: str = "SYNTH",
        n_days: int = 2016,
        start_price: float = 100.0,
        annual_drift: float = 0.07,
        annual_vol: float = 0.20,
        seed: int = 42,
        start: date | None = None,
    ) -> None:
        self.symbol = symbol
        self.n_days = int(n_days)
        self.start_price = float(start_price)
        self.annual_drift = float(annual_drift)
        self.annual_vol = float(annual_vol)
        self.seed = int(seed)
        self.start = start or date(2015, 1, 1)

    def get_bars(self) -> List[Bar]:
        rng = np.random.default_rng(self.seed)
        dt = 1.0 / 252.0
        mu = self.annual_drift
        sigma = self.annual_vol

        # Daily log returns of a GBM.
        shocks = rng.normal(
            (mu - 0.5 * sigma**2) * dt,
            sigma * np.sqrt(dt),
            size=self.n_days,
        )
        closes = self.start_price * np.exp(np.cumsum(shocks))

        bars: List[Bar] = []
        day = self.start
        prev_close = self.start_price
        for i in range(self.n_days):
            day = _next_business_day(day) if i > 0 else _align_business_day(day)
            close = float(closes[i])
            open_ = float(prev_close * (1.0 + rng.normal(0.0, 0.001)))
            hi = float(max(open_, close) * (1.0 + abs(rng.normal(0.0, 0.003))))
            lo = float(min(open_, close) * (1.0 - abs(rng.normal(0.0, 0.003))))
            vol = float(rng.integers(1_000_000, 5_000_000))
            bars.append(Bar(self.symbol, day, open_, hi, lo, close, vol))
            prev_close = close
        return bars


class YFinanceDataSource(DataSource):
    """Real daily bars via yfinance. Requires network access."""

    def __init__(self, symbol: str, start: str, end: str) -> None:
        self.symbol = symbol
        self.start = start
        self.end = end

    def get_bars(self) -> List[Bar]:
        import yfinance as yf  # imported lazily so the package works offline

        df = yf.download(
            self.symbol, start=self.start, end=self.end,
            progress=False, auto_adjust=True,
        )
        if df is None or df.empty:
            raise RuntimeError(f"No data returned for {self.symbol}")
        bars: List[Bar] = []
        for ts, row in df.iterrows():
            bars.append(
                Bar(
                    symbol=self.symbol,
                    day=ts.date() if hasattr(ts, "date") else ts,
                    open=float(row["Open"]),
                    high=float(row["High"]),
                    low=float(row["Low"]),
                    close=float(row["Close"]),
                    volume=float(row["Volume"]),
                )
            )
        return bars


def get_data_source(config: Dict) -> DataSource:
    """Factory: build a data source from the config's `data` section.

    Falls back to the synthetic source if `yfinance` is selected but the data
    cannot be fetched (e.g. no network), so a run always produces results.
    """
    data_cfg = config.get("data", {})
    source = data_cfg.get("source", "synthetic")

    if source == "yfinance":
        try:
            src = YFinanceDataSource(
                symbol=data_cfg.get("symbol", "SPY"),
                start=data_cfg.get("start", "2015-01-01"),
                end=data_cfg.get("end", "2023-12-31"),
            )
            # Probe: fetching happens in get_bars; caller handles the fallback.
            return src
        except Exception as e:  # pragma: no cover - defensive
            logger.warning("yfinance source unavailable (%s); using synthetic.", e)

    syn = data_cfg.get("synthetic", {})
    return SyntheticDataSource(
        symbol=data_cfg.get("symbol", "SYNTH"),
        n_days=syn.get("n_days", 2016),
        start_price=syn.get("start_price", 100.0),
        annual_drift=syn.get("annual_drift", 0.07),
        annual_vol=syn.get("annual_vol", 0.20),
        seed=syn.get("seed", 42),
        start=_parse_date(data_cfg.get("start", "2015-01-01")),
    )


def _parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def _align_business_day(d: date) -> date:
    while d.weekday() >= 5:  # Sat/Sun
        d += timedelta(days=1)
    return d


def _next_business_day(d: date) -> date:
    d += timedelta(days=1)
    return _align_business_day(d)
