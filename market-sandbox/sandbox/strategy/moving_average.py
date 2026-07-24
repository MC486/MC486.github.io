from __future__ import annotations

from typing import Dict

from .base import MarketView, Strategy


class MACrossoverStrategy(Strategy):
    """Classic trend-following baseline.

    Hold the asset (target weight 1.0) when the fast moving average is above the
    slow moving average, otherwise stay in cash (0.0).
    """

    def __init__(self, fast: int = 20, slow: int = 100) -> None:
        if fast >= slow:
            raise ValueError("fast window must be shorter than slow window")
        self.fast = fast
        self.slow = slow
        self.name = f"ma_crossover({fast}/{slow})"

    def on_bar(self, view: MarketView) -> Dict[str, float]:
        closes = view.closes
        if len(closes) < self.slow:
            return {view.symbol: 0.0}  # not enough history yet
        fast_ma = sum(closes[-self.fast:]) / self.fast
        slow_ma = sum(closes[-self.slow:]) / self.slow
        return {view.symbol: 1.0 if fast_ma > slow_ma else 0.0}
