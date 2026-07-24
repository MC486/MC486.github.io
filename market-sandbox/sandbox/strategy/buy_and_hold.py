from __future__ import annotations

from typing import Dict

from .base import MarketView, Strategy


class BuyAndHoldStrategy(Strategy):
    """Benchmark: buy on the first possible bar and hold to the end."""

    def __init__(self) -> None:
        self.name = "buy_and_hold"

    def on_bar(self, view: MarketView) -> Dict[str, float]:
        return {view.symbol: 1.0}
