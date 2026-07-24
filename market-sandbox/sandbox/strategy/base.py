from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List

from ..market.bar import Bar
from ..market.portfolio import Portfolio


@dataclass
class MarketView:
    """Everything a strategy is allowed to see when deciding on a bar.

    Critically, `history` only contains bars up to and including the current
    bar. The engine never places future bars here, so a strategy physically
    cannot look ahead.
    """
    symbol: str
    history: List[Bar]          # bars[0..t] inclusive
    portfolio: Portfolio
    equity: float               # marked-to-market equity at the current close

    @property
    def closes(self) -> List[float]:
        return [b.close for b in self.history]

    @property
    def current(self) -> Bar:
        return self.history[-1]


class Strategy(ABC):
    """A decision policy that maps a MarketView to target portfolio weights.

    Returns a mapping of symbol -> target weight in [0, 1] (long-only MVP).
    This is the same shape a future model *ensemble* would produce, so the
    ensemble can drop in behind this exact interface later.
    """

    name: str = "strategy"

    @abstractmethod
    def on_bar(self, view: MarketView) -> Dict[str, float]:
        raise NotImplementedError

    def update(self, reward: float) -> None:
        """Optional online-learning hook; no-op for rule-based strategies."""
        return None
