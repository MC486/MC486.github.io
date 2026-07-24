from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class Portfolio:
    """Tracks cash and share positions and marks to market.

    Single-currency, long-only for the MVP, but structured to extend to
    multiple symbols later.
    """
    cash: float
    positions: Dict[str, float] = field(default_factory=dict)

    def shares(self, symbol: str) -> float:
        return self.positions.get(symbol, 0.0)

    def value(self, prices: Dict[str, float]) -> float:
        """Total mark-to-market equity given current prices."""
        equity = self.cash
        for symbol, qty in self.positions.items():
            equity += qty * prices.get(symbol, 0.0)
        return equity

    def apply_fill(self, symbol: str, delta_shares: float, price: float, cost: float) -> None:
        """Apply a trade: change position by delta_shares at price, paying cost."""
        self.cash -= delta_shares * price
        self.cash -= cost
        new_qty = self.shares(symbol) + delta_shares
        if abs(new_qty) < 1e-9:
            self.positions.pop(symbol, None)
        else:
            self.positions[symbol] = new_qty
