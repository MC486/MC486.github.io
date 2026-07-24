from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Optional

from ..events import Event, EventBus, EventType
from ..market.bar import Bar
from ..market.portfolio import Portfolio
from ..strategy.base import MarketView, Strategy


@dataclass
class BacktestResult:
    strategy_name: str
    symbol: str
    days: List[date] = field(default_factory=list)
    equity: List[float] = field(default_factory=list)
    n_trades: int = 0
    starting_cash: float = 0.0


class BacktestEngine:
    """Steps through bars one at a time and simulates a long-only portfolio.

    Lookahead safety is structural: a strategy only ever receives bars up to and
    including the current one, and any target it produces is executed at the
    NEXT bar's open — never at a price it already saw when deciding.
    """

    def __init__(self, config: Dict, event_bus: Optional[EventBus] = None) -> None:
        bt = config.get("backtest", {})
        self.starting_cash = float(bt.get("starting_cash", 10_000.0))
        self.commission_bps = float(bt.get("commission_bps", 1.0))
        self.slippage_bps = float(bt.get("slippage_bps", 5.0))
        self.bus = event_bus

    def run(self, bars: List[Bar], strategy: Strategy) -> BacktestResult:
        if not bars:
            raise ValueError("no bars to backtest")
        symbol = bars[0].symbol
        portfolio = Portfolio(cash=self.starting_cash)
        result = BacktestResult(
            strategy_name=strategy.name,
            symbol=symbol,
            starting_cash=self.starting_cash,
        )

        pending_target: Optional[Dict[str, float]] = None

        for t, bar in enumerate(bars):
            # 1) Execute the target decided on the previous bar, at this open.
            if pending_target is not None:
                if self._rebalance(portfolio, symbol, pending_target, bar.open):
                    result.n_trades += 1
                    self._emit(EventType.ORDER_FILLED, {"day": bar.day, "price": bar.open})
                pending_target = None

            # 2) Mark to market at this bar's close.
            equity = portfolio.value({symbol: bar.close})
            result.days.append(bar.day)
            result.equity.append(equity)
            self._emit(EventType.EQUITY_UPDATED, {"day": bar.day, "equity": equity})

            # 3) Let the strategy decide using history through this bar only.
            view = MarketView(
                symbol=symbol,
                history=bars[: t + 1],
                portfolio=portfolio,
                equity=equity,
            )
            pending_target = strategy.on_bar(view)

        self._emit(EventType.BACKTEST_ENDED, {"final_equity": result.equity[-1]})
        return result

    def _rebalance(self, portfolio: Portfolio, symbol: str,
                   target: Dict[str, float], price: float) -> bool:
        """Rebalance the single symbol to its target weight at `price`.

        Costs (commission + slippage) are charged proportional to the traded
        notional. Returns True if a trade occurred.
        """
        weight = max(0.0, min(1.0, float(target.get(symbol, 0.0))))
        equity_now = portfolio.value({symbol: price})
        if equity_now <= 0 or price <= 0:
            return False
        target_shares = (weight * equity_now) / price
        delta = target_shares - portfolio.shares(symbol)
        if abs(delta) < 1e-9:
            return False
        notional = abs(delta) * price
        cost = notional * (self.commission_bps + self.slippage_bps) / 10_000.0
        portfolio.apply_fill(symbol, delta, price, cost)
        return True

    def _emit(self, event_type: EventType, data: Dict) -> None:
        if self.bus is not None:
            self.bus.emit(Event(event_type, data))
