from typing import Dict, List

from sandbox.engine import BacktestEngine
from sandbox.market.data import SyntheticDataSource
from sandbox.strategy.base import MarketView, Strategy


def _config(commission=0.0, slippage=0.0, cash=10_000.0) -> Dict:
    return {"backtest": {"starting_cash": cash,
                         "commission_bps": commission,
                         "slippage_bps": slippage}}


def _bars(seed=1, n=300):
    return SyntheticDataSource(symbol="T", n_days=n, seed=seed).get_bars()


class RecordingStrategy(Strategy):
    """Records what it was shown each bar so we can assert lookahead safety."""
    def __init__(self):
        self.name = "recording"
        self.observations = []

    def on_bar(self, view: MarketView) -> Dict[str, float]:
        self.observations.append(
            {
                "len": len(view.history),
                "current_day": view.current.day,
                "last_day": view.history[-1].day,
            }
        )
        return {view.symbol: 0.0}


class FlatStrategy(Strategy):
    name = "flat"
    def on_bar(self, view: MarketView) -> Dict[str, float]:
        return {view.symbol: 0.0}


def test_no_lookahead():
    """On bar t the strategy must see exactly t+1 bars, ending at the current bar."""
    bars = _bars()
    strat = RecordingStrategy()
    BacktestEngine(_config()).run(bars, strat)

    assert len(strat.observations) == len(bars)
    prev_day = None
    for i, obs in enumerate(strat.observations):
        assert obs["len"] == i + 1                      # only history through now
        assert obs["current_day"] == bars[i].day        # current == this bar
        assert obs["last_day"] == bars[i].day            # newest bar is the current one
        if prev_day is not None:
            assert obs["current_day"] > prev_day         # time only moves forward
        prev_day = obs["current_day"]


def test_deterministic():
    """Same seed and config -> identical equity curve (reproducibility)."""
    cfg = _config()
    r1 = BacktestEngine(cfg).run(_bars(seed=7), FlatStrategy())
    r2 = BacktestEngine(cfg).run(_bars(seed=7), FlatStrategy())
    assert r1.equity == r2.equity


def test_flat_strategy_preserves_cash():
    """Never investing => no trades, no costs, equity stays at starting cash."""
    result = BacktestEngine(_config(cash=10_000.0)).run(_bars(), FlatStrategy())
    assert result.n_trades == 0
    assert all(abs(e - 10_000.0) < 1e-6 for e in result.equity)


def test_costs_reduce_final_equity():
    """Buy-and-hold with costs should finish below the frictionless version."""
    from sandbox.strategy import BuyAndHoldStrategy
    bars = _bars(seed=3)
    free = BacktestEngine(_config(commission=0.0, slippage=0.0)).run(bars, BuyAndHoldStrategy())
    costed = BacktestEngine(_config(commission=1.0, slippage=5.0)).run(bars, BuyAndHoldStrategy())
    assert costed.equity[-1] < free.equity[-1]
