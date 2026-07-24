from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class PerformanceMetrics:
    total_return: float
    cagr: float
    annual_vol: float
    sharpe: float
    max_drawdown: float
    n_periods: int

    def as_dict(self) -> dict:
        return {
            "total_return": self.total_return,
            "cagr": self.cagr,
            "annual_vol": self.annual_vol,
            "sharpe": self.sharpe,
            "max_drawdown": self.max_drawdown,
            "n_periods": self.n_periods,
        }


def compute_metrics(equity: List[float], trading_days_per_year: int = 252) -> PerformanceMetrics:
    """Compute standard risk/return metrics from an equity curve.

    Sharpe and volatility are annualized assuming a risk-free rate of 0.
    """
    eq = np.asarray(equity, dtype=float)
    if eq.size < 2 or eq[0] <= 0:
        return PerformanceMetrics(0.0, 0.0, 0.0, 0.0, 0.0, int(eq.size))

    rets = np.diff(eq) / eq[:-1]
    total_return = float(eq[-1] / eq[0] - 1.0)

    n = eq.size
    years = (n - 1) / trading_days_per_year
    cagr = float((eq[-1] / eq[0]) ** (1.0 / years) - 1.0) if years > 0 else 0.0

    std = float(rets.std(ddof=1)) if rets.size > 1 else 0.0
    annual_vol = std * np.sqrt(trading_days_per_year)
    sharpe = (
        float(rets.mean() / std * np.sqrt(trading_days_per_year))
        if std > 0 else 0.0
    )

    running_max = np.maximum.accumulate(eq)
    drawdowns = eq / running_max - 1.0
    max_drawdown = float(drawdowns.min())

    return PerformanceMetrics(
        total_return=total_return,
        cagr=cagr,
        annual_vol=annual_vol,
        sharpe=sharpe,
        max_drawdown=max_drawdown,
        n_periods=int(n),
    )
