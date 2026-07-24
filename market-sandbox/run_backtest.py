#!/usr/bin/env python3
"""Run a backtest end-to-end: load data, run the strategy and a buy-and-hold
benchmark, print risk/return metrics, persist the run, and save an equity-curve
plot.

Usage:
    python run_backtest.py [--config config.yaml] [--out equity_curve.png]
"""
from __future__ import annotations

import argparse
import logging
import os
from typing import Dict, List

import yaml

from sandbox.engine import BacktestEngine
from sandbox.market import Bar, get_data_source
from sandbox.market.data import SyntheticDataSource
from sandbox.metrics import compute_metrics
from sandbox.persistence import RunRepository
from sandbox.strategy import BuyAndHoldStrategy, MACrossoverStrategy

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("run_backtest")


def load_config(path: str) -> Dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_bars(config: Dict) -> List[Bar]:
    """Load bars, falling back to synthetic data if the real source fails."""
    source = get_data_source(config)
    try:
        bars = source.get_bars()
        if bars:
            logger.info("Loaded %d bars for %s from %s.",
                        len(bars), bars[0].symbol, type(source).__name__)
            return bars
        raise RuntimeError("empty bar set")
    except Exception as e:
        logger.warning("Data source failed (%s); falling back to synthetic.", e)
        syn = config.get("data", {}).get("synthetic", {})
        fallback = SyntheticDataSource(
            symbol=config.get("data", {}).get("symbol", "SYNTH"),
            n_days=syn.get("n_days", 2016),
            start_price=syn.get("start_price", 100.0),
            annual_drift=syn.get("annual_drift", 0.07),
            annual_vol=syn.get("annual_vol", 0.20),
            seed=syn.get("seed", 42),
        )
        bars = fallback.get_bars()
        logger.info("Loaded %d synthetic bars.", len(bars))
        return bars


def build_strategy(config: Dict):
    strat_cfg = config.get("strategy", {})
    name = strat_cfg.get("name", "ma_crossover")
    if name == "ma_crossover":
        p = strat_cfg.get("ma_crossover", {})
        return MACrossoverStrategy(fast=p.get("fast", 20), slow=p.get("slow", 100))
    raise ValueError(f"unknown strategy: {name}")


def _fmt(m) -> str:
    d = m.as_dict()
    return (f"total_return={d['total_return']:+.1%}  cagr={d['cagr']:+.2%}  "
            f"sharpe={d['sharpe']:.2f}  max_dd={d['max_drawdown']:.1%}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "config.yaml"))
    parser.add_argument("--out", default="equity_curve.png")
    parser.add_argument("--db", default="sandbox_runs.db")
    args = parser.parse_args()

    config = load_config(args.config)
    tdays = config.get("evaluation", {}).get("trading_days_per_year", 252)
    bars = load_bars(config)

    engine = BacktestEngine(config)
    repo = RunRepository(args.db)

    # Strategy under test.
    strategy = build_strategy(config)
    strat_result = engine.run(bars, strategy)
    strat_metrics = compute_metrics(strat_result.equity, tdays)
    repo.save_run(strat_result, strat_metrics)

    # Benchmark: buy-and-hold.
    bench_result = engine.run(bars, BuyAndHoldStrategy())
    bench_metrics = compute_metrics(bench_result.equity, tdays)
    repo.save_run(bench_result, bench_metrics)

    print("\n=== Backtest Results (net of costs) ===")
    print(f"Symbol: {strat_result.symbol}   Bars: {strat_result.n_periods if hasattr(strat_result,'n_periods') else len(bars)}   "
          f"Starting cash: ${strat_result.starting_cash:,.0f}")
    print(f"\n{strategy.name:>22}: {_fmt(strat_metrics)}  trades={strat_result.n_trades}")
    print(f"{'buy_and_hold':>22}: {_fmt(bench_metrics)}  trades={bench_result.n_trades}")
    edge = strat_metrics.sharpe - bench_metrics.sharpe
    print(f"\nSharpe edge vs buy-and-hold: {edge:+.2f}")

    _save_plot(args.out, strat_result, bench_result, strategy.name)
    print(f"Equity curve saved to {args.out}\n")


def _save_plot(path: str, strat_result, bench_result, strat_name: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(strat_result.days, strat_result.equity, label=strat_name, linewidth=1.6)
    ax.plot(bench_result.days, bench_result.equity, label="buy_and_hold",
            linewidth=1.2, alpha=0.8, linestyle="--")
    ax.set_title(f"Equity Curve — {strat_result.symbol} (net of costs)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio value ($)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
