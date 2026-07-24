# Market Sandbox

A lookahead-safe backtesting sandbox that treats a market as a **turn-based game**:
each bar (day) is a turn, the strategy chooses target portfolio weights, and the
engine simulates fills, costs, and mark-to-market P&L. It reuses the
architectural "bones" of an event-driven engine, a pluggable strategy layer, and
a repository persistence layer.

> This is a **simulation / research** project (paper trading only). It is not
> financial advice and executes no real trades.

## Why it's honest by construction
- **No lookahead:** a strategy only ever receives bars up to and including the
  current one, and any target it produces is executed at the **next** bar's open
  — never at a price it already saw when deciding.
- **Costs modeled:** every fill pays commission + slippage (basis points).
- **Reproducible:** the default synthetic data source is deterministic (seeded),
  so a run is fully repeatable offline. Real data (via `yfinance`) can be swapped
  in through `config.yaml`, and the runner falls back to synthetic if the network
  is unavailable.
- **Benchmarked:** every run is compared against buy-and-hold on risk-adjusted
  terms (Sharpe, max drawdown), not raw return.

## Quick start
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python run_backtest.py            # uses config.yaml (synthetic data by default)
```
This prints metrics vs. buy-and-hold and writes `equity_curve.png`.

To use real data, set `data.source: yfinance` and a real `symbol` in `config.yaml`.

## Layout
```
sandbox/
  events.py            # pub/sub event bus
  market/              # Bar, Portfolio, data sources (synthetic + yfinance)
  strategy/            # Strategy interface + MA-crossover + buy-and-hold baselines
  engine/backtest.py   # lookahead-safe, cost-aware backtest loop
  metrics/             # Sharpe / CAGR / drawdown
  persistence/         # SQLite repository for runs + equity curves
run_backtest.py        # entry point
tests/                 # lookahead-safety, determinism, cost, and metric tests
```

## Roadmap (reusing the "bones")
The `Strategy.on_bar(view) -> target weights` interface is deliberately the same
shape a **model ensemble** would produce, so the next steps slot in cleanly:
1. A `Model` interface (`suggest`/`update`) and an ensemble strategy that blends
   several models with online weight adaptation based on realized reward.
2. Concrete models: Naive Bayes (direction), Markov (regime), MCTS (scenario
   rollouts), Q-learning (policy) — the same families from the origin project,
   repurposed as trading signals.
3. Walk-forward / out-of-sample evaluation and a multi-symbol portfolio.
