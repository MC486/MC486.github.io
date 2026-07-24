# Market Sandbox — Build Plan

A living design document: what we're building, why, what we reuse from the
origin project, and the phased roadmap with success criteria. This exists so we
plan and verify in thin vertical slices instead of stacking unverified layers
(the lesson from the word-game capstone).

> **Building it?** Read [`docs/BUILD_SPEC.md`](docs/BUILD_SPEC.md) — the
> execution-ready hand-off with current-state inventory, exact interface
> contracts, the origin→sandbox port map, and per-task acceptance tests. Any
> agent/engineer should be able to continue from the current point using it
> alone.

> Scope: a **paper-trading / research simulation**. No real orders, no financial
> advice. The goal is a rigorous, honest framework + analysis — not a money printer.

---

## 1. Vision & why

Build an event-driven sandbox that treats a market as a **turn-based game**: each
bar (day) is a turn, a decision policy chooses target portfolio weights, the
engine simulates fills/costs/P&L, and — eventually — a **learning ensemble** of
models adapts over time. Then let a human play the same historical tape against
the AI.

Why this over the word game:
- **Public, reproducible data** (anyone can rerun a backtest and get our curve).
- **Fast iteration** (thousands of bars replay in milliseconds).
- **No capital required** (we "keep track and pretend"), so we can run many
  portfolios and counterfactuals in parallel.
- **Heavier DS surface** (feature engineering, probabilistic models, RL,
  statistical evaluation) — a stronger CS+DS portfolio story.

The real constraint is not money or compute; it is **not fooling ourselves with a
finite history**. The architecture is designed so the backtest *cannot* cheat.

---

## 2. Design principles (honesty by construction)

1. **No lookahead (structural).** A strategy only ever receives bars up to and
   including "now"; targets execute at the **next** bar's open. It physically
   cannot see or trade on the future. — DONE in `engine/backtest.py`.
2. **Costs always modeled.** Commission + slippage on every fill. — DONE.
3. **Benchmark everything.** Report Sharpe / drawdown vs. buy-and-hold, not raw
   return. — DONE.
4. **Out-of-sample discipline.** A time-based holdout we never tune on;
   walk-forward evaluation. — PLANNED (Phase 3).
5. **Reproducible.** Deterministic seeded data; runs persisted. — DONE.
6. **Thin vertical slices.** Every phase runs end-to-end and is demoable before
   the next is started.

---

## 3. What we reuse from the origin project (honest status)

The genuinely valuable IP is the **ensemble decision logic** and the **four
models** — those are what we port. The thin scaffolding was reimplemented clean
(the originals were buggy and coupled to "words").

| Origin component | Plan | Status |
|---|---|---|
| Event bus (`GameEventManager`) | Reimplement clean | DONE (`events.py`) |
| Turn loop (`GameLoop`/`GameState`) | Reimplement as backtest loop | DONE (`engine/backtest.py`) |
| Repository pattern | Reimplement minimal, extend later | DONE (`persistence/`) |
| **`AIStrategy` ensemble**: candidate generation, weighted scoring, `_adjust_weights` online weight adaptation | **Port & adapt** — this is the core reusable | **TODO (Phase 2)** |
| **Markov / MCTS / Naive Bayes / Q-learning models** | **Port & repurpose** as trading signals | **TODO (Phase 2–3)** |
| `WordFrequencyAnalyzer` | Port as `FeatureEngine` | **TODO (Phase 1)** |
| Config + logging | Reimplement | DONE |

"Port & adapt" means: lift the algorithm/structure, replace the domain (a "word
candidate scored by models" becomes "a target weight proposed by models"), keep
the online weight-adaptation idea intact.

---

## 4. Target architecture

```
DataSource ── bars ──▶ BacktestEngine ──(MarketView, lookahead-safe)──▶ Strategy
                              │                                            │
                              │                                   ┌────────┴────────┐
                              │                                   │  EnsembleStrategy │
                              │                                   │   blends Models   │
                              │                                   └────────┬────────┘
                              │                            Model.suggest(view)│ (weight, confidence)
                              ▼                                              │
                      Portfolio (fills, costs)                     FeatureEngine (indicators)
                              │
                    EventBus ─┴─▶ Metrics + RunRepository (SQLite)
```

Core abstractions:
- **`Strategy.on_bar(view) -> {symbol: target_weight}`** — already exists.
- **`Model.suggest(view) -> (target_weight, confidence)`** and
  **`Model.update(reward)`** — to add (Phase 2). Same shape as a `Strategy`, so
  the ensemble is itself a `Strategy` composed of `Model`s.
- **`FeatureEngine`** — turns raw bars into features (returns, MAs, RSI,
  volatility, regime) that models consume.

---

## 5. Phased roadmap

Each phase is an independently demoable vertical slice with a success test.

### Phase 0 — Backtest MVP  ✅ DONE
- Lookahead-safe engine, costs, synthetic+yfinance data, MA-crossover +
  buy-and-hold, metrics vs benchmark, SQLite persistence, 8 tests.
- **Success:** `run_backtest.py` runs end-to-end and shows honest results vs
  buy-and-hold. ✔

### Phase 1 — Feature engine + more baselines
- Port `WordFrequencyAnalyzer` → `FeatureEngine` (returns, SMA/EMA, RSI,
  rolling vol, drawdown, discretized regime label).
- Add 1–2 more rule baselines (momentum, mean-reversion) to exercise features.
- **Why:** models are only as good as their inputs; build/validate features
  before models. **Success:** feature values match hand-computed fixtures.

### Phase 2 — Model interface + first learning model + ensemble
- Introduce `Model.suggest/update`; implement **Naive Bayes direction** and/or
  **Q-learning policy** (ported from origin, repurposed).
- Port `AIStrategy`'s ensemble: blend model outputs by weight, adapt weights
  from realized risk-adjusted reward.
- **Why:** this is the actual reuse of the valuable IP and the "learns over
  time" story. **Success (honesty bar):** on a **held-out** period the learner
  is evaluated fairly and reported vs buy-and-hold — winning is *not* required;
  not cheating is.

### Phase 3 — Rigorous evaluation
- Time-based train/test split + walk-forward; guard survivorship (fixed universe
  chosen at window start); richer report (turnover, hit rate, per-model
  attribution).
- Port **Markov (regime)** and **MCTS (scenario rollouts)** models.
- **Why:** credibility lives here. **Success:** a walk-forward report with
  out-of-sample curves and a written honest read of the results.

### Phase 4 — Multi-symbol + human-vs-AI UI
- Portfolio across several tickers; a small web UI (FastAPI + a chart) where a
  human trades the same tape against the AI.
- **Why:** brings back the "game" and makes the work legible at a glance.

---

## 6. Evaluation methodology (the anti-self-deception rules)

- **Holdout:** reserve the most recent ~30% of history; never tune on it.
- **Walk-forward:** fit on a rolling window, test on the next, roll forward.
- **Always net of costs**, always vs buy-and-hold, always risk-adjusted.
- **Report the losers too.** A strategy that loses honestly is a better artifact
  than one that "wins" via a leaked backtest.
- **Red flag protocol:** any result that looks too good triggers a lookahead/
  cost/survivorship audit before it's believed.

---

## 7. Risks & non-goals

- **Markets are efficient-ish at daily resolution.** Beating buy-and-hold net of
  costs out-of-sample is genuinely hard. Success = rigorous framework + honest
  analysis, not guaranteed alpha.
- **Overfitting a finite history** is the main threat → holdout + walk-forward.
- **Non-goals:** live trading, intraday/tick data (until daily is solid),
  brokerage integration, anything requiring real money.

---

## 8. Definition of done (portfolio framing)

A reviewer can clone it, run one command, and see: a lookahead-safe backtester
with a learning strategy ensemble, evaluated walk-forward out-of-sample vs
buy-and-hold, with clear equity/drawdown visuals and an honest written read of
what worked and what didn't.
