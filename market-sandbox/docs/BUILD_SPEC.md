# Market Sandbox — Execution-Ready Build Spec

This is the hand-off document. If you are a new engineer or agent, you should be
able to read this plus [`../PLAN.md`](../PLAN.md) and build the project to
completion without asking questions. It states exactly what exists, what to add,
the precise interfaces, the origin→sandbox port map, and per-task acceptance
tests.

- **Read order:** `PLAN.md` (why/roadmap) → this file (how, exactly).
- **Current phase:** Phase 0 is DONE. **You are starting Phase 1.**
- **Golden rule:** every change ships as a thin vertical slice that runs
  end-to-end with passing tests before the next task is started.

---

## 0. TL;DR for the next agent

1. Set up env (Section 1), run `pytest` (expect 8 passing) and `python run_backtest.py` (expect metrics + `equity_curve.png`).
2. Do tasks in order (Section 7). Each task lists: files to touch, exact behavior, and the test that proves it's done.
3. Never break the invariants in Section 3 (especially **no lookahead**).
4. Keep `PLAN.md`'s reuse table and phase checkboxes updated as you finish tasks.

---

## 1. Setup & how to run

```bash
cd market-sandbox
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest -q                       # expect: 8 passed
python run_backtest.py          # prints metrics vs buy-and-hold, writes equity_curve.png
```

- **Python:** 3.10+.
- **Network caveat:** `yfinance` needs internet. If it's blocked, `run_backtest.py`
  automatically falls back to the deterministic synthetic source, so the demo
  always works. Tests must **never** require network (use `SyntheticDataSource`).
- **Determinism:** synthetic data is seeded; identical config ⇒ identical output.

---

## 2. Current-state inventory (Phase 0, DONE)

Exact public surface as it exists today. Do not change these signatures without
updating callers and this doc.

### `sandbox/events.py`
- `EventType(Enum)`: `BAR_CLOSED, ORDER_FILLED, EQUITY_UPDATED, BACKTEST_ENDED`.
- `Event(type: EventType, data: dict)`.
- `EventBus`: `.subscribe(event_type, listener)`, `.emit(event)`. Synchronous.

### `sandbox/market/bar.py`
- `Bar(symbol: str, day: date, open, high, low, close, volume: float)` — frozen dataclass.

### `sandbox/market/portfolio.py`
- `Portfolio(cash: float, positions: dict[str, float])`.
- `.shares(symbol) -> float`, `.value(prices: dict) -> float`,
  `.apply_fill(symbol, delta_shares, price, cost) -> None`.

### `sandbox/market/data.py`
- `DataSource(ABC)`: `.get_bars() -> list[Bar]` (ascending by day).
- `SyntheticDataSource(symbol, n_days, start_price, annual_drift, annual_vol, seed, start)` — deterministic GBM, offline.
- `YFinanceDataSource(symbol, start, end)` — real data, lazy-imports yfinance.
- `get_data_source(config: dict) -> DataSource` — factory from `config["data"]`.

### `sandbox/strategy/base.py`
- `MarketView(symbol, history: list[Bar], portfolio: Portfolio, equity: float)`;
  props `.closes -> list[float]`, `.current -> Bar`. **`history` is bars[0..t]
  inclusive — the ONLY market data a strategy may read.**
- `Strategy(ABC)`: attr `name`; `.on_bar(view) -> dict[symbol, target_weight]`
  (weights in [0,1], long-only MVP); `.update(reward: float) -> None` (default no-op).

### `sandbox/strategy/moving_average.py`, `buy_and_hold.py`
- `MACrossoverStrategy(fast=20, slow=100)`, `BuyAndHoldStrategy()`.

### `sandbox/engine/backtest.py`
- `BacktestEngine(config: dict, event_bus: EventBus | None = None)`.
- `.run(bars, strategy) -> BacktestResult`.
- `BacktestResult(strategy_name, symbol, days: list[date], equity: list[float], n_trades: int, starting_cash: float)`.
- **Execution model:** decision made on bar `t` (sees bars[0..t]) executes at
  bar `t+1`'s **open**. Costs = `|Δshares|·price·(commission_bps+slippage_bps)/1e4`.

### `sandbox/metrics/performance.py`
- `compute_metrics(equity, trading_days_per_year=252) -> PerformanceMetrics`.
- `PerformanceMetrics(total_return, cagr, annual_vol, sharpe, max_drawdown, n_periods)`, `.as_dict()`.

### `sandbox/persistence/repository.py`
- `RunRepository(db_path)`: `.save_run(result, metrics) -> run_id`. Tables `runs`, `equity_points`.

### `run_backtest.py`
- `load_config`, `load_bars` (with synthetic fallback), `build_strategy`, `main`. Writes an equity-curve PNG.

### `config.yaml` sections
- `data` (source/symbol/start/end/synthetic), `backtest` (cash/commission_bps/slippage_bps), `strategy` (name + params), `evaluation` (trading_days_per_year, holdout_fraction).

### Tests (8, all passing)
- `tests/test_backtest.py`: no-lookahead, determinism, flat-strategy-preserves-cash, costs-reduce-equity.
- `tests/test_metrics.py`: flat/known-return/max-drawdown/positive-trend.

---

## 3. Invariants & conventions (do not violate)

1. **No lookahead.** Any new component that consumes market data takes
   `history: list[Bar]` (bars[0..t]) or `MarketView`, never the full series and
   never future bars. Feature computation at `t` uses only data through `t`.
2. **Costs & benchmark.** Never report return without costs; always compare to
   buy-and-hold on risk-adjusted terms.
3. **Determinism.** All randomness takes a `seed`. Tests never touch the network.
4. **Interfaces first.** New models implement the `Model` interface (Section 4);
   the ensemble is itself a `Strategy`.
5. **Style.** Type hints everywhere; docstrings that state intent/constraints,
   not narration. Small modules. `from __future__ import annotations` at top.
6. **Testing bar.** Every task adds/updates tests and leaves `pytest` green.
   Model tests prove *learning a known synthetic signal* — NOT "beating the
   market" (that is explicitly not required).
7. **Git.** One feature branch + PR per task/phase; PR body states what changed
   and shows the passing test count. Keep the app runnable on every commit.

---

## 4. Contracts to ADD (copy these signatures exactly)

### 4a. `sandbox/features/engine.py` (Phase 1)
```python
from dataclasses import dataclass
from datetime import date

@dataclass(frozen=True)
class Features:
    day: date
    close: float
    ret_1d: float        # (close_t - close_{t-1}) / close_{t-1}, 0.0 on first bar
    sma_fast: float
    sma_slow: float
    ema: float
    rsi_14: float        # 0..100; 50.0 until enough data
    vol_20: float        # rolling std of daily returns over 20 bars (raw, not annualized)
    drawdown: float      # current drawdown from running max of close, <= 0
    regime: int          # discretized state id in [0, n_regimes)

class FeatureEngine:
    def __init__(self, config: dict): ...
    def compute(self, history: list["Bar"]) -> Features:
        """Compute features from bars[0..t]. MUST be lookahead-safe."""
    def state_hash(self, f: Features) -> str:
        """Discretize features into a small categorical state for RL/Markov."""
```
Config keys (add under `features:` in `config.yaml`): `sma_fast`, `sma_slow`,
`ema_span`, `rsi_period`, `vol_window`, `n_regimes`.

### 4b. `sandbox/models/base.py` (Phase 2)
```python
from dataclasses import dataclass

@dataclass
class Signal:
    weight: float        # desired target weight in [0,1]
    confidence: float    # in [0,1]; ensemble uses this to weight the vote

@dataclass
class Transition:
    features_prev: Features
    weight: float        # action actually taken last bar (blended target)
    reward: float        # realized reward attributable to that action
    features_next: Features

class Model(ABC):
    name: str
    def suggest(self, view: MarketView, features: Features) -> Signal: ...
    def update(self, transition: Transition) -> None:  # default no-op
        return None
```

### 4c. `sandbox/strategy/ensemble.py` (Phase 2)
```python
class EnsembleStrategy(Strategy):
    """Blends several Models into one target weight and adapts their weights
    online from realized reward. This is the ported AIStrategy 'muscle'."""
    def __init__(self, models: list[Model], feature_engine: FeatureEngine,
                 config: dict): ...
    def on_bar(self, view: MarketView) -> dict[str, float]: ...
```
Required behavior of `on_bar` (exact order each bar):
1. `f = feature_engine.compute(view.history)`.
2. If a previous decision exists, compute realized reward (Section 5), build a
   `Transition`, call `model.update(transition)` for each model, then
   `_adjust_weights(...)`.
3. Each `model.suggest(view, f)` → `Signal`.
4. Blend: `target = Σ(w_m · conf_m · signal_m.weight) / Σ(w_m · conf_m)`
   (fallback to simple mean if denominator ≈ 0); clip to [0,1].
5. Stash `(f, target)` as the pending decision; return `{symbol: target}`.

`model_weights` init: equal, or from `config["ensemble"]["weights"]`. Persist
per-model weight history for later attribution.

---

## 5. Reward & timing spec (precise, no engine change needed)

- `EnsembleStrategy` tracks `last_equity` across `on_bar` calls.
- On bar `t` (t ≥ 1): `realized_return = (view.equity - last_equity) / last_equity`.
  This is the portfolio return over the bar just closed, which is the outcome of
  the **previous** bar's decision (executed at this bar's open). Attribute it to
  the previous decision.
- **Reward for learning:** `reward = realized_return` for v1. Optional upgrade
  (config flag `reward: sharpe_like`): `reward = realized_return - λ·max(0, -realized_return)`
  to emphasize downside, or a differential-Sharpe increment. Keep v1 simple.
- Update `last_equity = view.equity` at the end of each `on_bar`.
- Per-model credit in `_adjust_weights` (ported idea): after realized `reward`,
  for each model compute `alignment = sign(reward) · (signal_prev_m.weight - 0.5)`;
  nudge that model's weight by `+η·alignment` (η small, e.g. 0.02), floor each
  weight at a small min (e.g. 0.05), renormalize to sum 1. Store `signal_prev_m`
  from the previous bar.

---

## 6. Origin → sandbox porting map (specific)

Origin = the word-game repo (`ai/…`). "Port & adapt" = lift the algorithm, swap
the domain, keep the learning idea. Do NOT copy word-specific coupling.

| Origin file / symbol | Reusable idea | Sandbox target | Adaptation |
|---|---|---|---|
| `ai/strategy/ai_strategy.py::AIStrategy._generate_candidates/_score_candidates/_select_best_word` | candidate → score → select | `strategy/ensemble.py::EnsembleStrategy.on_bar` | "candidate word" → "target weight from a model"; blend by weight·confidence |
| `AIStrategy._initialize_weights/_adjust_weights/model_weights` | **online weight adaptation** | `EnsembleStrategy._adjust_weights` | credit models by alignment with realized reward (Section 5) |
| `ai/word_analysis.py::WordFrequencyAnalyzer` (`analyze_word_list`, `get_word_score`, freq tables) | feature/statistics engine | `features/engine.py::FeatureEngine` | letter freqs → price features (SMA/EMA/RSI/vol/regime) |
| `ai/models/q_learning_model.py::QLearningAgent` (`q_table`, epsilon-greedy, `choose_action`, `update`) | tabular RL policy | `models/q_learning.py::QLearningModel` | state = `FeatureEngine.state_hash`; actions = {flat=0.0, invested=1.0}; reward = Section 5 |
| `ai/models/naive_bayes.py::NaiveBayes` (probability tables, update) | probabilistic classifier | `models/naive_bayes.py::NaiveBayesDirectionModel` | classify next-bar direction (up/down) from discretized features; weight = P(up) |
| `ai/models/markov_chain.py::MarkovChain` (`_build_transition_matrix`, transitions) | sequence/regime transitions | `models/markov.py::MarkovRegimeModel` | states = discretized regimes; learn transitions; weight from P(next regime favorable) |
| `ai/models/mcts.py::MCTS` (UCB1, rollouts) | scenario simulation | `models/mcts.py::MCTSRolloutModel` | bootstrap-resample recent returns to simulate paths; weight = argmax expected risk-adj reward |
| `database/repositories/base_repository.py` + `repository_manager.py` | repository pattern | extend `persistence/` | add `bars` cache + per-model weight/attribution tables when needed |

Known origin pitfalls to NOT repeat (learned this session): don't declare schema
columns a repo never uses; keep the model's public method that produces an action
actually implemented (in the origin only Markov had `get_suggestion` — here every
`Model` MUST implement `suggest`); seed all randomness; and keep one integration
test that runs the whole pipeline.

---

## 7. Task list (do in order; each is a shippable slice)

### Phase 1 — Feature engine + baselines
- **T1.1 FeatureEngine.** Create `sandbox/features/engine.py` per §4a. Add
  `features:` config block.
  - *Acceptance:* `tests/test_features.py` — SMA/EMA/RSI/vol/drawdown match
    hand-computed values on a fixed input; `compute(history[:k])` never reads
    beyond index `k-1` (assert by constructing history where a future bar would
    change the result and confirming it doesn't).
- **T1.2 Feature-driven baselines.** Add `MomentumStrategy` and
  `MeanReversionStrategy` in `sandbox/strategy/` using `FeatureEngine`.
  - *Acceptance:* both run in a backtest; deterministic; a test asserts momentum
    goes long after sustained up-moves on a rigged series.

### Phase 2 — Model interface + first learner + ensemble
- **T2.1 Model/Signal/Transition** per §4b in `sandbox/models/base.py`.
- **T2.2 First model.** Implement `QLearningModel` (recommended first) OR
  `NaiveBayesDirectionModel` per the port map.
  - *Acceptance:* on a synthetic series with a *learnable* pattern (e.g.
    mean-reverting price, seed fixed), the model's out-of-sample weighting yields
    higher reward than a random policy. Test asserts learning (post-training
    behavior differs from untrained in the expected direction).
- **T2.3 EnsembleStrategy** per §4c and §5.
  - *Acceptance:* `tests/test_ensemble.py` — with two rigged models (one always
    right, one always wrong on a synthetic series), the ensemble's weight on the
    good model **increases** over time and ends higher than the bad model's.
- **T2.4 Wire into `run_backtest.py`.** `build_strategy` can construct the
  ensemble from config; `run_backtest.py` still runs end-to-end with an artifact.

### Phase 3 — Rigorous evaluation + remaining models
- **T3.1 WalkForwardEvaluator** in `sandbox/eval/walkforward.py`: split bars into
  rolling train/test windows + a final untouched holdout; run and aggregate OOS
  metrics; write a report (table + OOS equity curve).
  - *Acceptance:* `tests/test_walkforward.py` — windows are non-overlapping in
    the tested sense, holdout is never passed to any `update()` during "training",
    and the evaluator returns per-window + aggregate metrics.
- **T3.2 `evaluate.py` entry point.** One command → walk-forward report vs
  buy-and-hold, saved plot + printed table.
- **T3.3 Port `MarkovRegimeModel` and `MCTSRolloutModel`.** Add to ensemble.
  - *Acceptance:* each learns/behaves correctly on a synthetic fixture; ensemble
    still runs; report regenerates.

### Phase 4 — Multi-symbol + human-vs-AI UI
- **T4.1 Multi-symbol portfolio.** Generalize engine/portfolio to N symbols with
  per-symbol target weights summing to ≤ 1.
- **T4.2 Web UI (FastAPI + a chart).** Step the same historical tape; a human
  enters trades and races the AI's equity curve.

---

## 8. Evaluation harness spec (Phase 3 detail)

- **Split:** given N bars and `holdout_fraction` (config), the last fraction is
  the untouched holdout. The remainder is used for walk-forward: train window of
  `W` bars, test window of `T` bars, step by `T`.
- **"Training"** = letting learning models observe transitions and adapt
  (`update`) on the train window only. During test windows and the holdout,
  models may `suggest` but must not `update` (config flag `frozen=True` while
  evaluating), so reported performance is genuinely out-of-sample.
- **Report:** per-window and aggregate `total_return, cagr, sharpe, max_drawdown,
  turnover, hit_rate` for strategy vs buy-and-hold; save OOS equity curve.
- **Red-flag audit:** if OOS Sharpe is implausibly high, re-check lookahead,
  costs, and that the holdout was truly frozen.

---

## 9. Definition of done (project)

A reviewer clones the repo, runs `pip install -r requirements.txt` then one
command (`python evaluate.py`), and sees: a lookahead-safe backtest with a
learning model ensemble, evaluated walk-forward out-of-sample vs buy-and-hold,
with equity/drawdown visuals and an honest written read of what worked and what
didn't. `pytest` is green. `PLAN.md` phase checkboxes are all ticked.

Per-task DoD: code + tests + `pytest` green + app still runs end-to-end +
`PLAN.md` updated.

---

## 10. Anti-hiccup checklist (read before you start)

- [ ] `pytest` is green before you begin (8 passing).
- [ ] You added `from __future__ import annotations` to new modules.
- [ ] New data-consuming code takes `history`/`MarketView`, never future bars.
- [ ] Randomness is seeded; no test hits the network.
- [ ] You compared against buy-and-hold, net of costs.
- [ ] Model tests prove learning a synthetic signal, not beating the market.
- [ ] You updated `PLAN.md` (reuse table + phase checkbox) and this file if a
      contract changed.
- [ ] The app still runs end-to-end and (if UI/plots) you captured an artifact.
