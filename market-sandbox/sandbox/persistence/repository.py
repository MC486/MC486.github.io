from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from typing import Iterator

from ..engine.backtest import BacktestResult
from ..metrics.performance import PerformanceMetrics


class RunRepository:
    """Persists backtest runs and their equity curves to SQLite.

    Mirrors the repository-pattern "bone": all persistence goes through a small
    typed API rather than scattered SQL.
    """

    def __init__(self, db_path: str = "sandbox_runs.db") -> None:
        self.db_path = db_path
        self._init_schema()

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self._conn() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    strategy TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    starting_cash REAL NOT NULL,
                    final_equity REAL NOT NULL,
                    total_return REAL NOT NULL,
                    cagr REAL NOT NULL,
                    sharpe REAL NOT NULL,
                    max_drawdown REAL NOT NULL,
                    n_trades INTEGER NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS equity_points (
                    run_id INTEGER NOT NULL,
                    day TEXT NOT NULL,
                    equity REAL NOT NULL,
                    FOREIGN KEY (run_id) REFERENCES runs(id) ON DELETE CASCADE
                );
                """
            )

    def save_run(self, result: BacktestResult, metrics: PerformanceMetrics) -> int:
        with self._conn() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO runs (strategy, symbol, starting_cash, final_equity,
                                  total_return, cagr, sharpe, max_drawdown, n_trades)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    result.strategy_name,
                    result.symbol,
                    result.starting_cash,
                    result.equity[-1] if result.equity else 0.0,
                    metrics.total_return,
                    metrics.cagr,
                    metrics.sharpe,
                    metrics.max_drawdown,
                    result.n_trades,
                ),
            )
            run_id = cur.lastrowid
            cur.executemany(
                "INSERT INTO equity_points (run_id, day, equity) VALUES (?, ?, ?)",
                [(run_id, d.isoformat(), e) for d, e in zip(result.days, result.equity)],
            )
            return run_id
