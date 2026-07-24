import math

from sandbox.metrics import compute_metrics


def test_flat_equity_is_zero_risk_zero_return():
    m = compute_metrics([100.0] * 300)
    assert m.total_return == 0.0
    assert m.sharpe == 0.0
    assert m.max_drawdown == 0.0
    assert m.annual_vol == 0.0


def test_total_return():
    m = compute_metrics([100.0, 110.0, 121.0])
    assert math.isclose(m.total_return, 0.21, rel_tol=1e-9)


def test_max_drawdown():
    # Up to 120, down to 90 -> worst drawdown from the 120 peak is -25%.
    m = compute_metrics([100.0, 120.0, 90.0, 100.0])
    assert math.isclose(m.max_drawdown, -0.25, rel_tol=1e-9)


def test_positive_trend_has_positive_sharpe():
    equity = [100.0 * (1.01 ** i) for i in range(250)]  # steady +1%/day
    m = compute_metrics(equity)
    assert m.total_return > 0
    assert m.cagr > 0
    assert m.sharpe > 0
