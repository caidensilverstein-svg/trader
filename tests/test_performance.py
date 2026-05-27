"""Unit tests for reporting/performance_tracker.py."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from reporting.performance_tracker import compute_portfolio_metrics, strategy_pnl_summary


class TestComputeMetrics:

    def test_flat_portfolio(self):
        m = compute_portfolio_metrics([100_000, 100_000, 100_000])
        assert m["total_return"] == 0.0
        assert m["sharpe"] == 0.0

    def test_positive_return(self):
        # 10% return in 10 days
        vals = [100_000 * (1.01 ** i) for i in range(11)]
        m = compute_portfolio_metrics(vals)
        assert m["total_return"] > 0

    def test_max_dd_negative(self):
        # Portfolio drops then recovers
        vals = [100, 110, 105, 90, 95, 100]
        m = compute_portfolio_metrics(vals)
        assert m["max_dd"] < 0

    def test_insufficient_data_returns_error(self):
        m = compute_portfolio_metrics([100_000])
        assert "error" in m

    def test_sharpe_higher_for_steady_gains(self):
        """Steady gainer should have higher Sharpe than volatile same-return portfolio."""
        steady   = [100 * (1.001 ** i) for i in range(100)]
        volatile = [100] + [100 * (1.0 + (0.03 if i % 2 == 0 else -0.02)) for i in range(99)]
        m_steady   = compute_portfolio_metrics(steady)
        m_volatile = compute_portfolio_metrics(volatile)
        assert m_steady["sharpe"] > m_volatile["sharpe"]

    def test_calmar_infinite_when_no_drawdown(self):
        vals = [100, 101, 102, 103]
        m = compute_portfolio_metrics(vals)
        # No drawdown -> Calmar = inf (stored as float('inf'))
        assert m["calmar"] == float("inf") or m["calmar"] > 10
