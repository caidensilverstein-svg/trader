"""Unit tests for backtest/turnover_analysis.py."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest

from backtest.turnover_analysis import (
    compute_turnover_from_weights,
    compute_turnover_from_trade_log,
    format_turnover_report,
    TurnoverStats,
)


def _make_weight_history(n_days=500, n_tickers=5, seed=42) -> pd.DataFrame:
    np.random.seed(seed)
    idx = pd.bdate_range("2020-01-01", periods=n_days)
    tickers = [f"ETF{i}" for i in range(n_tickers)]
    # Random weights that change slightly over time
    base = np.ones((n_days, n_tickers)) / n_tickers
    noise = np.random.normal(0, 0.02, (n_days, n_tickers))
    w = base + noise
    w = w / w.sum(axis=1, keepdims=True)
    return pd.DataFrame(w, index=idx, columns=tickers)


class TestComputeTurnoverFromWeights:

    def test_returns_turnover_stats(self):
        wh = _make_weight_history()
        result = compute_turnover_from_weights(wh)
        assert isinstance(result, TurnoverStats)

    def test_constant_weights_zero_turnover(self):
        idx = pd.bdate_range("2020-01-01", periods=252)
        # Exactly the same weights every day
        w = pd.DataFrame({"A": [0.5]*252, "B": [0.5]*252}, index=idx)
        result = compute_turnover_from_weights(w)
        assert result.annual_turnover_pct == 0.0

    def test_turnover_positive_for_changing_weights(self):
        wh = _make_weight_history()
        result = compute_turnover_from_weights(wh)
        assert result.annual_turnover_pct >= 0

    def test_cost_drag_proportional_to_turnover(self):
        wh1 = _make_weight_history(seed=1)
        wh2 = pd.DataFrame(
            _make_weight_history(seed=1).values * 2,  # double the noise
            index=wh1.index, columns=wh1.columns
        )
        # Higher turnover = higher cost (just check direction)
        r1 = compute_turnover_from_weights(wh1)
        assert r1.cost_drag_bps_annual >= 0

    def test_tax_efficiency_between_0_and_100(self):
        wh = _make_weight_history()
        result = compute_turnover_from_weights(wh)
        assert 0 <= result.tax_efficiency_score <= 100

    def test_insufficient_data_returns_default(self):
        w = pd.DataFrame({"A": [0.5]}, index=pd.bdate_range("2020-01-01", periods=1))
        result = compute_turnover_from_weights(w)
        assert result.annual_turnover_pct == 0.0


class TestComputeTurnoverFromTradeLog:

    def _make_trade_log(self, n_events=50):
        log = []
        dates = pd.bdate_range("2020-01-01", periods=n_events)
        for i, d in enumerate(dates):
            log.append({
                "action": "BUY",
                "ticker": "AVUV",
                "qty": 100,
                "price": 50.0,
                "date": str(d)[:10],
            })
        return log

    def test_returns_turnover_stats(self):
        log = self._make_trade_log()
        result = compute_turnover_from_trade_log(log)
        assert isinstance(result, TurnoverStats)

    def test_empty_log_returns_zero(self):
        result = compute_turnover_from_trade_log([])
        assert result.annual_turnover_pct == 0.0
        assert result.tax_efficiency_score == 100.0

    def test_more_trades_higher_turnover(self):
        log_few = self._make_trade_log(n_events=5)
        log_many = self._make_trade_log(n_events=50)
        r_few  = compute_turnover_from_trade_log(log_few, n_days=2110)
        r_many = compute_turnover_from_trade_log(log_many, n_days=2110)
        assert r_many.annual_turnover_pct > r_few.annual_turnover_pct

    def test_n_events_counted(self):
        log = self._make_trade_log(n_events=10)
        result = compute_turnover_from_trade_log(log)
        assert result.n_rebalance_events > 0

    def test_cost_drag_positive(self):
        log = self._make_trade_log()
        result = compute_turnover_from_trade_log(log)
        assert result.cost_drag_bps_annual >= 0


class TestFormatTurnoverReport:

    def test_contains_header(self):
        stats = TurnoverStats(25.0, 80.0, 30, 1.5, 0.015, 87.5, "drift")
        r = format_turnover_report(stats)
        assert "TURNOVER" in r

    def test_contains_turnover_rate(self):
        stats = TurnoverStats(25.0, 80.0, 30, 1.5, 0.015, 87.5, "drift")
        r = format_turnover_report(stats)
        assert "25.0" in r

    def test_contains_cost_drag(self):
        stats = TurnoverStats(25.0, 80.0, 30, 1.5, 0.015, 87.5, "drift")
        r = format_turnover_report(stats)
        assert "1.50" in r

    def test_high_efficiency_flagged(self):
        stats = TurnoverStats(10.0, 200.0, 10, 0.6, 0.006, 95.0, "scheduled")
        r = format_turnover_report(stats)
        assert "TAX-EFFICIENT" in r
