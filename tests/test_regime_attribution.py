"""Unit tests for core/regime_attribution.py."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest

from core.regime_attribution import (
    compute_regime_attribution,
    regime_attribution_summary,
    format_regime_attribution,
    RegimeStats,
)


def _make_data(n=500, seed=42):
    np.random.seed(seed)
    idx = pd.bdate_range("2020-01-01", periods=n)
    rets = np.random.normal(0.0003, 0.01, n)
    equity = pd.Series(100_000 * np.cumprod(1 + rets), index=idx)

    # Alternate between BULL and BEAR in 50-day blocks
    regimes_list = []
    for i in range(n):
        block = i // 50
        regimes_list.append("BULL" if block % 2 == 0 else "BEAR")
    regime = pd.Series(regimes_list, index=idx)
    return equity, regime


class TestComputeRegimeAttribution:

    def test_returns_list_of_regime_stats(self):
        eq, reg = _make_data()
        result = compute_regime_attribution(eq, reg)
        assert isinstance(result, list)
        assert all(isinstance(s, RegimeStats) for s in result)

    def test_two_regimes_detected(self):
        eq, reg = _make_data()
        result = compute_regime_attribution(eq, reg)
        assert len(result) == 2

    def test_regime_names_preserved(self):
        eq, reg = _make_data()
        result = compute_regime_attribution(eq, reg)
        names = {s.regime for s in result}
        assert "BULL" in names
        assert "BEAR" in names

    def test_pct_time_sums_to_100(self):
        eq, reg = _make_data()
        result = compute_regime_attribution(eq, reg)
        total = sum(s.pct_time for s in result)
        assert abs(total - 100.0) < 2.0  # allow small rounding

    def test_n_days_sums_to_total(self):
        eq, reg = _make_data()
        result = compute_regime_attribution(eq, reg)
        total_days = sum(s.n_days for s in result)
        assert abs(total_days - 500) <= 5

    def test_sorted_by_cumulative_return_descending(self):
        eq, reg = _make_data()
        result = compute_regime_attribution(eq, reg)
        returns = [s.cumulative_return for s in result]
        assert returns == sorted(returns, reverse=True)

    def test_max_drawdown_non_positive(self):
        eq, reg = _make_data()
        for s in compute_regime_attribution(eq, reg):
            assert s.max_drawdown <= 0

    def test_n_episodes_positive(self):
        eq, reg = _make_data()
        for s in compute_regime_attribution(eq, reg):
            assert s.n_episodes >= 1

    def test_avg_duration_positive(self):
        eq, reg = _make_data()
        for s in compute_regime_attribution(eq, reg):
            assert s.avg_duration > 0

    def test_five_regimes_handled(self):
        n = 500
        idx = pd.bdate_range("2020-01-01", periods=n)
        np.random.seed(1)
        equity = pd.Series(100_000 * np.cumprod(1 + np.random.normal(0.0003, 0.01, n)), index=idx)
        regime_labels = ["BULL", "MILD_BULL", "SIDEWAYS", "BEAR", "BEAR_CRISIS"]
        regime = pd.Series([regime_labels[i // 100] for i in range(n)], index=idx)
        result = compute_regime_attribution(equity, regime)
        assert len(result) == 5

    def test_insufficient_data_returns_empty(self):
        eq = pd.Series([100, 101], index=pd.bdate_range("2020-01-01", periods=2))
        reg = pd.Series(["BULL", "BULL"], index=eq.index)
        result = compute_regime_attribution(eq, reg)
        assert result == []

    def test_bull_drift_higher_return_than_bear(self):
        n = 600
        idx = pd.bdate_range("2020-01-01", periods=n)
        np.random.seed(55)
        bull_rets = np.random.normal(0.001, 0.005, n)
        bear_rets = np.random.normal(-0.001, 0.005, n)
        # First 300 days BULL, last 300 BEAR
        rets = np.where(np.arange(n) < 300, bull_rets, bear_rets)
        equity = pd.Series(100_000 * np.cumprod(1 + rets), index=idx)
        regime = pd.Series(["BULL"] * 300 + ["BEAR"] * 300, index=idx)
        result = compute_regime_attribution(equity, regime)
        bull_stat = next(s for s in result if s.regime == "BULL")
        bear_stat = next(s for s in result if s.regime == "BEAR")
        assert bull_stat.ann_return > bear_stat.ann_return


class TestRegimeAttributionSummary:

    def _make_stats(self):
        return [
            RegimeStats("BULL", 300, 60.0, 20.0, 8.0, 0.8, -5.0, 3, 100.0),
            RegimeStats("MILD_BULL", 100, 20.0, 5.0, 3.0, 0.4, -3.0, 2, 50.0),
            RegimeStats("BEAR", 100, 20.0, -8.0, -4.0, -0.5, -12.0, 2, 50.0),
        ]

    def test_n_regimes_correct(self):
        s = regime_attribution_summary(self._make_stats())
        assert s["n_regimes"] == 3

    def test_best_regime_correct(self):
        s = regime_attribution_summary(self._make_stats())
        assert s["best_regime"] == "BULL"

    def test_worst_regime_correct(self):
        s = regime_attribution_summary(self._make_stats())
        assert s["worst_regime"] == "BEAR"

    def test_positive_regimes_count(self):
        s = regime_attribution_summary(self._make_stats())
        assert s["positive_regimes"] == 2

    def test_empty_returns_empty(self):
        assert regime_attribution_summary([]) == {}


class TestFormatRegimeAttribution:

    def _make_stats(self):
        return [
            RegimeStats("BULL", 300, 60.0, 20.0, 8.0, 0.8, -5.0, 3, 100.0),
            RegimeStats("BEAR", 100, 20.0, -8.0, -4.0, -0.5, -12.0, 2, 50.0),
        ]

    def test_contains_header(self):
        r = format_regime_attribution(self._make_stats())
        assert "REGIME" in r

    def test_contains_regime_names(self):
        r = format_regime_attribution(self._make_stats())
        assert "BULL" in r
        assert "BEAR" in r

    def test_contains_return_values(self):
        r = format_regime_attribution(self._make_stats())
        assert "20.0" in r

    def test_empty_returns_message(self):
        r = format_regime_attribution([])
        assert "No" in r or len(r) < 50
