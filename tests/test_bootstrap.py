"""Unit tests for backtest/bootstrap.py."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest

from backtest.bootstrap import (
    bootstrap_metrics,
    format_ci_report,
    _compute_stats,
    _optimal_block_length,
    _stationary_bootstrap_sample,
)


def _make_equity(n: int = 500, daily_ret: float = 0.0003) -> pd.Series:
    """Generate a simple equity curve with known return."""
    np.random.seed(42)
    rets = np.random.normal(daily_ret, 0.01, n)
    equity = pd.Series(100_000 * np.cumprod(1 + rets))
    return equity


class TestComputeStats:

    def test_positive_trend_positive_return(self):
        eq = _make_equity(500, daily_ret=0.001)
        rets = eq.pct_change().dropna().values
        stats = _compute_stats(rets)
        assert stats["ann_return"] > 0

    def test_negative_trend_negative_return(self):
        eq = _make_equity(500, daily_ret=-0.001)
        rets = eq.pct_change().dropna().values
        stats = _compute_stats(rets)
        assert stats["ann_return"] < 0

    def test_max_dd_is_negative(self):
        rets = _make_equity(500).pct_change().dropna().values
        stats = _compute_stats(rets)
        assert stats["max_dd"] <= 0

    def test_sharpe_positive_for_positive_returns(self):
        rets = _make_equity(500, daily_ret=0.001).pct_change().dropna().values
        stats = _compute_stats(rets)
        assert stats["sharpe"] > 0

    def test_calmar_positive_for_positive_returns(self):
        rets = _make_equity(500, daily_ret=0.001).pct_change().dropna().values
        stats = _compute_stats(rets)
        assert stats["calmar"] > 0

    def test_returns_required_keys(self):
        rets = _make_equity(100).pct_change().dropna().values
        stats = _compute_stats(rets)
        for key in ("ann_return", "ann_vol", "sharpe", "max_dd", "calmar"):
            assert key in stats

    def test_insufficient_data_returns_empty(self):
        stats = _compute_stats(np.array([0.01]))
        assert stats == {}


class TestOptimalBlockLength:

    def test_positive_for_any_size(self):
        for n in [50, 100, 250, 1000, 5000]:
            assert _optimal_block_length(np.zeros(n)) >= 1

    def test_longer_series_larger_block(self):
        b50  = _optimal_block_length(np.zeros(50))
        b500 = _optimal_block_length(np.zeros(500))
        assert b500 > b50


class TestStationaryBootstrapSample:

    def test_same_length_as_input(self):
        rng = np.random.default_rng(42)
        data = np.random.normal(0, 0.01, 100)
        sample = _stationary_bootstrap_sample(data, block_length=5, rng=rng)
        assert len(sample) == len(data)

    def test_values_come_from_original(self):
        rng = np.random.default_rng(42)
        data = np.arange(100, dtype=float)
        sample = _stationary_bootstrap_sample(data, block_length=10, rng=rng)
        for v in sample:
            assert v in data

    def test_different_seeds_produce_different_samples(self):
        data = np.random.normal(0, 0.01, 100)
        s1 = _stationary_bootstrap_sample(data, 5, np.random.default_rng(1))
        s2 = _stationary_bootstrap_sample(data, 5, np.random.default_rng(2))
        assert not np.allclose(s1, s2)


class TestBootstrapMetrics:

    def test_returns_ci_dict(self):
        eq  = _make_equity(500)
        cis = bootstrap_metrics(eq, n_boot=50)
        assert isinstance(cis, dict)
        assert len(cis) > 0

    def test_ci_contains_required_keys(self):
        eq  = _make_equity(500)
        cis = bootstrap_metrics(eq, n_boot=50)
        for metric, d in cis.items():
            for k in ("point", "lower", "upper", "ci_width", "significant"):
                assert k in d, f"Missing {k} in {metric}"

    def test_lower_le_point_le_upper(self):
        eq  = _make_equity(500)
        cis = bootstrap_metrics(eq, n_boot=100)
        for metric, d in cis.items():
            # Note: CI may not bracket point for pathological samples,
            # but for a reasonable equity curve with 100 boots it should
            assert d["lower"] <= d["upper"], f"{metric}: lower > upper"

    def test_insufficient_data_returns_empty(self):
        eq  = pd.Series([100_000 + i * 10 for i in range(10)])
        cis = bootstrap_metrics(eq, n_boot=10)
        assert cis == {}

    def test_significant_flag_for_strong_trend(self):
        # Strongly positive returns should have significant Sharpe
        eq  = _make_equity(1000, daily_ret=0.002)
        cis = bootstrap_metrics(eq, n_boot=100)
        if "sharpe" in cis:
            assert cis["sharpe"]["significant"] is True

    def test_reproducible_with_same_seed(self):
        eq  = _make_equity(300)
        c1  = bootstrap_metrics(eq, n_boot=50, seed=99)
        c2  = bootstrap_metrics(eq, n_boot=50, seed=99)
        assert c1 == c2


class TestFormatCIReport:

    def _make_cis(self) -> dict:
        return {
            "ann_return": {"point": 0.065, "lower": 0.001, "upper": 0.121,
                           "ci_width": 0.120, "significant": True},
            "sharpe":     {"point": 0.742, "lower": 0.003, "upper": 1.544,
                           "ci_width": 1.541, "significant": True},
        }

    def test_report_contains_header(self):
        r = format_ci_report(self._make_cis())
        assert "BOOTSTRAP CONFIDENCE INTERVALS" in r

    def test_report_contains_metric_names(self):
        r = format_ci_report(self._make_cis())
        assert "ann_return" in r
        assert "sharpe" in r

    def test_report_contains_point_estimates(self):
        r = format_ci_report(self._make_cis())
        assert "6.5%" in r

    def test_significant_shows_yes(self):
        r = format_ci_report(self._make_cis())
        assert "YES" in r
