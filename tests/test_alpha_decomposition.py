"""Unit tests for backtest/alpha_decomposition.py."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest

from backtest.alpha_decomposition import (
    compute_alpha_metrics,
    format_alpha_report,
    AlphaMetrics,
)


def _make_series(n=500, seed=42):
    np.random.seed(seed)
    idx = pd.bdate_range("2020-01-01", periods=n)
    return idx


class TestComputeAlphaMetrics:

    def _make_returns(self, n=500, port_alpha=0.0, beta=0.8, seed=42):
        np.random.seed(seed)
        idx = pd.bdate_range("2020-01-01", periods=n)
        bench_rets = np.random.normal(0.0003, 0.01, n)
        port_rets  = port_alpha / 252 + beta * bench_rets + np.random.normal(0, 0.003, n)
        return (pd.Series(port_rets, index=idx),
                pd.Series(bench_rets, index=idx))

    def test_returns_alpha_metrics_object(self):
        p, b = self._make_returns()
        result = compute_alpha_metrics(p, b)
        assert isinstance(result, AlphaMetrics)

    def test_beta_roughly_correct(self):
        p, b = self._make_returns(beta=0.75)
        result = compute_alpha_metrics(p, b)
        assert abs(result.market_beta - 0.75) < 0.1

    def test_positive_alpha_portfolio_has_positive_ir(self):
        p, b = self._make_returns(port_alpha=0.05)  # 5% annual alpha
        result = compute_alpha_metrics(p, b)
        assert result.information_ratio > 0

    def test_r_squared_between_0_and_1(self):
        p, b = self._make_returns()
        result = compute_alpha_metrics(p, b)
        assert 0 <= result.r_squared <= 1.0

    def test_tracking_error_positive(self):
        p, b = self._make_returns()
        result = compute_alpha_metrics(p, b)
        assert result.tracking_error_pct > 0

    def test_up_capture_positive(self):
        p, b = self._make_returns()
        result = compute_alpha_metrics(p, b)
        assert result.up_capture > 0

    def test_down_capture_positive(self):
        p, b = self._make_returns()
        result = compute_alpha_metrics(p, b)
        assert result.down_capture > 0

    def test_perfect_tracking_gives_zero_te(self):
        # Portfolio = benchmark exactly
        idx = pd.bdate_range("2020-01-01", periods=300)
        np.random.seed(1)
        b = pd.Series(np.random.normal(0.0003, 0.01, 300), index=idx)
        p = b.copy()  # exact same returns
        result = compute_alpha_metrics(p, b)
        assert abs(result.tracking_error_pct) < 0.1

    def test_perfect_beta_1(self):
        idx = pd.bdate_range("2020-01-01", periods=300)
        np.random.seed(2)
        b = pd.Series(np.random.normal(0.0003, 0.01, 300), index=idx)
        p = b + 0.0001  # constant alpha, same beta
        result = compute_alpha_metrics(p, b)
        assert abs(result.market_beta - 1.0) < 0.05

    def test_insufficient_data_raises(self):
        idx = pd.bdate_range("2020-01-01", periods=10)
        p = pd.Series(np.random.normal(0, 0.01, 10), index=idx)
        b = pd.Series(np.random.normal(0, 0.01, 10), index=idx)
        with pytest.raises(ValueError):
            compute_alpha_metrics(p, b)

    def test_higher_up_lower_down_gives_good_timing_ratio(self):
        np.random.seed(5)
        idx = pd.bdate_range("2020-01-01", periods=500)
        b = pd.Series(np.random.normal(0.0003, 0.01, 500), index=idx)
        # Portfolio captures more upside, less downside
        p = pd.Series(
            np.where(b.values > 0, b.values * 1.1, b.values * 0.9),
            index=idx
        )
        result = compute_alpha_metrics(p, b)
        assert result.up_capture > result.down_capture


class TestFormatAlphaReport:

    def _make_metrics(self):
        return AlphaMetrics(
            jensen_alpha_pct=2.5, alpha_t_stat=2.1, alpha_significant=True,
            market_beta=0.75, r_squared=0.85,
            tracking_error_pct=4.2, information_ratio=0.60,
            treynor_ratio=6.3, up_capture=92.0, down_capture=76.0,
            active_return_pct=2.5,
        )

    def test_contains_header(self):
        r = format_alpha_report(self._make_metrics())
        assert "ALPHA" in r

    def test_contains_ir(self):
        r = format_alpha_report(self._make_metrics())
        assert "0.600" in r or "Information Ratio" in r

    def test_contains_beta(self):
        r = format_alpha_report(self._make_metrics())
        assert "0.750" in r

    def test_contains_capture_ratios(self):
        r = format_alpha_report(self._make_metrics())
        assert "92.0" in r
        assert "76.0" in r

    def test_significant_alpha_flagged(self):
        r = format_alpha_report(self._make_metrics())
        assert "significant" in r.lower()

    def test_favorable_capture_detected(self):
        r = format_alpha_report(self._make_metrics())
        assert "FAVORABLE" in r
