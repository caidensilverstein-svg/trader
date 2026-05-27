"""Unit tests for core/var_calculator.py."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest

from core.var_calculator import (
    historical_var,
    historical_cvar,
    parametric_var,
    compute_var_report,
    format_var_report,
    _daily_returns,
)


def _make_equity(n: int = 500, daily_vol: float = 0.01) -> pd.Series:
    """Generate equity curve with known volatility."""
    np.random.seed(42)
    rets = np.random.normal(0.0003, daily_vol, n)
    return pd.Series(100_000 * np.cumprod(1 + rets))


class TestHistoricalVaR:

    def test_var_positive(self):
        eq = _make_equity()
        rets = eq.pct_change().dropna().values
        var = historical_var(rets, confidence=0.95, portfolio_value=100_000)
        assert var > 0

    def test_99_var_gt_95_var(self):
        eq = _make_equity()
        rets = eq.pct_change().dropna().values
        var95 = historical_var(rets, confidence=0.95, portfolio_value=100_000)
        var99 = historical_var(rets, confidence=0.99, portfolio_value=100_000)
        assert var99 > var95

    def test_10d_var_gt_1d_var(self):
        eq = _make_equity()
        rets = eq.pct_change().dropna().values
        var1d  = historical_var(rets, confidence=0.95, horizon_days=1,  portfolio_value=100_000)
        var10d = historical_var(rets, confidence=0.95, horizon_days=10, portfolio_value=100_000)
        assert var10d > var1d

    def test_var_scales_with_portfolio_value(self):
        eq = _make_equity()
        rets = eq.pct_change().dropna().values
        var100k = historical_var(rets, confidence=0.95, portfolio_value=100_000)
        var200k = historical_var(rets, confidence=0.95, portfolio_value=200_000)
        assert abs(var200k / var100k - 2.0) < 0.001

    def test_insufficient_data_returns_zero(self):
        rets = np.array([0.01, -0.01, 0.02])
        var  = historical_var(rets, confidence=0.95, portfolio_value=100_000)
        assert var == 0.0


class TestHistoricalCVaR:

    def test_cvar_positive(self):
        eq = _make_equity()
        rets = eq.pct_change().dropna().values
        cvar = historical_cvar(rets, confidence=0.95, portfolio_value=100_000)
        assert cvar > 0

    def test_cvar_gt_var(self):
        eq = _make_equity()
        rets = eq.pct_change().dropna().values
        var  = historical_var(rets, confidence=0.95, portfolio_value=100_000)
        cvar = historical_cvar(rets, confidence=0.95, portfolio_value=100_000)
        assert cvar > var  # CVaR >= VaR by construction

    def test_99_cvar_gt_95_cvar(self):
        eq = _make_equity()
        rets = eq.pct_change().dropna().values
        cvar95 = historical_cvar(rets, confidence=0.95, portfolio_value=100_000)
        cvar99 = historical_cvar(rets, confidence=0.99, portfolio_value=100_000)
        assert cvar99 > cvar95

    def test_insufficient_data_returns_zero(self):
        rets = np.array([0.01, -0.01])
        cvar = historical_cvar(rets, confidence=0.95, portfolio_value=100_000)
        assert cvar == 0.0


class TestParametricVaR:

    def test_var_positive(self):
        eq = _make_equity()
        rets = eq.pct_change().dropna().values
        var = parametric_var(rets, confidence=0.95, portfolio_value=100_000)
        assert var > 0

    def test_parametric_close_to_historical_for_normal_returns(self):
        # With normally distributed returns, both should be close
        np.random.seed(42)
        rets = np.random.normal(0, 0.01, 1000)
        hist = historical_var(rets, confidence=0.95, portfolio_value=100_000)
        para = parametric_var(rets, confidence=0.95, portfolio_value=100_000)
        # Within 25% of each other for normal distribution
        assert abs(hist - para) / max(hist, para) < 0.25

    def test_99_var_gt_95_var(self):
        eq = _make_equity()
        rets = eq.pct_change().dropna().values
        var95 = parametric_var(rets, confidence=0.95, portfolio_value=100_000)
        var99 = parametric_var(rets, confidence=0.99, portfolio_value=100_000)
        assert var99 > var95


class TestComputeVarReport:

    def test_returns_dict(self):
        eq = _make_equity()
        r  = compute_var_report(eq, portfolio_value=100_000)
        assert isinstance(r, dict)
        assert len(r) > 0

    def test_contains_key_metrics(self):
        eq = _make_equity()
        r  = compute_var_report(eq)
        assert "hist_var_95_1d" in r
        assert "hist_cvar_95_1d" in r
        assert "hist_var_99_1d" in r
        assert "ann_vol_pct" in r

    def test_all_values_positive(self):
        eq = _make_equity()
        r  = compute_var_report(eq)
        for k, v in r.items():
            assert v >= 0, f"{k} is negative: {v}"

    def test_insufficient_data_returns_empty(self):
        eq = pd.Series([100_000, 100_100, 100_200])
        r  = compute_var_report(eq)
        assert r == {}


class TestFormatVarReport:

    def _make_var_data(self) -> dict:
        return {
            "hist_var_95_1d": 820, "hist_var_95_10d": 2592,
            "hist_cvar_95_1d": 1378, "hist_cvar_95_10d": 4359,
            "param_var_95_1d": 878, "param_var_95_10d": 2775,
            "hist_var_99_1d": 1655, "hist_var_99_10d": 5235,
            "hist_cvar_99_1d": 2479, "hist_cvar_99_10d": 7838,
            "param_var_99_1d": 1252, "param_var_99_10d": 3960,
            "ann_vol_pct": 8.73,
        }

    def test_report_contains_header(self):
        r = format_var_report(self._make_var_data(), 100_000)
        assert "VALUE-AT-RISK" in r

    def test_report_shows_95_and_99(self):
        r = format_var_report(self._make_var_data(), 100_000)
        assert "95%" in r
        assert "99%" in r

    def test_report_shows_dollar_values(self):
        r = format_var_report(self._make_var_data(), 100_000)
        assert "820" in r
        assert "1,655" in r
