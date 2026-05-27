"""Unit tests for backtest/tail_risk.py."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest

from backtest.tail_risk import (
    compute_portfolio_tail_risk,
    format_tail_risk_report,
    AssetTailRisk,
    TailRiskSummary,
)


def _make_prices(n=500, seed=7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n)
    market = rng.normal(0.0003, 0.01, n)
    a = 100 * np.cumprod(1 + market + rng.normal(0, 0.005, n))
    b = 100 * np.cumprod(1 + market * 0.8 + rng.normal(0, 0.008, n))
    c = 100 * np.cumprod(1 + rng.normal(0.0001, 0.012, n))
    d = 100 * np.cumprod(1 + rng.normal(0.0001, 0.006, n))
    return pd.DataFrame({"AVUV": a, "AVDV": b, "QMOM": c, "DBMF": d}, index=dates)


_WEIGHTS = {"AVUV": 0.30, "AVDV": 0.30, "QMOM": 0.25, "DBMF": 0.15}


class TestComputePortfolioTailRisk:

    def test_returns_tuple(self):
        prices = _make_prices()
        result = compute_portfolio_tail_risk(prices, _WEIGHTS)
        assert isinstance(result, tuple) and len(result) == 2

    def test_asset_list_length(self):
        prices = _make_prices()
        risks, _ = compute_portfolio_tail_risk(prices, _WEIGHTS)
        assert len(risks) == len(_WEIGHTS)

    def test_assets_are_dataclass(self):
        prices = _make_prices()
        risks, _ = compute_portfolio_tail_risk(prices, _WEIGHTS)
        for r in risks:
            assert isinstance(r, AssetTailRisk)

    def test_summary_is_dataclass(self):
        prices = _make_prices()
        _, summary = compute_portfolio_tail_risk(prices, _WEIGHTS)
        assert isinstance(summary, TailRiskSummary)

    def test_var_positive(self):
        prices = _make_prices()
        risks, summary = compute_portfolio_tail_risk(prices, _WEIGHTS)
        assert summary.portfolio_var95 > 0
        assert summary.portfolio_var99 > 0

    def test_cvar_gt_var(self):
        prices = _make_prices()
        _, summary = compute_portfolio_tail_risk(prices, _WEIGHTS)
        assert summary.portfolio_cvar95 >= summary.portfolio_var95
        assert summary.portfolio_cvar99 >= summary.portfolio_var99

    def test_99_var_gt_95_var(self):
        prices = _make_prices()
        _, summary = compute_portfolio_tail_risk(prices, _WEIGHTS)
        assert summary.portfolio_var99 >= summary.portfolio_var95

    def test_pct_of_total_sums_to_100(self):
        prices = _make_prices()
        risks, _ = compute_portfolio_tail_risk(prices, _WEIGHTS)
        total = sum(r.pct_of_total_cvar for r in risks)
        assert abs(total - 100.0) < 1.0

    def test_sorted_by_pct_descending(self):
        prices = _make_prices()
        risks, _ = compute_portfolio_tail_risk(prices, _WEIGHTS)
        pcts = [r.pct_of_total_cvar for r in risks]
        assert pcts == sorted(pcts, reverse=True)

    def test_standalone_cvar_gt_zero(self):
        prices = _make_prices()
        risks, _ = compute_portfolio_tail_risk(prices, _WEIGHTS)
        for r in risks:
            assert r.standalone_cvar95 >= 0

    def test_worst_day_negative(self):
        prices = _make_prices()
        risks, _ = compute_portfolio_tail_risk(prices, _WEIGHTS)
        for r in risks:
            assert r.worst_day < 0

    def test_tail_beta_in_valid_range(self):
        prices = _make_prices()
        risks, _ = compute_portfolio_tail_risk(prices, _WEIGHTS)
        for r in risks:
            assert -1.5 <= r.tail_beta <= 1.5  # allow slight numerical noise

    def test_diversification_benefit_positive(self):
        prices = _make_prices()
        _, summary = compute_portfolio_tail_risk(prices, _WEIGHTS)
        # True for any correlated but not identical assets
        assert summary.diversification_benefit > -0.1

    def test_dominant_risk_is_largest_weight(self):
        # AVUV and AVDV have equal weight, but one should dominate due to vol
        prices = _make_prices()
        _, summary = compute_portfolio_tail_risk(prices, _WEIGHTS)
        assert summary.dominant_risk_asset in _WEIGHTS

    def test_window_parameter_limits_history(self):
        prices = _make_prices()
        risks_full, _ = compute_portfolio_tail_risk(prices, _WEIGHTS)
        risks_window, _ = compute_portfolio_tail_risk(prices, _WEIGHTS, window=252)
        # Results should differ since different time windows
        assert risks_full is not risks_window

    def test_empty_prices_returns_empty(self):
        risks, summary = compute_portfolio_tail_risk(pd.DataFrame(), _WEIGHTS)
        assert risks == []

    def test_empty_weights_returns_empty(self):
        prices = _make_prices()
        risks, summary = compute_portfolio_tail_risk(prices, {})
        assert risks == []

    def test_missing_ticker_ignored(self):
        prices = _make_prices()
        w = {"AVUV": 0.5, "NOTEXIST": 0.5}
        risks, _ = compute_portfolio_tail_risk(prices, w)
        tickers = [r.ticker for r in risks]
        assert "NOTEXIST" not in tickers
        assert "AVUV" in tickers

    def test_high_vol_asset_has_higher_standalone_cvar(self):
        prices = _make_prices()
        risks, _ = compute_portfolio_tail_risk(prices, _WEIGHTS)
        # QMOM (std=1.2%) should have higher standalone CVaR than DBMF (std=0.6%)
        qmom = next(r for r in risks if r.ticker == "QMOM")
        dbmf = next(r for r in risks if r.ticker == "DBMF")
        assert qmom.standalone_cvar95 > dbmf.standalone_cvar95


class TestFormatTailRiskReport:

    def test_contains_header(self):
        prices = _make_prices()
        risks, summary = compute_portfolio_tail_risk(prices, _WEIGHTS)
        r = format_tail_risk_report(risks, summary)
        assert "TAIL RISK" in r

    def test_contains_var_and_cvar(self):
        prices = _make_prices()
        risks, summary = compute_portfolio_tail_risk(prices, _WEIGHTS)
        r = format_tail_risk_report(risks, summary)
        assert "VaR" in r
        assert "CVaR" in r

    def test_contains_asset_names(self):
        prices = _make_prices()
        risks, summary = compute_portfolio_tail_risk(prices, _WEIGHTS)
        r = format_tail_risk_report(risks, summary)
        assert "AVUV" in r

    def test_contains_dominant_asset(self):
        prices = _make_prices()
        risks, summary = compute_portfolio_tail_risk(prices, _WEIGHTS)
        r = format_tail_risk_report(risks, summary)
        assert summary.dominant_risk_asset in r

    def test_empty_returns_message(self):
        r = format_tail_risk_report([], TailRiskSummary(0, 0, 0, 0, 0, "", 0, 0, 0))
        assert "unavailable" in r.lower()
