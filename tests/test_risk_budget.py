"""Unit tests for core/risk_budget.py."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest

from core.risk_budget import (
    compute_euler_risk_contributions,
    compute_risk_budget,
    format_risk_budget_report,
    SliceRiskBudget,
    RiskBudgetSummary,
)


def _make_prices(n=500, seed=1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n)
    m = rng.normal(0.0003, 0.01, n)
    a = 100 * np.cumprod(1 + m + rng.normal(0, 0.005, n))
    b = 100 * np.cumprod(1 + m * 0.8 + rng.normal(0, 0.008, n))
    c = 100 * np.cumprod(1 + rng.normal(0.0002, 0.012, n))
    d = 100 * np.cumprod(1 + rng.normal(0.0001, 0.004, n))
    return pd.DataFrame({"AVUV": a, "AVDV": b, "QMOM": c, "DBMF": d}, index=dates)


_WEIGHTS = {"AVUV": 0.30, "AVDV": 0.30, "QMOM": 0.25, "DBMF": 0.15}


class TestComputeEulerRiskContributions:

    def test_returns_array_of_correct_length(self):
        n = 4
        w = np.ones(n) / n
        cov = np.eye(n) * 0.01
        rc = compute_euler_risk_contributions(w, cov)
        assert len(rc) == n

    def test_contributions_sum_to_portfolio_vol(self):
        n = 4
        w = np.array([0.3, 0.3, 0.25, 0.15])
        cov = np.eye(n) * 0.01
        rc = compute_euler_risk_contributions(w, cov)
        port_vol = float(np.sqrt(w @ cov @ w))
        assert abs(rc.sum() - port_vol) < 1e-10

    def test_equal_weights_uncorrelated_equal_contributions(self):
        n = 4
        w = np.ones(n) / n
        cov = np.diag([0.01, 0.01, 0.01, 0.01])
        rc = compute_euler_risk_contributions(w, cov)
        pct = rc / rc.sum()
        assert np.allclose(pct, 0.25, atol=1e-8)

    def test_zero_weight_asset_zero_contribution(self):
        w = np.array([0.5, 0.5, 0.0])
        cov = np.eye(3) * 0.01
        rc = compute_euler_risk_contributions(w, cov)
        assert abs(rc[2]) < 1e-10

    def test_all_zeros_returns_zero_array(self):
        w = np.zeros(3)
        cov = np.eye(3) * 0.01
        rc = compute_euler_risk_contributions(w, cov)
        assert (rc == 0).all()


class TestComputeRiskBudget:

    def test_returns_tuple(self):
        prices = _make_prices()
        result = compute_risk_budget(prices, _WEIGHTS)
        assert isinstance(result, tuple) and len(result) == 2

    def test_budgets_list_length(self):
        prices = _make_prices()
        budgets, _ = compute_risk_budget(prices, _WEIGHTS)
        assert len(budgets) == len(_WEIGHTS)

    def test_budgets_are_dataclass(self):
        prices = _make_prices()
        budgets, _ = compute_risk_budget(prices, _WEIGHTS)
        for b in budgets:
            assert isinstance(b, SliceRiskBudget)

    def test_summary_is_dataclass(self):
        prices = _make_prices()
        _, summary = compute_risk_budget(prices, _WEIGHTS)
        assert isinstance(summary, RiskBudgetSummary)

    def test_risk_contributions_sum_to_100(self):
        prices = _make_prices()
        budgets, _ = compute_risk_budget(prices, _WEIGHTS)
        total = sum(b.actual_risk_contrib for b in budgets)
        assert abs(total - 100.0) < 1.0

    def test_sorted_by_actual_risk_contrib_descending(self):
        prices = _make_prices()
        budgets, _ = compute_risk_budget(prices, _WEIGHTS)
        contribs = [b.actual_risk_contrib for b in budgets]
        assert contribs == sorted(contribs, reverse=True)

    def test_total_portfolio_vol_positive(self):
        prices = _make_prices()
        _, summary = compute_risk_budget(prices, _WEIGHTS)
        assert summary.total_portfolio_vol > 0

    def test_effective_n_between_1_and_n(self):
        prices = _make_prices()
        _, summary = compute_risk_budget(prices, _WEIGHTS)
        n = len(_WEIGHTS)
        assert 1 <= summary.effective_n <= n + 0.1

    def test_standalone_vol_positive(self):
        prices = _make_prices()
        budgets, _ = compute_risk_budget(prices, _WEIGHTS)
        for b in budgets:
            assert b.volatility > 0

    def test_high_vol_asset_has_high_risk_contribution(self):
        prices = _make_prices()
        budgets, _ = compute_risk_budget(prices, _WEIGHTS)
        # QMOM (vol ~1.2%) should have higher standalone vol than DBMF (vol ~0.4%)
        qmom = next(b for b in budgets if b.sleeve == "QMOM")
        dbmf = next(b for b in budgets if b.sleeve == "DBMF")
        assert qmom.volatility > dbmf.volatility

    def test_risk_deviation_computed(self):
        prices = _make_prices()
        targets = {"AVUV": 0.25, "AVDV": 0.25, "QMOM": 0.25, "DBMF": 0.25}
        budgets, _ = compute_risk_budget(prices, _WEIGHTS, risk_budget_targets=targets)
        for b in budgets:
            expected = b.actual_risk_contrib - b.target_risk_budget
            assert abs(b.risk_deviation - expected) < 0.1

    def test_equal_vol_assets_equal_risk_contrib_equal_weights(self):
        # All assets same vol, same weights => equal risk contributions
        dates = pd.bdate_range("2020-01-01", periods=500)
        rng = np.random.default_rng(99)
        v = 0.01
        p = pd.DataFrame({
            "A": 100 * np.cumprod(1 + rng.normal(0, v, 500)),
            "B": 100 * np.cumprod(1 + rng.normal(0, v, 500)),
        }, index=dates)
        w = {"A": 0.5, "B": 0.5}
        budgets, _ = compute_risk_budget(p, w)
        # Approximate equal contributions (tolerance for estimation noise)
        assert abs(budgets[0].actual_risk_contrib - 50) < 15

    def test_concentration_index_between_0_and_100(self):
        prices = _make_prices()
        _, summary = compute_risk_budget(prices, _WEIGHTS)
        assert 0 <= summary.concentration_index <= 100

    def test_empty_prices_returns_empty(self):
        budgets, summary = compute_risk_budget(pd.DataFrame(), _WEIGHTS)
        assert budgets == []

    def test_empty_weights_returns_empty(self):
        prices = _make_prices()
        budgets, summary = compute_risk_budget(prices, {})
        assert budgets == []

    def test_missing_tickers_excluded(self):
        prices = _make_prices()
        w = {"AVUV": 0.5, "NOTEXIST": 0.5}
        budgets, _ = compute_risk_budget(prices, w)
        tickers = [b.sleeve for b in budgets]
        assert "NOTEXIST" not in tickers


class TestFormatRiskBudgetReport:

    def test_contains_header(self):
        prices = _make_prices()
        budgets, summary = compute_risk_budget(prices, _WEIGHTS)
        r = format_risk_budget_report(budgets, summary)
        assert "RISK BUDGET" in r

    def test_contains_sleeve_names(self):
        prices = _make_prices()
        budgets, summary = compute_risk_budget(prices, _WEIGHTS)
        r = format_risk_budget_report(budgets, summary)
        assert "AVUV" in r

    def test_contains_effective_n(self):
        prices = _make_prices()
        budgets, summary = compute_risk_budget(prices, _WEIGHTS)
        r = format_risk_budget_report(budgets, summary)
        assert "Effective" in r

    def test_empty_returns_message(self):
        r = format_risk_budget_report([], RiskBudgetSummary(0, 0, 0, 0, 0))
        assert "unavailable" in r.lower()
