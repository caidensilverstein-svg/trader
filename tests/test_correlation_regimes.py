"""Unit tests for backtest/correlation_regimes.py."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest

from backtest.correlation_regimes import (
    compute_rolling_correlations,
    compute_regime_correlations,
    format_correlation_regime_report,
    PairCorrelation,
    CorrelationRegimeSummary,
)


def _make_prices(n=500, seed=42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2021-01-01", periods=n)
    # Correlated assets: A and B highly correlated, C uncorrelated
    market = rng.normal(0.0004, 0.01, n)
    a = 100 * np.cumprod(1 + market + rng.normal(0, 0.005, n))
    b = 100 * np.cumprod(1 + market + rng.normal(0, 0.005, n))
    c = 100 * np.cumprod(1 + rng.normal(0.0002, 0.015, n))
    d = 100 * np.cumprod(1 + rng.normal(0.0001, 0.008, n))
    return pd.DataFrame({"AVUV": a, "AVDV": b, "QMOM": c, "DBMF": d}, index=dates)


def _make_regime(prices: pd.DataFrame, stress_frac=0.2) -> pd.Series:
    n = len(prices)
    labels = ["BULL"] * n
    for i in range(int(n * stress_frac)):
        labels[i + int(n * 0.4)] = "BEAR"
    return pd.Series(labels, index=prices.index)


class TestComputeRollingCorrelations:

    def test_returns_dict(self):
        prices = _make_prices()
        result = compute_rolling_correlations(prices, window=63)
        assert isinstance(result, dict)

    def test_correct_number_of_pairs(self):
        prices = _make_prices()
        result = compute_rolling_correlations(prices, window=63)
        n = len(prices.columns)
        expected = n * (n - 1) // 2
        assert len(result) == expected

    def test_all_keys_are_tuples(self):
        prices = _make_prices()
        result = compute_rolling_correlations(prices, window=63)
        for key in result:
            assert isinstance(key, tuple)
            assert len(key) == 2

    def test_values_are_series(self):
        prices = _make_prices()
        result = compute_rolling_correlations(prices, window=63)
        for v in result.values():
            assert isinstance(v, pd.Series)

    def test_rolling_corr_range(self):
        prices = _make_prices()
        result = compute_rolling_correlations(prices, window=63)
        for series in result.values():
            valid = series.dropna()
            assert (valid >= -1.01).all() and (valid <= 1.01).all()

    def test_corr_of_identical_series_is_one(self):
        dates = pd.bdate_range("2022-01-01", periods=200)
        p = np.cumprod(1 + np.random.default_rng(0).normal(0, 0.01, 200)) * 100
        prices = pd.DataFrame({"A": p, "B": p}, index=dates)
        result = compute_rolling_correlations(prices, window=63)
        corr = result[("A", "B")].dropna()
        assert (corr - 1.0).abs().max() < 1e-8


class TestComputeRegimeCorrelations:

    def test_returns_tuple(self):
        prices = _make_prices()
        result = compute_regime_correlations(prices)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_pairs_list_correct_length(self):
        prices = _make_prices()
        pairs, _ = compute_regime_correlations(prices)
        n = len(prices.columns)
        assert len(pairs) == n * (n - 1) // 2

    def test_pairs_are_pair_correlation_dataclass(self):
        prices = _make_prices()
        pairs, _ = compute_regime_correlations(prices)
        for p in pairs:
            assert isinstance(p, PairCorrelation)

    def test_summary_is_dataclass(self):
        prices = _make_prices()
        _, summary = compute_regime_correlations(prices)
        assert isinstance(summary, CorrelationRegimeSummary)

    def test_full_corr_in_valid_range(self):
        prices = _make_prices()
        pairs, _ = compute_regime_correlations(prices)
        for p in pairs:
            assert -1.01 <= p.full_corr <= 1.01

    def test_regime_breakdown_computed(self):
        prices = _make_prices()
        reg = _make_regime(prices)
        pairs, _ = compute_regime_correlations(prices, regime_series=reg)
        for p in pairs:
            expected = round(p.stress_corr - p.calm_corr, 3)
            assert abs(p.corr_breakdown - expected) < 0.002

    def test_sorted_by_full_corr_descending(self):
        prices = _make_prices()
        pairs, _ = compute_regime_correlations(prices)
        corrs = [p.full_corr for p in pairs]
        assert corrs == sorted(corrs, reverse=True)

    def test_correlated_assets_have_high_full_corr(self):
        prices = _make_prices()
        pairs, _ = compute_regime_correlations(prices)
        av_pair = next(p for p in pairs if {p.asset_a, p.asset_b} == {"AVUV", "AVDV"})
        assert av_pair.full_corr > 0.7

    def test_uncorrelated_asset_low_corr(self):
        prices = _make_prices()
        pairs, _ = compute_regime_correlations(prices)
        dbmf_pairs = [p for p in pairs if "DBMF" in (p.asset_a, p.asset_b)]
        for p in dbmf_pairs:
            assert abs(p.full_corr) < 0.6  # DBMF is largely uncorrelated

    def test_summary_avg_full_corr_consistent(self):
        prices = _make_prices()
        pairs, summary = compute_regime_correlations(prices)
        expected = round(np.mean([p.full_corr for p in pairs]), 3)
        assert abs(summary.avg_full_corr - expected) < 0.01

    def test_diversification_calm_plus_avg_calm_eq_1(self):
        prices = _make_prices()
        reg = _make_regime(prices)
        pairs, summary = compute_regime_correlations(prices, regime_series=reg)
        assert abs(summary.diversification_calm + summary.avg_calm_corr - 1.0) < 0.01

    def test_empty_prices_returns_empty(self):
        prices = pd.DataFrame()
        pairs, summary = compute_regime_correlations(prices)
        assert pairs == []

    def test_single_asset_returns_empty(self):
        dates = pd.bdate_range("2022-01-01", periods=100)
        prices = pd.DataFrame({"A": np.ones(100) * 100}, index=dates)
        pairs, summary = compute_regime_correlations(prices)
        assert pairs == []

    def test_no_regime_series_uses_full_period(self):
        prices = _make_prices()
        pairs, _ = compute_regime_correlations(prices, regime_series=None)
        for p in pairs:
            # calm_corr defaults to full period when no regime
            assert -1.01 <= p.calm_corr <= 1.01

    def test_worst_pair_in_summary(self):
        prices = _make_prices()
        reg = _make_regime(prices)
        pairs, summary = compute_regime_correlations(prices, regime_series=reg)
        if summary.worst_pair:
            worst = max(pairs, key=lambda p: p.corr_breakdown)
            assert summary.worst_pair == f"{worst.asset_a}/{worst.asset_b}"

    def test_best_pair_in_summary(self):
        prices = _make_prices()
        reg = _make_regime(prices)
        pairs, summary = compute_regime_correlations(prices, regime_series=reg)
        if summary.best_pair:
            best = min(pairs, key=lambda p: p.stress_corr)
            assert summary.best_pair == f"{best.asset_a}/{best.asset_b}"


class TestFormatCorrelationRegimeReport:

    def test_contains_header(self):
        prices = _make_prices()
        pairs, summary = compute_regime_correlations(prices)
        r = format_correlation_regime_report(pairs, summary)
        assert "CORRELATION" in r

    def test_contains_asset_names(self):
        prices = _make_prices()
        pairs, summary = compute_regime_correlations(prices)
        r = format_correlation_regime_report(pairs, summary)
        assert "AVUV" in r

    def test_contains_summary_section(self):
        prices = _make_prices()
        pairs, summary = compute_regime_correlations(prices)
        r = format_correlation_regime_report(pairs, summary)
        assert "SUMMARY" in r

    def test_empty_returns_message(self):
        r = format_correlation_regime_report([], CorrelationRegimeSummary(0, 0, 0, 0, "", "", 0, 0))
        assert "unavailable" in r.lower()

    def test_breakdown_flag_present_for_high_breakdown(self):
        pair = PairCorrelation("A", "B", 0.8, 0.5, 0.9, 0.4, -0.1, 0.95, 0.7)
        summary = CorrelationRegimeSummary(0.8, 0.5, 0.9, 0.4, "A/B", "A/B", 0.5, 0.1)
        r = format_correlation_regime_report([pair], summary)
        assert "(!)" in r
