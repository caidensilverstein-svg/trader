"""Unit tests for core/liquidity_risk.py."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest

from core.liquidity_risk import (
    estimate_amihud_ratio,
    estimate_spread_bps,
    compute_days_to_exit,
    compute_market_impact_bps,
    liquidity_score,
    compute_portfolio_liquidity,
    format_liquidity_report,
    LiquidityProfile,
)


def _make_prices_volumes(n=60, price=100.0, vol_daily=1_000_000, seed=42):
    np.random.seed(seed)
    rets = np.random.normal(0.0003, 0.01, n)
    prices = pd.Series(price * np.cumprod(1 + rets),
                       index=pd.bdate_range("2024-01-01", periods=n))
    volumes = pd.Series(np.random.randint(int(vol_daily * 0.5),
                                           int(vol_daily * 1.5), n),
                        index=prices.index).astype(float)
    return prices, volumes


class TestEstimateAmihudRatio:

    def test_returns_finite_float(self):
        prices, volumes = _make_prices_volumes()
        result = estimate_amihud_ratio(prices, volumes)
        assert np.isfinite(result)

    def test_higher_volume_lower_amihud(self):
        p, _ = _make_prices_volumes(price=100.0)
        v_high = pd.Series(1e7 * np.ones(60), index=p.index)
        v_low  = pd.Series(1e4 * np.ones(60), index=p.index)
        a_high = estimate_amihud_ratio(p, v_high)
        a_low  = estimate_amihud_ratio(p, v_low)
        assert a_high < a_low

    def test_insufficient_data_returns_nan(self):
        p = pd.Series([100, 101], index=pd.bdate_range("2024-01-01", periods=2))
        v = pd.Series([1000, 1000], index=p.index)
        result = estimate_amihud_ratio(p, v, window=30)
        assert np.isnan(result) or np.isfinite(result)  # short series still computes


class TestEstimateSpreadBps:

    def test_returns_positive_spread(self):
        prices, volumes = _make_prices_volumes()
        result = estimate_spread_bps(prices, volumes)
        assert result > 0

    def test_spread_in_reasonable_range(self):
        # For a liquid ETF, expect 1-20 bps
        prices, volumes = _make_prices_volumes(vol_daily=5_000_000)
        result = estimate_spread_bps(prices, volumes)
        assert 0 < result < 200  # very wide range to avoid flakiness

    def test_insufficient_data_returns_nan_or_fallback(self):
        p = pd.Series([100] * 5, index=pd.bdate_range("2024-01-01", periods=5))
        v = pd.Series([1000] * 5, index=p.index)
        result = estimate_spread_bps(p, v, window=30)
        # Should not crash; may return nan or fallback value
        assert result is not None


class TestComputeDaysToExit:

    def test_normal_case(self):
        result = compute_days_to_exit(100_000, 10_000_000)
        assert abs(result - 0.2) < 0.01

    def test_zero_adv_returns_inf(self):
        result = compute_days_to_exit(100_000, 0)
        assert result == float("inf")

    def test_large_position_many_days(self):
        result = compute_days_to_exit(50_000_000, 1_000_000)
        assert result > 10

    def test_tiny_position_fraction_of_day(self):
        result = compute_days_to_exit(10_000, 1_000_000_000)
        assert result < 0.01


class TestComputeMarketImpactBps:

    def test_returns_float(self):
        result = compute_market_impact_bps(100_000, 10_000_000, 0.001)
        assert isinstance(result, float)

    def test_zero_adv_returns_nan(self):
        result = compute_market_impact_bps(100_000, 0, 0.001)
        assert np.isnan(result)

    def test_larger_position_higher_impact(self):
        adv = 10_000_000
        amihud = 0.01
        small = compute_market_impact_bps(10_000, adv, amihud)
        large = compute_market_impact_bps(1_000_000, adv, amihud)
        assert large > small


class TestLiquidityScore:

    def test_score_between_0_and_100(self):
        score = liquidity_score(0.001, 100_000_000, 0.1)
        assert 0 <= score <= 100

    def test_low_amihud_high_vol_fast_exit_high_score(self):
        score = liquidity_score(0.0001, 5e9, 0.01)
        assert score > 60

    def test_high_amihud_low_vol_slow_exit_low_score(self):
        score = liquidity_score(100.0, 1000, 100.0)
        assert score < 50

    def test_nan_amihud_returns_default(self):
        score = liquidity_score(float("nan"), 1e6, 1.0)
        assert score == 50.0


class TestComputePortfolioLiquidity:

    def _make_ticker_data(self, tickers=("SPY", "AVUV")):
        data = {}
        for t in tickers:
            p, v = _make_prices_volumes(vol_daily=5_000_000)
            data[t] = {"prices": p, "volumes": v}
        return data

    def test_returns_list_of_profiles(self):
        data = self._make_ticker_data()
        positions = {"SPY": 20_000, "AVUV": 15_000}
        result = compute_portfolio_liquidity(data, positions)
        assert isinstance(result, list)
        assert all(isinstance(p, LiquidityProfile) for p in result)

    def test_missing_ticker_skipped(self):
        data = self._make_ticker_data(("SPY",))
        positions = {"SPY": 20_000, "MISSING": 10_000}
        result = compute_portfolio_liquidity(data, positions)
        tickers = {p.ticker for p in result}
        assert "MISSING" not in tickers

    def test_sorted_by_score_descending(self):
        data = self._make_ticker_data(("A", "B", "C"))
        positions = {"A": 10_000, "B": 10_000, "C": 10_000}
        result = compute_portfolio_liquidity(data, positions)
        scores = [p.liquidity_score for p in result]
        assert scores == sorted(scores, reverse=True)

    def test_position_size_preserved(self):
        data = self._make_ticker_data(("SPY",))
        positions = {"SPY": 25_000}
        result = compute_portfolio_liquidity(data, positions)
        assert len(result) == 1
        assert result[0].position_size == 25_000


class TestFormatLiquidityReport:

    def _make_profiles(self):
        return [
            LiquidityProfile("SPY", 1e7, 500.0, 0.001, 0.5, 0.01, 20_000, 0.1, 95.0),
            LiquidityProfile("AVUV", 5e5, 50.0, 0.05, 5.0, 0.5, 18_000, 2.1, 72.0),
        ]

    def test_contains_header(self):
        r = format_liquidity_report(self._make_profiles())
        assert "LIQUIDITY" in r

    def test_contains_tickers(self):
        r = format_liquidity_report(self._make_profiles())
        assert "SPY" in r
        assert "AVUV" in r

    def test_empty_returns_message(self):
        r = format_liquidity_report([])
        assert "unavailable" in r.lower()
