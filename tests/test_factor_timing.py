"""Unit tests for core/factor_timing.py."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest

from core.factor_timing import (
    compute_etf_momentum,
    compute_timing_multipliers,
    apply_factor_timing,
    factor_timing_summary,
    NEGATIVE_MOM_PENALTY,
    POSITIVE_MOM_BOOST,
    MIN_WEIGHT_FRACTION,
)


def _make_prices(n: int, monthly_ret: float) -> pd.Series:
    """Generate price series with known monthly return."""
    daily_r = (1 + monthly_ret) ** (1 / 21) - 1
    prices  = [100.0]
    for _ in range(n - 1):
        prices.append(prices[-1] * (1 + daily_r))
    return pd.Series(prices)


class TestComputeETFMomentum:

    def test_positive_trend_positive_momentum(self):
        prices = {"AVUV": _make_prices(200, 0.02)}
        mom = compute_etf_momentum(prices, lookback=126)
        assert mom["AVUV"] > 0

    def test_negative_trend_negative_momentum(self):
        prices = {"AVUV": _make_prices(200, -0.02)}
        mom = compute_etf_momentum(prices, lookback=126)
        assert mom["AVUV"] < 0

    def test_insufficient_data_returns_zero(self):
        prices = {"CTA": _make_prices(50, 0.01)}  # < 126 days
        mom = compute_etf_momentum(prices, lookback=126)
        assert mom["CTA"] == 0.0

    def test_multiple_tickers(self):
        prices = {
            "AVUV": _make_prices(200, 0.03),
            "AVDV": _make_prices(200, -0.01),
        }
        mom = compute_etf_momentum(prices)
        assert mom["AVUV"] > 0
        assert mom["AVDV"] < 0

    def test_momentum_calculation_correct(self):
        # 126 days at 0% return -> momentum = 0
        prices = {"SPY": pd.Series([100.0] * 200)}
        mom = compute_etf_momentum(prices, lookback=126)
        assert abs(mom["SPY"]) < 0.001


class TestComputeTimingMultipliers:

    def test_negative_momentum_gets_penalty(self):
        mom  = {"AVUV": -0.05}
        mult = compute_timing_multipliers(mom)
        assert mult["AVUV"] == NEGATIVE_MOM_PENALTY

    def test_positive_momentum_gets_boost(self):
        mom  = {"AVUV": 0.10}
        mult = compute_timing_multipliers(mom)
        assert mult["AVUV"] == POSITIVE_MOM_BOOST

    def test_zero_momentum_neutral(self):
        mom  = {"AVUV": 0.0}
        mult = compute_timing_multipliers(mom)
        assert mult["AVUV"] == 1.0

    def test_custom_penalty_and_boost(self):
        mom  = {"A": 0.05, "B": -0.05}
        mult = compute_timing_multipliers(mom, penalty=0.70, boost=1.20)
        assert mult["A"] == 1.20
        assert mult["B"] == 0.70


class TestApplyFactorTiming:

    def test_total_weight_preserved(self):
        base = {"AVUV": 0.18, "AVDV": 0.22, "QMOM": 0.09, "DBMF": 0.12, "CTA": 0.05}
        # Mixed momentum
        mom  = {"AVUV": 0.10, "AVDV": -0.05, "QMOM": 0.08, "DBMF": 0.03, "CTA": -0.02}
        timed, _ = apply_factor_timing(base, mom)
        assert abs(sum(timed.values()) - sum(base.values())) < 0.001

    def test_negative_mom_etf_gets_lower_weight(self):
        base = {"AVUV": 0.18, "AVDV": 0.22}
        mom  = {"AVUV": 0.10, "AVDV": -0.10}
        timed, _ = apply_factor_timing(base, mom)
        # AVDV (negative mom) should be reduced vs AVUV (positive)
        assert timed["AVDV"] < timed["AVUV"]

    def test_all_same_momentum_no_change(self):
        base = {"AVUV": 0.18, "AVDV": 0.22, "QMOM": 0.09}
        # All positive -> all get boost -> normalize back to base
        mom  = {"AVUV": 0.10, "AVDV": 0.12, "QMOM": 0.08}
        timed, _ = apply_factor_timing(base, mom)
        for ticker in base:
            assert abs(timed[ticker] - base[ticker]) < 0.001

    def test_minimum_weight_floor(self):
        base = {"AVUV": 0.20, "AVDV": 0.20}
        # Very negative momentum for AVDV
        mom  = {"AVUV": 0.50, "AVDV": -0.50}
        timed, _ = apply_factor_timing(base, mom)
        # AVDV should not go below 50% of its target weight
        min_avdv = base["AVDV"] * MIN_WEIGHT_FRACTION
        assert timed["AVDV"] >= min_avdv - 0.001

    def test_returns_multipliers(self):
        base = {"AVUV": 0.18, "AVDV": 0.22}
        mom  = {"AVUV": 0.05, "AVDV": -0.05}
        timed, mults = apply_factor_timing(base, mom)
        assert "AVUV" in mults
        assert "AVDV" in mults

    def test_all_weights_positive(self):
        base = {"AVUV": 0.18, "AVDV": 0.22, "QMOM": 0.09, "DBMF": 0.12, "CTA": 0.05}
        mom  = {"AVUV": -0.05, "AVDV": -0.08, "QMOM": 0.03, "DBMF": -0.01, "CTA": 0.02}
        timed, _ = apply_factor_timing(base, mom)
        assert all(v > 0 for v in timed.values())


class TestFactorTimingSummary:

    def test_contains_tickers(self):
        base   = {"AVUV": 0.18, "AVDV": 0.22}
        timed  = {"AVUV": 0.19, "AVDV": 0.21}
        mom    = {"AVUV": 0.05, "AVDV": -0.03}
        mults  = {"AVUV": 1.10, "AVDV": 0.80}
        report = factor_timing_summary(base, timed, mom, mults)
        assert "AVUV" in report
        assert "AVDV" in report

    def test_contains_total_row(self):
        base   = {"AVUV": 0.18, "AVDV": 0.22}
        timed  = {"AVUV": 0.19, "AVDV": 0.21}
        mom    = {"AVUV": 0.05, "AVDV": -0.03}
        mults  = {"AVUV": 1.10, "AVDV": 0.80}
        report = factor_timing_summary(base, timed, mom, mults)
        assert "TOTAL" in report

    def test_contains_header(self):
        base   = {"AVUV": 0.18}
        report = factor_timing_summary(base, base, {"AVUV": 0.05}, {"AVUV": 1.10})
        assert "FACTOR TIMING" in report
