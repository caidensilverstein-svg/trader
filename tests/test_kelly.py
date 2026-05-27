"""Unit tests for core/kelly.py."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import math
import pytest
from core.kelly import (
    kelly_fraction,
    half_kelly_pead,
    half_kelly_ma,
    log_growth_rate,
    kelly_summary,
)


class TestKellyFraction:

    def test_positive_ev_gives_positive_kelly(self):
        # win_rate=0.6, win/loss=1 -> Kelly = 0.6 - 0.4 = 0.2
        f = kelly_fraction(0.6, 1.0, 1.0)
        assert abs(f - 0.2) < 0.001

    def test_negative_ev_gives_zero(self):
        # win_rate=0.4, 1:1 payoff -> EV negative
        f = kelly_fraction(0.4, 1.0, 1.0)
        assert f == 0.0

    def test_high_win_size_increases_kelly(self):
        f_low  = kelly_fraction(0.55, 0.05, 0.05)
        f_high = kelly_fraction(0.55, 0.15, 0.05)
        assert f_high > f_low

    def test_bad_inputs_return_zero(self):
        assert kelly_fraction(0.0, 1.0, 1.0) == 0.0
        assert kelly_fraction(1.0, 1.0, 1.0) == 0.0
        assert kelly_fraction(0.5, 0.0, 1.0) == 0.0
        assert kelly_fraction(0.5, 1.0, 0.0) == 0.0

    def test_formula_verification(self):
        # b = 2/1 = 2, p=0.6, q=0.4 -> f* = (0.6*2 - 0.4)/2 = 0.8/2 = 0.4
        f = kelly_fraction(0.6, 2.0, 1.0)
        assert abs(f - 0.4) < 0.001

    def test_returns_float(self):
        f = kelly_fraction(0.55, 0.08, 0.07)
        assert isinstance(f, float)


class TestHalfKellyPead:

    def test_within_bounds(self):
        notional = half_kelly_pead()
        assert 2_000 <= notional <= 5_000

    def test_higher_win_rate_increases_size(self):
        low  = half_kelly_pead(win_rate=0.50)
        high = half_kelly_pead(win_rate=0.65)
        assert high >= low

    def test_custom_capital(self):
        n = half_kelly_pead(capital=200_000, max_notional=10_000)
        assert n <= 10_000

    def test_clamped_to_min(self):
        # Very low win rate -> Kelly near zero -> clamped to min
        n = half_kelly_pead(win_rate=0.45)  # EV negative -> Kelly=0 -> clamped
        assert n == 2_000


class TestHalfKellyMA:

    def test_within_bounds(self):
        notional = half_kelly_ma(spread_pct=0.02)
        assert 1_500 <= notional <= 3_500

    def test_higher_spread_increases_size(self):
        low  = half_kelly_ma(spread_pct=0.005)
        high = half_kelly_ma(spread_pct=0.05)
        assert high >= low

    def test_zero_spread_gives_min(self):
        n = half_kelly_ma(spread_pct=0.0)
        assert n == 1_500


class TestLogGrowthRate:

    def test_full_kelly_beats_half(self):
        g_full = log_growth_rate(0.1563, 0.55, 0.08, 0.07)
        g_half = log_growth_rate(0.0781, 0.55, 0.08, 0.07)
        assert g_full >= g_half

    def test_zero_fraction_returns_zero(self):
        g = log_growth_rate(0.0, 0.55, 0.08, 0.07)
        assert g == 0.0

    def test_positive_ev_gives_positive_growth(self):
        g = log_growth_rate(0.10, 0.55, 0.08, 0.07)
        assert g > 0


class TestKellySummary:

    def test_returns_required_keys(self):
        s = kelly_summary(0.55, 0.08, 0.07, 100_000, "Test")
        required = [
            "win_rate", "win_size", "loss_size", "expected_value",
            "kelly_fraction", "half_kelly", "notional_full", "notional_half",
        ]
        for k in required:
            assert k in s, f"Missing key: {k}"

    def test_half_kelly_is_half_full(self):
        s = kelly_summary(0.55, 0.08, 0.07, 100_000)
        assert abs(s["half_kelly"] - s["kelly_fraction"] / 2) < 0.001

    def test_notional_proportional_to_capital(self):
        s1 = kelly_summary(0.55, 0.08, 0.07, 100_000)
        s2 = kelly_summary(0.55, 0.08, 0.07, 200_000)
        assert abs(s2["notional_half"] / s1["notional_half"] - 2.0) < 0.001
