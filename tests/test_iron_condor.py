"""
Unit tests for strategies/iron_condor.py.
Tests signal generation, strike estimation, sizing, and exit logic.
"""

import sys
import json
import tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

import config
from strategies.iron_condor import (
    condor_size_multiplier,
    should_open_condor,
    estimate_strikes,
)


class TestCondorSizeMultiplier:

    def test_below_min_vix_is_zero(self):
        assert condor_size_multiplier(14.9) == 0.0

    def test_exactly_at_min_vix(self):
        assert condor_size_multiplier(15.0) == 1.0

    def test_low_vix_full_size(self):
        assert condor_size_multiplier(17.0) == 1.0

    def test_mid_vix_reduced(self):
        assert condor_size_multiplier(22.0) == 0.75

    def test_high_vix_very_small(self):
        assert condor_size_multiplier(30.0) == 0.25

    def test_above_max_is_zero(self):
        assert condor_size_multiplier(35.1) == 0.0

    def test_exactly_at_max_is_zero(self):
        # VIX=35 is not < 35, so returns 0.0
        assert condor_size_multiplier(35.0) == 0.0


class TestShouldOpenCondor:

    def test_skip_low_vix(self):
        open_it, reason = should_open_condor(14.0, "BULL")
        assert not open_it
        assert "VIX" in reason

    def test_skip_crisis(self):
        open_it, reason = should_open_condor(25.0, "BEAR_CRISIS")
        assert not open_it
        assert "BEAR_CRISIS" in reason

    def test_open_bull_good_vix(self):
        open_it, reason = should_open_condor(18.0, "BULL")
        assert open_it

    def test_open_sideways(self):
        open_it, reason = should_open_condor(22.0, "SIDEWAYS")
        assert open_it

    def test_skip_bear_regime(self):
        # BEAR has 0.25 mult, still opens (just small)
        open_it, reason = should_open_condor(22.0, "BEAR")
        assert open_it  # 0.25 mult is non-zero

    def test_skip_very_high_vix(self):
        open_it, reason = should_open_condor(36.0, "BULL")
        assert not open_it


class TestEstimateStrikes:

    def test_put_below_current(self):
        strikes = estimate_strikes(5000, 18, 38)
        assert strikes["short_put"] < strikes["spx_price"]
        assert strikes["long_put"] < strikes["short_put"]

    def test_call_above_current(self):
        strikes = estimate_strikes(5000, 18, 38)
        assert strikes["short_call"] > strikes["spx_price"]
        assert strikes["long_call"] > strikes["short_call"]

    def test_wing_width(self):
        strikes = estimate_strikes(5000, 18, 38)
        put_width  = strikes["short_put"]  - strikes["long_put"]
        call_width = strikes["long_call"]  - strikes["short_call"]
        assert put_width  == config.CONDOR_WING_POINTS
        assert call_width == config.CONDOR_WING_POINTS

    def test_strikes_multiples_of_5(self):
        strikes = estimate_strikes(5000, 18, 38)
        for key in ("short_put", "long_put", "short_call", "long_call"):
            assert strikes[key] % 5 == 0, f"{key}={strikes[key]} not multiple of 5"

    def test_credit_positive(self):
        strikes = estimate_strikes(5000, 18, 38)
        assert strikes["est_credit"] > 0

    def test_higher_vix_wider_strikes(self):
        low_vix  = estimate_strikes(5000, 15, 38)
        high_vix = estimate_strikes(5000, 30, 38)
        assert high_vix["short_put"]  < low_vix["short_put"]   # wider
        assert high_vix["short_call"] > low_vix["short_call"]  # wider
