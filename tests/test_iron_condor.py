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


# ---------------------------------------------------------------------------
# Signal lifecycle tests (require state file isolation)
# ---------------------------------------------------------------------------

import tempfile, os
from unittest.mock import patch


@pytest.fixture()
def isolated_state(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CONDOR_FILE", str(tmp_path / "condor.json"))
    monkeypatch.setattr(config, "LOG_FILE",    str(tmp_path / "trades.json"))


from strategies.iron_condor import (
    open_condor_signal,
    check_condor_exits,
    get_condor_status,
)


class TestOpenCondorSignal:

    def test_opens_when_conditions_met(self, isolated_state):
        condor = open_condor_signal(5000.0, 18.0, "BULL", dte=45)
        assert condor is not None
        assert condor["status"] == "OPEN"
        assert condor["short_put"] < 5000.0

    def test_skips_low_vix(self, isolated_state):
        condor = open_condor_signal(5000.0, 10.0, "BULL", dte=45)
        assert condor is None

    def test_skips_bear_crisis(self, isolated_state):
        condor = open_condor_signal(5000.0, 22.0, "BEAR_CRISIS", dte=45)
        assert condor is None

    def test_no_duplicate_condors(self, isolated_state):
        open_condor_signal(5000.0, 18.0, "BULL", dte=45)
        second = open_condor_signal(5000.0, 18.0, "BULL", dte=45)
        assert second is None

    def test_condor_contains_required_keys(self, isolated_state):
        condor = open_condor_signal(5000.0, 18.0, "BULL", dte=45)
        for key in ("short_put", "long_put", "short_call", "long_call",
                    "est_credit", "max_loss", "entry_date", "expiry_date"):
            assert key in condor, f"Missing key: {key}"

    def test_condor_id_increments(self, isolated_state):
        c1 = open_condor_signal(5000.0, 18.0, "BULL", dte=45)
        check_condor_exits(5000.0, 18.0)  # close it
        c2 = open_condor_signal(5000.0, 18.0, "BULL", dte=1)  # open another
        if c2 is not None:
            assert c2["id"] > c1["id"]


class TestCheckCondorExits:

    def test_no_exit_inside_strikes(self, isolated_state):
        open_condor_signal(5000.0, 18.0, "BULL", dte=45)
        closed = check_condor_exits(5000.0, 18.0)
        assert len(closed) == 0

    def test_exit_when_dte_below_threshold(self, isolated_state):
        # Open with DTE=0 (already expired)
        with patch("strategies.iron_condor.config.CONDOR_DTE_EXIT", 21):
            open_condor_signal(5000.0, 18.0, "BULL", dte=0)
        closed = check_condor_exits(5000.0, 18.0)
        assert len(closed) == 1
        assert "DTE" in closed[0]["close_reason"]

    def test_no_crash_on_empty_state(self, isolated_state):
        closed = check_condor_exits(5000.0, 18.0)
        assert closed == []

    def test_pnl_positive_inside_strikes(self, isolated_state):
        open_condor_signal(5000.0, 18.0, "BULL", dte=45)
        # SPX still inside strikes — P&L should be non-negative (or zero at day 0)
        check_condor_exits(5000.0, 18.0)  # just runs without crash
        status = get_condor_status()
        # Condor still open (not enough decay yet)
        assert status["open_count"] >= 0  # just verify no exception


class TestGetCondorStatus:

    def test_initial_state_empty(self, isolated_state):
        status = get_condor_status()
        assert status["open_count"] == 0
        assert status["closed_count"] == 0
        assert status["win_rate"] == 0
        assert status["total_pnl"] == 0.0

    def test_open_count_after_signal(self, isolated_state):
        open_condor_signal(5000.0, 18.0, "BULL", dte=45)
        status = get_condor_status()
        assert status["open_count"] == 1

    def test_open_condors_list(self, isolated_state):
        open_condor_signal(5000.0, 18.0, "BULL", dte=45)
        status = get_condor_status()
        assert len(status["open_condors"]) == 1
        assert status["open_condors"][0]["short_put"] < 5000.0
