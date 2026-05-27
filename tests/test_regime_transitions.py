"""Unit tests for core/regime_transitions.py."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from core.regime_transitions import (
    compute_transition_matrix,
    expected_dwell_time,
    stationary_distribution,
    regime_persistence_score,
    format_transition_report,
    REGIMES,
)


def _simple_sequence():
    """Deterministic regime sequence for predictable tests."""
    return ["BULL", "BULL", "BULL", "BEAR", "BULL", "BULL",
            "SIDEWAYS", "BULL", "BULL", "BEAR", "BEAR", "BULL"]


class TestComputeTransitionMatrix:

    def test_rows_sum_to_one_for_observed_regimes(self):
        tm = compute_transition_matrix(_simple_sequence())
        for regime in ["BULL", "BEAR", "SIDEWAYS"]:
            total = sum(tm[regime].values())
            assert abs(total - 1.0) < 0.001, f"{regime} row sum={total}"

    def test_unobserved_regime_sums_to_zero(self):
        seq = ["BULL", "BEAR", "BULL"]
        tm  = compute_transition_matrix(seq)
        # MILD_BULL and BEAR_CRISIS never observed
        total = sum(tm["MILD_BULL"].values())
        assert total == 0.0

    def test_absorbing_state_self_transition_is_one(self):
        # Only one regime observed
        seq = ["BULL"] * 10
        tm  = compute_transition_matrix(seq)
        assert abs(tm["BULL"]["BULL"] - 1.0) < 0.001

    def test_high_bull_persistence_in_bull_sequence(self):
        seq = ["BULL"] * 8 + ["BEAR"] + ["BULL"]
        tm  = compute_transition_matrix(seq)
        assert tm["BULL"]["BULL"] > 0.7

    def test_short_sequence_does_not_crash(self):
        tm = compute_transition_matrix(["BULL"])
        assert isinstance(tm, dict)

    def test_empty_sequence_returns_zeros(self):
        tm = compute_transition_matrix([])
        for r in REGIMES:
            assert r in tm

    def test_all_regimes_present_as_keys(self):
        tm = compute_transition_matrix(_simple_sequence())
        for r in REGIMES:
            assert r in tm


class TestExpectedDwellTime:

    def test_absorbing_state_is_infinite(self):
        tm = {"BULL": {"BULL": 1.0, **{r: 0.0 for r in REGIMES if r != "BULL"}}}
        for r in REGIMES:
            if r != "BULL":
                tm[r] = {r2: 0.0 for r2 in REGIMES}
        dwell = expected_dwell_time(tm)
        assert dwell["BULL"] == float("inf")

    def test_no_self_transition_is_one(self):
        # P_ii = 0 -> dwell = 1 / (1 - 0) = 1
        tm = compute_transition_matrix(["BULL", "BEAR", "BULL", "BEAR"])
        dwell = expected_dwell_time(tm)
        # BEAR has 0% self-transition in this sequence
        assert dwell["BEAR"] == pytest.approx(1.0, abs=0.1)

    def test_high_persistence_high_dwell(self):
        seq = ["BULL"] * 9 + ["BEAR"]
        tm  = compute_transition_matrix(seq)
        dwell = expected_dwell_time(tm)
        assert dwell["BULL"] > 5.0


class TestStationaryDistribution:

    def test_sums_to_one(self):
        tm    = compute_transition_matrix(_simple_sequence())
        stat  = stationary_distribution(tm)
        total = sum(stat.values())
        assert abs(total - 1.0) < 0.01

    def test_all_probs_non_negative(self):
        tm   = compute_transition_matrix(_simple_sequence())
        stat = stationary_distribution(tm)
        for r, p in stat.items():
            assert p >= -0.001, f"{r} has negative prob {p}"

    def test_bull_dominant_in_bull_heavy_sequence(self):
        seq  = ["BULL"] * 7 + ["BEAR"] + ["BULL"] * 2 + ["BEAR"]
        tm   = compute_transition_matrix(seq)
        stat = stationary_distribution(tm)
        # BULL should dominate
        assert stat["BULL"] > 0.5


class TestRegimePersistenceScore:

    def test_high_persistence_for_bull_heavy(self):
        seq   = ["BULL"] * 8 + ["BEAR"]
        tm    = compute_transition_matrix(seq)
        score = regime_persistence_score("BULL", tm)
        assert score > 0.7

    def test_zero_for_unobserved_regime(self):
        seq   = ["BULL", "BEAR"]
        tm    = compute_transition_matrix(seq)
        score = regime_persistence_score("BEAR_CRISIS", tm)
        assert score == 0.0


class TestFormatTransitionReport:

    def test_report_contains_header(self):
        report = format_transition_report(_simple_sequence(), "BULL")
        assert "REGIME TRANSITION ANALYSIS" in report

    def test_report_contains_all_regimes(self):
        report = format_transition_report(_simple_sequence(), "BULL")
        for r in REGIMES:
            assert r[:5] in report

    def test_report_contains_current_regime(self):
        report = format_transition_report(_simple_sequence(), "BEAR")
        assert "Current Regime: BEAR" in report

    def test_report_shows_persistence(self):
        report = format_transition_report(_simple_sequence(), "BULL")
        assert "Persistence Probability" in report

    def test_report_contains_stationary_dist(self):
        report = format_transition_report(_simple_sequence(), "BULL")
        assert "Stationary Distribution" in report
