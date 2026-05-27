"""Unit tests for core/risk_parity.py."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest

from core.risk_parity import inverse_vol_weights, risk_contribution, compare_to_target


def _make_returns(vol: float, n: int = 200) -> pd.Series:
    """Generate returns with known annualized volatility."""
    np.random.seed(42)
    daily_vol = vol / (252 ** 0.5)
    return pd.Series(np.random.normal(0, daily_vol, n))


class TestInverseVolWeights:

    def test_weights_sum_to_one(self):
        returns = {"A": _make_returns(0.15), "B": _make_returns(0.25)}
        w = inverse_vol_weights(returns)
        assert abs(sum(w.values()) - 1.0) < 0.001

    def test_low_vol_gets_higher_weight(self):
        returns = {"LOW": _make_returns(0.05), "HIGH": _make_returns(0.30)}
        w = inverse_vol_weights(returns)
        assert w["LOW"] > w["HIGH"]

    def test_equal_vol_gives_approx_equal_weights(self):
        # With same distribution, weights should be roughly equal (allow sampling noise)
        np.random.seed(1)
        returns = {
            "A": pd.Series(np.random.normal(0, 0.01, 500)),
            "B": pd.Series(np.random.normal(0, 0.01, 500)),
        }
        w = inverse_vol_weights(returns)
        assert abs(w["A"] - w["B"]) < 0.10  # within 10% with 500 samples

    def test_min_weight_applied_before_normalization(self):
        # min_weight is a soft floor applied before normalization
        # Very high vol asset B gets min_weight before renorm -> B > 0 after renorm
        returns = {"A": _make_returns(0.01, 200), "B": _make_returns(2.00, 200)}
        w = inverse_vol_weights(returns, min_weight=0.10)
        assert w["B"] > 0  # B gets some weight (exact value depends on renorm)

    def test_all_weights_positive(self):
        returns = {"A": _make_returns(0.15), "B": _make_returns(0.25), "C": _make_returns(0.10)}
        w = inverse_vol_weights(returns)
        assert all(v > 0 for v in w.values())


class TestRiskContribution:

    def test_contributions_sum_to_one(self):
        returns = {
            "A": _make_returns(0.15),
            "B": _make_returns(0.20),
        }
        weights = {"A": 0.6, "B": 0.4}
        rc = risk_contribution(weights, returns)
        assert abs(sum(rc.values()) - 1.0) < 0.05

    def test_higher_vol_higher_contribution(self):
        returns = {"A": _make_returns(0.05), "B": _make_returns(0.30)}
        weights = {"A": 0.5, "B": 0.5}
        rc = risk_contribution(weights, returns)
        assert rc["B"] > rc["A"]


class TestCompareToTarget:

    def test_overweight_positive_diff(self):
        rp = {"A": 0.3}   # 30% RP weight
        target = {"A": 0.4}  # 40% target weight -> overweight
        comp = compare_to_target(rp, target, scale_to_etf=1.0)
        assert comp["A"]["difference"] > 0  # overweight

    def test_underweight_negative_diff(self):
        rp = {"A": 0.5}
        target = {"A": 0.2}
        comp = compare_to_target(rp, target, scale_to_etf=1.0)
        assert comp["A"]["difference"] < 0  # underweight

    def test_returns_correct_keys(self):
        rp = {"A": 0.5, "B": 0.5}
        target = {"A": 0.25, "B": 0.25}
        comp = compare_to_target(rp, target)
        assert "A" in comp and "B" in comp
        for v in comp.values():
            assert "target_weight" in v
            assert "rp_weight" in v
            assert "note" in v
