"""
Unit tests for strategies/etf_manager.py.
No network calls, no Alpaca connection.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest

import config
from strategies.etf_manager import (
    compute_bsc_scalar,
    compute_effective_weights,
    compute_drift,
    needs_rebalance,
)


class TestBSCScalar:

    def _make_prices(self, n: int, vol: float) -> pd.Series:
        """Generate price series with known volatility."""
        np.random.seed(42)
        daily_vol = vol / (252 ** 0.5)
        returns = np.random.normal(0, daily_vol, n)
        prices = [100.0]
        for r in returns:
            prices.append(prices[-1] * (1 + r))
        return pd.Series(prices)

    def test_high_vol_gives_low_scalar(self):
        # Realized vol = 25% >> target 12% -> scalar < 1
        prices = self._make_prices(200, 0.25)
        scalar = compute_bsc_scalar(prices)
        assert scalar < 1.0

    def test_low_vol_gives_high_scalar(self):
        # Realized vol = 5% << target 12% -> scalar > 1
        prices = self._make_prices(200, 0.05)
        scalar = compute_bsc_scalar(prices)
        assert scalar > 1.0

    def test_scalar_clipped_to_min(self):
        # Extreme high vol -> clipped at BSC_MIN_SCALAR
        prices = self._make_prices(200, 0.50)
        scalar = compute_bsc_scalar(prices)
        assert scalar == config.BSC_MIN_SCALAR

    def test_scalar_clipped_to_max(self):
        # Extreme low vol -> clipped at BSC_MAX_SCALAR
        prices = self._make_prices(200, 0.01)
        scalar = compute_bsc_scalar(prices)
        assert scalar == config.BSC_MAX_SCALAR

    def test_insufficient_data_returns_min(self):
        prices = pd.Series([100.0, 101.0, 99.0])  # too few
        scalar = compute_bsc_scalar(prices)
        assert scalar == config.BSC_MIN_SCALAR

    def test_scalar_is_float(self):
        prices = self._make_prices(200, 0.15)
        scalar = compute_bsc_scalar(prices)
        assert isinstance(scalar, float)


class TestEffectiveWeights:

    def test_total_weight_bull(self):
        """In BULL regime with scalar=1, QMOM=18%, total=75%."""
        weights = compute_effective_weights(1.0, "BULL")
        total = sum(weights.values())
        # BULL mult=1.0, base sums to (18+22+18+12+5)/100 = 0.75
        assert abs(total - 0.75) < 0.001

    def test_qmom_halved_at_0_5_scalar(self):
        weights = compute_effective_weights(0.5, "BULL")
        assert abs(weights["QMOM"] - 0.09) < 0.001  # 18% * 0.5 = 9%

    def test_bear_crisis_reduces_all(self):
        bull_w = compute_effective_weights(1.0, "BULL")
        crisis_w = compute_effective_weights(1.0, "BEAR_CRISIS")
        assert all(crisis_w[t] < bull_w[t] for t in bull_w)

    def test_all_weights_nonnegative(self):
        for regime in ("BULL", "MILD_BULL", "SIDEWAYS", "BEAR", "BEAR_CRISIS"):
            w = compute_effective_weights(0.5, regime)
            assert all(v >= 0 for v in w.values())

    def test_regime_mult_applied(self):
        bear_w = compute_effective_weights(1.0, "BEAR")
        bull_w = compute_effective_weights(1.0, "BULL")
        bear_mult = config.REGIME_ETF_MULT["BEAR"]
        bull_mult = config.REGIME_ETF_MULT["BULL"]
        for ticker in bull_w:
            expected_ratio = bear_mult / bull_mult
            assert abs(bear_w[ticker] / bull_w[ticker] - expected_ratio) < 0.001


class TestDriftAndRebalance:

    def _make_values(self, weights: dict, total: float) -> dict:
        return {t: w * total for t, w in weights.items()}

    def test_no_drift_no_rebalance(self):
        targets = {"AVUV": 0.18, "AVDV": 0.22, "QMOM": 0.09, "DBMF": 0.12, "CTA": 0.05}
        values = self._make_values(targets, 100_000)
        drift = compute_drift(values, targets, 100_000)
        assert not needs_rebalance(drift)

    def test_large_drift_triggers_rebalance(self):
        targets = {"AVUV": 0.18, "AVDV": 0.22, "QMOM": 0.09, "DBMF": 0.12, "CTA": 0.05}
        # AVUV drifted to 25% (7pp over target)
        values = {"AVUV": 25_000, "AVDV": 22_000, "QMOM": 9_000, "DBMF": 12_000, "CTA": 5_000}
        drift = compute_drift(values, targets, 100_000)
        assert needs_rebalance(drift)

    def test_drift_sign_correct(self):
        targets = {"AVUV": 0.18}
        values = {"AVUV": 20_000}  # 20% vs target 18% -> overweight
        drift = compute_drift(values, targets, 100_000)
        assert drift["AVUV"] > 0  # overweight = positive drift

    def test_missing_ticker_zero_drift(self):
        """Ticker not in positions => 0 value => negative drift."""
        targets = {"AVUV": 0.18}
        values = {}  # not held
        drift = compute_drift(values, targets, 100_000)
        assert drift["AVUV"] < 0  # underweight

    def test_threshold_boundary(self):
        targets = {"AVUV": 0.18}
        # 22.9% = 4.9pp over 18% -> just below threshold, no rebalance
        values = {"AVUV": 22_900}
        drift = compute_drift(values, targets, 100_000)
        assert not needs_rebalance(drift)

        # 23.6% = 5.6pp over -> clearly over threshold -> triggers
        values = {"AVUV": 23_600}
        drift = compute_drift(values, targets, 100_000)
        assert needs_rebalance(drift)
