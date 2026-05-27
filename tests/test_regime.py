"""
Unit tests for core/regime.py.
All tests use synthetic data — no network calls.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest

from core.regime import get_regime, compute_regime_indicators, regime_from_history


# ---------------------------------------------------------------------------
# get_regime — pure logic tests
# ---------------------------------------------------------------------------

class TestGetRegime:

    def test_bear_crisis_high_vix(self):
        assert get_regime(500, 505, 0.02, 32, -0.05) == "BEAR_CRISIS"

    def test_bear_crisis_deep_drawdown(self):
        assert get_regime(400, 505, -0.10, 18, -0.22) == "BEAR_CRISIS"

    def test_bear(self):
        assert get_regime(480, 505, -0.08, 22, -0.05) == "BEAR"

    def test_bull_all_green(self):
        assert get_regime(520, 505, 0.05, 15, -0.02) == "BULL"

    def test_mild_bull_above_ma_low_vix(self):
        # above 200MA, low VIX, but no positive momentum
        assert get_regime(510, 505, -0.01, 18, -0.05) == "MILD_BULL"

    def test_sideways_mixed(self):
        # above 200MA but VIX elevated
        assert get_regime(510, 505, 0.02, 22, -0.05) == "SIDEWAYS"

    def test_bull_boundary_vix_exactly_20(self):
        # VIX exactly 20: low_vix = (20 < 20) = False -> MILD_BULL
        assert get_regime(510, 505, 0.05, 20, -0.02) in ("MILD_BULL", "SIDEWAYS")

    def test_bear_crisis_overrides_bull_signals(self):
        # VIX>30 overrides everything including positive momentum
        assert get_regime(520, 505, 0.08, 35, -0.02) == "BEAR_CRISIS"


# ---------------------------------------------------------------------------
# compute_regime_indicators
# ---------------------------------------------------------------------------

class TestComputeRegimeIndicators:

    def _make_spy(self, n: int = 252, trend: float = 0.0005) -> pd.Series:
        """Create synthetic SPY prices with a given daily drift."""
        prices = [500.0]
        np.random.seed(42)
        for _ in range(n - 1):
            prices.append(prices[-1] * (1 + trend + np.random.normal(0, 0.01)))
        return pd.Series(prices)

    def _make_vix(self, n: int = 252, level: float = 18.0) -> pd.Series:
        np.random.seed(99)
        vix = np.random.normal(level, 1.5, n)
        return pd.Series(np.clip(vix, 10, 80))

    def test_returns_five_values(self):
        spy = self._make_spy()
        vix = self._make_vix()
        result = compute_regime_indicators(spy, vix)
        assert len(result) == 5

    def test_ma200_correct(self):
        spy = self._make_spy()
        vix = self._make_vix()
        price, ma200, mom60, v, dd = compute_regime_indicators(spy, vix)
        expected_ma200 = spy.rolling(200).mean().iloc[-1]
        assert abs(ma200 - expected_ma200) < 0.01

    def test_insufficient_data_raises(self):
        spy = pd.Series([500.0] * 100)  # only 100 days
        vix = pd.Series([18.0] * 100)
        with pytest.raises(ValueError):
            compute_regime_indicators(spy, vix)

    def test_drawdown_negative(self):
        spy = self._make_spy()
        vix = self._make_vix()
        _, _, _, _, dd = compute_regime_indicators(spy, vix)
        assert dd <= 0.0


# ---------------------------------------------------------------------------
# regime_from_history — integration
# ---------------------------------------------------------------------------

class TestRegimeFromHistory:

    def test_bull_trending_market(self):
        np.random.seed(1)
        # Trending UP market (positive drift)
        prices = [500.0]
        for _ in range(251):
            prices.append(prices[-1] * (1 + 0.001 + np.random.normal(0, 0.005)))
        spy = pd.Series(prices)
        vix = pd.Series([15.0] * 252)  # low VIX
        result = regime_from_history(spy, vix)
        assert result in ("BULL", "MILD_BULL")

    def test_bear_crisis_very_high_vix(self):
        np.random.seed(2)
        prices = [500.0]
        for _ in range(251):
            prices.append(prices[-1] * (1 + np.random.normal(0, 0.01)))
        spy = pd.Series(prices)
        vix = pd.Series([40.0] * 252)  # very high VIX
        result = regime_from_history(spy, vix)
        assert result == "BEAR_CRISIS"
