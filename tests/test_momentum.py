"""Unit tests for core/momentum_timing.py."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest

from core.momentum_timing import (
    spy_time_series_momentum,
    etf_momentum_scores,
    combined_regime_signal,
)


def _make_trend(n: int, monthly_return: float) -> pd.Series:
    """Generate a price series with known monthly return."""
    daily_r = (1 + monthly_return) ** (1 / 21) - 1
    prices  = [100.0]
    for _ in range(n - 1):
        prices.append(prices[-1] * (1 + daily_r))
    return pd.Series(prices)


class TestSpyMomentum:

    def test_positive_trend_gives_positive_composite(self):
        prices = _make_trend(300, 0.03)  # +3%/month trend
        result = spy_time_series_momentum(prices)
        assert result["composite"] > 0
        assert result["signal"] == "positive"

    def test_negative_trend_gives_negative_composite(self):
        prices = _make_trend(300, -0.02)  # -2%/month trend
        result = spy_time_series_momentum(prices)
        assert result["composite"] < 0
        assert result["signal"] == "negative"

    def test_flat_market_negative_due_to_rf(self):
        # Flat market (0% return) should have negative excess vs 5% risk-free
        prices = pd.Series([100.0] * 300)
        result = spy_time_series_momentum(prices)
        assert result["composite"] < 0

    def test_insufficient_data_returns_none_for_long_windows(self):
        # Only 50 days of data -- 12-month lookback should be None
        prices = _make_trend(50, 0.01)
        result = spy_time_series_momentum(prices)
        assert result.get("mom_12m") is None

    def test_returns_required_keys(self):
        prices = _make_trend(300, 0.02)
        result = spy_time_series_momentum(prices)
        assert "composite" in result
        assert "signal" in result


class TestETFMomentumScores:

    def test_positive_trend_all_positive(self):
        prices = {"AVUV": _make_trend(200, 0.02), "AVDV": _make_trend(200, 0.01)}
        scores = etf_momentum_scores(prices)
        for t, s in scores.items():
            assert s["signal"] == "positive", f"{t} should be positive"

    def test_negative_trend_all_negative(self):
        prices = {"AVUV": _make_trend(200, -0.02)}
        scores = etf_momentum_scores(prices)
        assert scores["AVUV"]["signal"] == "negative"

    def test_insufficient_data(self):
        prices = {"CTA": _make_trend(50, 0.01)}  # < 126 days
        scores = etf_momentum_scores(prices)
        assert scores["CTA"]["mom_6m"] is None


class TestCombinedSignal:

    def test_bull_positive_mom_is_aggressive(self):
        assert combined_regime_signal("BULL", 5.0) == "AGGRESSIVE"

    def test_bull_negative_mom_is_cautious(self):
        assert combined_regime_signal("BULL", -1.0) == "CAUTIOUS"

    def test_bear_negative_mom_is_defensive(self):
        assert combined_regime_signal("BEAR", -2.0) == "DEFENSIVE"
        assert combined_regime_signal("BEAR_CRISIS", -3.0) == "DEFENSIVE"

    def test_bear_positive_mom_is_cautious(self):
        assert combined_regime_signal("BEAR", 2.0) == "CAUTIOUS"

    def test_mild_bull_positive_is_aggressive(self):
        assert combined_regime_signal("MILD_BULL", 3.0) == "AGGRESSIVE"

    def test_sideways_neutral(self):
        result = combined_regime_signal("SIDEWAYS", 1.0)
        assert result == "NEUTRAL"
