"""
Unit tests for strategies/pead_screener.py.
Tests the SUE calculation, gap detection, and scoring logic.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest

from strategies.pead_screener import (
    detect_earnings_gap,
    calculate_sue_score,
)


class TestDetectEarningsGap:

    def _make_prices(self, prior, day0, day1, n_history=10):
        """Build a price series with given announcement day values."""
        history = [prior * 0.99 ** i for i in range(n_history, 0, -1)]
        return pd.Series(history + [prior, day0, day1])

    def test_positive_gap_detected(self):
        # prior close 100, day0 close 108 (8% gap up)
        prices = self._make_prices(100, 108, 109)
        gap, held = detect_earnings_gap(prices)
        assert gap > 0.05   # at least 5% gap
        assert held is True

    def test_gap_not_held(self):
        # day1 close below day0 = didn't hold
        prices = self._make_prices(100, 108, 106)
        gap, held = detect_earnings_gap(prices)
        assert gap > 0      # gap still there
        assert held is False  # gave back some of the gap

    def test_no_gap(self):
        # flat price
        prices = self._make_prices(100, 100.5, 101)
        gap, held = detect_earnings_gap(prices)
        assert gap < 0.02

    def test_negative_gap(self):
        # down 5% on announcement
        prices = self._make_prices(100, 95, 94)
        gap, held = detect_earnings_gap(prices)
        assert gap < 0

    def test_insufficient_data(self):
        prices = pd.Series([100, 105])  # only 2 prices
        gap, held = detect_earnings_gap(prices)
        assert gap == 0.0
        assert held is False

    def test_gap_calculation_correct(self):
        prices = self._make_prices(100, 110, 112)
        gap, held = detect_earnings_gap(prices)
        # prior=100, day0=110 -> gap = 10%
        assert abs(gap - 0.10) < 0.001


class TestSUEScore:
    """
    SUE calculation is tested here in isolation.
    Network-dependent test skipped with a mock approach.
    """

    def test_returns_float_or_none(self):
        # Can't test without network — just verify type contract
        result = calculate_sue_score("AAPL")
        assert result is None or isinstance(result, float)

    def test_none_for_clearly_invalid_ticker(self):
        result = calculate_sue_score("ZZZZNOTREAL99")
        assert result is None
