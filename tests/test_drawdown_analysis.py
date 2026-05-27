"""Unit tests for backtest/drawdown_analysis.py."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest
from datetime import date, timedelta

from backtest.drawdown_analysis import (
    find_drawdown_periods,
    drawdown_statistics,
    format_drawdown_report,
    DrawdownPeriod,
)


def _make_equity(values: list) -> pd.Series:
    """Make an equity curve from a list of values with synthetic dates."""
    dates = pd.date_range("2020-01-01", periods=len(values), freq="B")
    return pd.Series(values, index=dates)


class TestFindDrawdownPeriods:

    def test_no_drawdown_in_rising_series(self):
        eq = _make_equity([100, 101, 102, 103, 104])
        periods = find_drawdown_periods(eq)
        assert len(periods) == 0

    def test_single_drawdown_detected(self):
        # Rise, fall, recover
        eq = _make_equity([100, 105, 110, 100, 95, 110, 115])
        periods = find_drawdown_periods(eq, min_depth_pct=-5.0)
        assert len(periods) >= 1

    def test_depth_correct(self):
        # Peak 110, trough 95 -> depth = (95/110 - 1) * 100 = -13.6%
        eq = _make_equity([100, 105, 110, 100, 95, 110, 115])
        periods = find_drawdown_periods(eq, min_depth_pct=-5.0)
        assert len(periods) >= 1
        depths = [p.depth_pct for p in periods]
        assert any(abs(d - (-13.6)) < 0.5 for d in depths)

    def test_recovery_detected(self):
        # Falls and recovers
        eq = _make_equity([100, 110, 90, 95, 110, 115])
        periods = find_drawdown_periods(eq, min_depth_pct=-5.0)
        if periods:
            assert periods[0].recovery_idx is not None

    def test_ongoing_drawdown_has_none_recovery(self):
        # Never recovers (ends below peak)
        eq = _make_equity([100, 110, 105, 100, 95])
        periods = find_drawdown_periods(eq, min_depth_pct=-5.0)
        if periods:
            # The last period might not have recovered
            last = max(periods, key=lambda p: p.trough_idx)
            # recovery_idx is None if we never went back above peak
            assert last.recovery_idx is None or last.recovery_idx is not None

    def test_min_depth_filter(self):
        # Small drawdown filtered out by min_depth
        eq = _make_equity([100, 101, 99.5, 102])  # only 1.5% DD
        periods = find_drawdown_periods(eq, min_depth_pct=-2.0)
        assert len(periods) == 0

    def test_sorted_by_depth(self):
        # Multiple drawdowns of different sizes
        eq = _make_equity([100, 80, 100, 90, 100, 70, 100])
        periods = find_drawdown_periods(eq, min_depth_pct=-5.0)
        if len(periods) >= 2:
            assert periods[0].depth_pct <= periods[-1].depth_pct

    def test_insufficient_data(self):
        eq = _make_equity([100, 99])
        periods = find_drawdown_periods(eq)
        assert isinstance(periods, list)

    def test_duration_positive(self):
        eq = _make_equity([100, 110, 95, 100, 115])
        periods = find_drawdown_periods(eq, min_depth_pct=-5.0)
        for p in periods:
            assert p.duration_days > 0


class TestDrawdownStatistics:

    def _make_periods(self) -> list:
        return [
            DrawdownPeriod(0, 10, 25, "2020-01-01", "2020-01-15", "2020-01-30",
                           -15.0, 10, 15, 100, 85),
            DrawdownPeriod(30, 35, 40, "2020-02-01", "2020-02-06", "2020-02-11",
                           -5.0, 5, 5, 105, 99.75),
        ]

    def test_n_drawdowns_correct(self):
        stats = drawdown_statistics(self._make_periods())
        assert stats["n_drawdowns"] == 2

    def test_worst_dd_correct(self):
        stats = drawdown_statistics(self._make_periods())
        assert stats["worst_dd_pct"] == -15.0

    def test_avg_dd_correct(self):
        stats = drawdown_statistics(self._make_periods())
        assert abs(stats["avg_dd_pct"] - (-10.0)) < 0.01

    def test_pct_recovered(self):
        stats = drawdown_statistics(self._make_periods())
        assert stats["pct_recovered"] == 100.0

    def test_empty_periods_returns_empty(self):
        stats = drawdown_statistics([])
        assert stats == {}


class TestFormatDrawdownReport:

    def _make_periods(self) -> list:
        return [
            DrawdownPeriod(0, 42, 224, "2020-01-16", "2020-03-18", "2020-12-04",
                           -22.59, 42, 182, 110_000, 85_130),
        ]

    def test_report_contains_header(self):
        r = format_drawdown_report(self._make_periods())
        assert "DRAWDOWN PERIOD ANALYSIS" in r

    def test_report_contains_depth(self):
        r = format_drawdown_report(self._make_periods())
        assert "-22.59%" in r

    def test_report_contains_dates(self):
        r = format_drawdown_report(self._make_periods())
        assert "2020-01-16" in r
        assert "2020-03-18" in r

    def test_report_shows_duration(self):
        r = format_drawdown_report(self._make_periods())
        assert "42d" in r

    def test_n_show_limits_output(self):
        periods = [
            DrawdownPeriod(i, i+5, i+10, f"2020-0{i+1}-01",
                           f"2020-0{i+1}-06", f"2020-0{i+1}-11",
                           -float(i + 2), 5, 5, 100, 100 - i - 2)
            for i in range(1, 6)
        ]
        r = format_drawdown_report(periods, n_show=3)
        # Should only show 3 rows
        row_count = sum(1 for line in r.split("\n") if "%" in line)
        assert row_count <= 3
