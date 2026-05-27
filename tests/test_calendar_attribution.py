"""Unit tests for backtest/calendar_attribution.py."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest

from backtest.calendar_attribution import (
    compute_calendar_attribution,
    calendar_summary_stats,
    format_calendar_report,
    YearStats,
)


def _make_equity(start="2020-01-01", end="2022-12-31", drift=0.0003) -> pd.Series:
    np.random.seed(42)
    dates = pd.bdate_range(start, end)
    rets = np.random.normal(drift, 0.01, len(dates))
    vals = 100_000 * np.cumprod(1 + rets)
    return pd.Series(vals, index=dates)


class TestComputeCalendarAttribution:

    def test_returns_one_entry_per_year(self):
        eq = _make_equity("2020-01-01", "2022-12-31")
        years = compute_calendar_attribution(eq)
        assert len(years) == 3
        assert [y.year for y in years] == [2020, 2021, 2022]

    def test_annual_return_calculation(self):
        # Monotonically rising: 2021 should be positive
        dates = pd.bdate_range("2021-01-01", "2021-12-31")
        vals = pd.Series(100_000 * np.cumprod(1 + np.full(len(dates), 0.001)), index=dates)
        years = compute_calendar_attribution(vals)
        assert len(years) == 1
        assert years[0].annual_return > 0

    def test_negative_year_detected(self):
        dates = pd.bdate_range("2020-01-01", "2020-12-31")
        vals = pd.Series(100_000 * np.cumprod(1 + np.full(len(dates), -0.001)), index=dates)
        years = compute_calendar_attribution(vals)
        assert years[0].annual_return < 0

    def test_max_drawdown_negative(self):
        eq = _make_equity()
        for year in compute_calendar_attribution(eq):
            assert year.max_drawdown <= 0

    def test_sharpe_finite(self):
        eq = _make_equity()
        for year in compute_calendar_attribution(eq):
            assert np.isfinite(year.sharpe)

    def test_start_end_values_match_curve(self):
        eq = _make_equity("2021-01-01", "2021-12-31")
        years = compute_calendar_attribution(eq)
        assert len(years) == 1
        assert abs(years[0].start_value - float(eq.iloc[0])) < 1.0
        assert abs(years[0].end_value - float(eq.iloc[-1])) < 1.0

    def test_n_days_positive(self):
        eq = _make_equity()
        for y in compute_calendar_attribution(eq):
            assert y.n_days > 0

    def test_best_month_ge_worst_month(self):
        eq = _make_equity()
        for y in compute_calendar_attribution(eq):
            assert y.best_month >= y.worst_month

    def test_insufficient_data_returns_empty(self):
        eq = pd.Series([100, 101, 102], index=pd.bdate_range("2020-01-01", periods=3))
        years = compute_calendar_attribution(eq)
        assert years == []

    def test_sorted_chronologically(self):
        eq = _make_equity("2018-01-01", "2023-12-31")
        years = compute_calendar_attribution(eq)
        assert years == sorted(years, key=lambda y: y.year)

    def test_benchmark_return_computed(self):
        eq = _make_equity("2020-01-01", "2021-12-31")
        bench = _make_equity("2020-01-01", "2021-12-31", drift=0.0002)
        years = compute_calendar_attribution(eq, benchmark_curve=bench)
        for y in years:
            assert y.benchmark_return is not None

    def test_no_benchmark_returns_none(self):
        eq = _make_equity()
        years = compute_calendar_attribution(eq)
        for y in years:
            assert y.benchmark_return is None

    def test_multi_year_span(self):
        eq = _make_equity("2018-01-01", "2025-12-31")
        years = compute_calendar_attribution(eq)
        assert len(years) >= 7


class TestCalendarSummaryStats:

    def _make_year_stats(self):
        return [
            YearStats(2020, -5.0, -15.0, -0.3, 4.0, -8.0, 252, 100_000, 95_000),
            YearStats(2021, 18.0, -4.0, 1.5, 6.0, -2.0, 252, 95_000, 112_100),
            YearStats(2022, -8.0, -12.0, -0.8, 3.0, -9.0, 252, 112_100, 103_132),
            YearStats(2023, 12.0, -3.0, 1.1, 5.0, -1.5, 252, 103_132, 115_508),
        ]

    def test_n_years_correct(self):
        stats = calendar_summary_stats(self._make_year_stats())
        assert stats["n_years"] == 4

    def test_positive_negative_count(self):
        stats = calendar_summary_stats(self._make_year_stats())
        assert stats["positive_years"] == 2
        assert stats["negative_years"] == 2

    def test_pct_positive(self):
        stats = calendar_summary_stats(self._make_year_stats())
        assert stats["pct_positive"] == 50.0

    def test_best_worst_year_correct(self):
        stats = calendar_summary_stats(self._make_year_stats())
        assert stats["best_year"] == 2021
        assert stats["worst_year"] == 2022

    def test_avg_return_correct(self):
        stats = calendar_summary_stats(self._make_year_stats())
        assert abs(stats["avg_annual_ret"] - 4.25) < 0.1

    def test_empty_returns_empty(self):
        assert calendar_summary_stats([]) == {}


class TestFormatCalendarReport:

    def _make_years(self):
        return [
            YearStats(2021, 18.0, -4.0, 1.5, 6.0, -2.0, 252, 95_000, 112_100),
            YearStats(2022, -8.0, -12.0, -0.8, 3.0, -9.0, 252, 112_100, 103_132),
        ]

    def test_contains_header(self):
        r = format_calendar_report(self._make_years())
        assert "CALENDAR YEAR" in r

    def test_contains_years(self):
        r = format_calendar_report(self._make_years())
        assert "2021" in r
        assert "2022" in r

    def test_contains_return_values(self):
        r = format_calendar_report(self._make_years())
        assert "18.0" in r
        assert "-8.0" in r

    def test_contains_summary_stats(self):
        r = format_calendar_report(self._make_years())
        assert "Positive years" in r or "positive" in r.lower()

    def test_empty_returns_message(self):
        r = format_calendar_report([])
        assert "No calendar" in r or len(r) < 50
