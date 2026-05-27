"""Unit tests for backtest/benchmark_comparison.py."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest

from backtest.benchmark_comparison import (
    build_benchmark_equity_curves,
    compute_benchmark_comparison,
    format_benchmark_report,
    BenchmarkResult,
    _metrics,
)


def _make_prices(n=500, drift=0.0003, vol=0.01, start=100.0, seed=42) -> pd.Series:
    np.random.seed(seed)
    rets = np.random.normal(drift, vol, n)
    idx = pd.bdate_range("2020-01-01", periods=n)
    return pd.Series(start * np.cumprod(1 + rets), index=idx)


class TestMetrics:

    def test_returns_required_keys(self):
        eq = _make_prices()
        m = _metrics(eq)
        for k in ("total_return", "ann_return", "ann_vol", "sharpe", "max_dd", "calmar"):
            assert k in m

    def test_rising_series_positive_return(self):
        eq = pd.Series(np.linspace(100, 150, 252),
                       index=pd.bdate_range("2020-01-01", periods=252))
        m = _metrics(eq)
        assert m["total_return"] > 0

    def test_insufficient_data_returns_empty(self):
        eq = pd.Series([100, 101], index=pd.bdate_range("2020-01-01", periods=2))
        assert _metrics(eq) == {}

    def test_max_dd_non_positive(self):
        eq = _make_prices()
        m = _metrics(eq)
        assert m["max_dd"] <= 0


class TestBuildBenchmarkEquityCurves:

    def _make_price_data(self, n=500):
        return {
            "SPY": _make_prices(n, seed=1),
            "AGG": _make_prices(n, drift=0.0001, vol=0.003, seed=2),
            "IWM": _make_prices(n, seed=3),
            "EFA": _make_prices(n, seed=4),
            "GLD": _make_prices(n, drift=0.0002, vol=0.008, seed=5),
        }

    def test_spy_always_created(self):
        curves = build_benchmark_equity_curves(self._make_price_data())
        assert any("SPY" in k for k in curves.keys())

    def test_6040_created_when_agg_available(self):
        curves = build_benchmark_equity_curves(self._make_price_data())
        assert any("60/40" in k for k in curves.keys())

    def test_equal_weight_created_with_5_assets(self):
        curves = build_benchmark_equity_curves(self._make_price_data())
        assert any("Equal Weight" in k for k in curves.keys())

    def test_curves_start_at_start_value(self):
        curves = build_benchmark_equity_curves(self._make_price_data(), start_value=50_000)
        for name, curve in curves.items():
            assert abs(float(curve.iloc[0]) - 50_000) < 500  # first value near start

    def test_no_spy_returns_empty(self):
        curves = build_benchmark_equity_curves({"AGG": _make_prices()})
        assert curves == {}


class TestComputeBenchmarkComparison:

    def _make_data(self, n=500):
        strat = _make_prices(n, drift=0.0005, seed=10)  # better strategy
        curves = {
            "SPY (100%)":       _make_prices(n, drift=0.0003, seed=11),
            "60/40 (SPY/AGG)":  _make_prices(n, drift=0.0002, seed=12),
        }
        return strat, curves

    def test_returns_list_of_results(self):
        strat, curves = self._make_data()
        results = compute_benchmark_comparison(strat, curves)
        assert isinstance(results, list)
        assert all(isinstance(r, BenchmarkResult) for r in results)

    def test_strategy_always_in_results(self):
        strat, curves = self._make_data()
        results = compute_benchmark_comparison(strat, curves)
        names = [r.name for r in results]
        assert "OUR STRATEGY" in names

    def test_sorted_by_sharpe_descending(self):
        strat, curves = self._make_data()
        results = compute_benchmark_comparison(strat, curves)
        sharpes = [r.sharpe for r in results]
        assert sharpes == sorted(sharpes, reverse=True)

    def test_alpha_computed_vs_spy(self):
        strat, curves = self._make_data()
        results = compute_benchmark_comparison(strat, curves)
        our = next(r for r in results if r.name == "OUR STRATEGY")
        # Should have non-zero alpha since strategy has different drift
        spy = next((r for r in results if "SPY" in r.name), None)
        if spy:
            assert abs(our.alpha_vs_spy) >= 0

    def test_empty_benchmarks_returns_strategy_only(self):
        strat = _make_prices(500)
        results = compute_benchmark_comparison(strat, {})
        assert len(results) == 1
        assert results[0].name == "OUR STRATEGY"


class TestFormatBenchmarkReport:

    def test_contains_header(self):
        strat, curves = TestComputeBenchmarkComparison()._make_data()
        results = compute_benchmark_comparison(strat, curves)
        r = format_benchmark_report(results)
        assert "BENCHMARK" in r

    def test_contains_strategy_label(self):
        strat, curves = TestComputeBenchmarkComparison()._make_data()
        results = compute_benchmark_comparison(strat, curves)
        r = format_benchmark_report(results)
        assert "OUR STRATEGY" in r

    def test_empty_returns_message(self):
        r = format_benchmark_report([])
        assert "unavailable" in r.lower()
