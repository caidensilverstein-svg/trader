"""Unit tests for backtest/monte_carlo.py."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest

from backtest.monte_carlo import (
    run_monte_carlo,
    format_mc_report,
    _block_resample_returns,
)


def _make_equity(n: int = 500, daily_ret: float = 0.0003) -> pd.Series:
    np.random.seed(42)
    rets = np.random.normal(daily_ret, 0.01, n)
    return pd.Series(100_000 * np.cumprod(1 + rets))


class TestBlockResampleReturns:

    def test_correct_length(self):
        returns = np.random.normal(0, 0.01, 100)
        rng = np.random.default_rng(42)
        result = _block_resample_returns(returns, 252, block_size=5, rng=rng)
        assert len(result) == 252

    def test_values_come_from_original(self):
        returns = np.arange(20, dtype=float) / 100
        rng = np.random.default_rng(42)
        result = _block_resample_returns(returns, 100, block_size=5, rng=rng)
        for v in result:
            assert round(v, 10) in [round(r, 10) for r in returns]

    def test_different_seeds_give_different_results(self):
        returns = np.random.normal(0, 0.01, 100)
        s1 = _block_resample_returns(returns, 252, 5, np.random.default_rng(1))
        s2 = _block_resample_returns(returns, 252, 5, np.random.default_rng(2))
        assert not np.allclose(s1, s2)


class TestRunMonteCarlo:

    def test_returns_dict_with_horizon_keys(self):
        eq = _make_equity()
        mc = run_monte_carlo(eq, n_simulations=50, horizons=(252,))
        assert "1yr" in mc

    def test_contains_required_percentiles(self):
        eq = _make_equity()
        mc = run_monte_carlo(eq, n_simulations=50, horizons=(252,))
        for k in ("p05", "p25", "p50", "p75", "p95", "mean", "prob_loss"):
            assert k in mc["1yr"], f"Missing {k}"

    def test_percentiles_ordered(self):
        eq = _make_equity()
        mc = run_monte_carlo(eq, n_simulations=200, horizons=(252,))
        d  = mc["1yr"]
        assert d["p05"] <= d["p25"] <= d["p50"] <= d["p75"] <= d["p95"]

    def test_prob_loss_between_0_and_100(self):
        eq = _make_equity()
        mc = run_monte_carlo(eq, n_simulations=100, horizons=(252,))
        assert 0 <= mc["1yr"]["prob_loss"] <= 100

    def test_longer_horizons_higher_median(self):
        eq = _make_equity(1000, daily_ret=0.001)  # strong positive drift
        mc = run_monte_carlo(eq, n_simulations=200, horizons=(252, 756))
        assert mc["3yr"]["p50"] > mc["1yr"]["p50"]

    def test_insufficient_data_returns_empty(self):
        eq = _make_equity(20)  # too short
        mc = run_monte_carlo(eq, n_simulations=50, horizons=(252,))
        assert mc == {}

    def test_reproducible_with_same_seed(self):
        eq = _make_equity()
        mc1 = run_monte_carlo(eq, n_simulations=100, horizons=(252,), seed=99)
        mc2 = run_monte_carlo(eq, n_simulations=100, horizons=(252,), seed=99)
        assert mc1 == mc2

    def test_multiple_horizons(self):
        eq = _make_equity()
        mc = run_monte_carlo(eq, n_simulations=50, horizons=(252, 756, 1260))
        assert "1yr" in mc
        assert "3yr" in mc
        assert "5yr" in mc


class TestFormatMCReport:

    def _make_mc(self) -> dict:
        return {
            "1yr": {"p05": 91_000, "p25": 100_000, "p50": 107_000,
                    "p75": 113_000, "p95": 123_000, "mean": 107_500,
                    "prob_loss": 22.6, "prob_2x": 0.1, "n_sims": 1000},
            "5yr": {"p05": 97_000, "p25": 120_000, "p50": 137_000,
                    "p75": 157_000, "p95": 189_000, "mean": 140_000,
                    "prob_loss": 7.3, "prob_2x": 2.1, "n_sims": 1000},
        }

    def test_report_contains_header(self):
        r = format_mc_report(self._make_mc())
        assert "MONTE CARLO" in r

    def test_report_contains_horizons(self):
        r = format_mc_report(self._make_mc())
        assert "1yr" in r
        assert "5yr" in r

    def test_report_contains_percentile_values(self):
        r = format_mc_report(self._make_mc())
        assert "107,000" in r  # median for 1yr

    def test_report_shows_prob_loss(self):
        r = format_mc_report(self._make_mc())
        assert "22.6%" in r

    def test_report_contains_disclaimer(self):
        r = format_mc_report(self._make_mc())
        assert "past performance" in r.lower()
