"""Unit tests for core/factor_exposure.py."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest

from core.factor_exposure import (
    compute_factor_exposures,
    build_factor_returns,
    format_factor_report,
    FactorExposure,
)


def _make_returns(n=300, seed=42) -> pd.Series:
    np.random.seed(seed)
    rets = np.random.normal(0.0003, 0.01, n)
    idx = pd.bdate_range("2022-01-01", periods=n)
    return pd.Series(rets, index=idx)


def _make_factor_data(portfolio: pd.Series, n_factors=3, seed=7) -> dict:
    np.random.seed(seed)
    factor_data = {}
    for i in range(n_factors):
        noise = np.random.normal(0, 0.005, len(portfolio))
        factor_data[f"factor_{i}"] = pd.Series(
            portfolio.values * (0.5 + i * 0.2) + noise,
            index=portfolio.index,
        )
    return factor_data


class TestComputeFactorExposures:

    def test_returns_list_of_factor_exposures(self):
        port = _make_returns()
        factors = _make_factor_data(port)
        result = compute_factor_exposures(port, factors)
        assert isinstance(result, list)
        assert all(isinstance(e, FactorExposure) for e in result)

    def test_returns_one_per_factor(self):
        port = _make_returns()
        factors = _make_factor_data(port, n_factors=3)
        result = compute_factor_exposures(port, factors)
        assert len(result) == 3

    def test_high_correlation_gives_significant_beta(self):
        port = _make_returns(seed=1)
        # Factor perfectly correlated with portfolio
        factor_data = {"market": port * 0.8 + pd.Series(
            np.random.normal(0, 0.001, len(port)), index=port.index)}
        result = compute_factor_exposures(port, factor_data)
        assert len(result) == 1
        assert result[0].significant

    def test_sorted_by_abs_t_stat(self):
        port = _make_returns()
        factors = _make_factor_data(port, n_factors=4)
        result = compute_factor_exposures(port, factors)
        t_stats = [abs(e.t_stat) for e in result]
        assert t_stats == sorted(t_stats, reverse=True)

    def test_insufficient_data_returns_empty(self):
        port = _make_returns(n=20)
        factors = {"market": port}
        result = compute_factor_exposures(port, factors, min_overlap=63)
        assert result == []

    def test_uncorrelated_factor_not_significant(self):
        np.random.seed(42)
        port = _make_returns(seed=42)
        # Completely independent factor
        noise = pd.Series(np.random.normal(0, 0.01, len(port)), index=port.index)
        result = compute_factor_exposures(port, {"noise_factor": noise})
        if result:
            assert not result[0].significant

    def test_beta_roughly_correct(self):
        # Build a portfolio that is exactly 0.8x SPY-like factor
        np.random.seed(100)
        n = 500
        idx = pd.bdate_range("2020-01-01", periods=n)
        factor_ret = pd.Series(np.random.normal(0.0003, 0.01, n), index=idx)
        port_ret   = 0.8 * factor_ret + pd.Series(np.random.normal(0, 0.001, n), index=idx)
        result = compute_factor_exposures(port_ret, {"market": factor_ret})
        assert len(result) == 1
        assert abs(result[0].beta - 0.8) < 0.1  # within 10% of true beta

    def test_empty_factor_data_returns_empty(self):
        port = _make_returns()
        result = compute_factor_exposures(port, {})
        assert result == []

    def test_factor_names_preserved(self):
        port = _make_returns()
        factors = {"size_smb": port * 0.3, "value_hml": port * 0.1}
        result = compute_factor_exposures(port, factors)
        names = {e.factor for e in result}
        assert "size_smb" in names
        assert "value_hml" in names


class TestBuildFactorReturns:

    def _make_prices(self, n=200, start=100.0, seed=42) -> pd.Series:
        np.random.seed(seed)
        rets = np.random.normal(0.0003, 0.01, n)
        idx = pd.bdate_range("2022-01-01", periods=n)
        return pd.Series(start * np.cumprod(1 + rets), index=idx)

    def test_market_beta_present_when_market_price_given(self):
        prices = {"market": self._make_prices(seed=1)}
        result = build_factor_returns(prices)
        assert "market_beta" in result

    def test_smb_computed_when_size_and_market_given(self):
        prices = {
            "market": self._make_prices(seed=1),
            "size":   self._make_prices(seed=2),
        }
        result = build_factor_returns(prices)
        assert "size_smb" in result

    def test_hml_computed_when_value_and_growth_given(self):
        prices = {
            "value":  self._make_prices(seed=3),
            "growth": self._make_prices(seed=4),
        }
        result = build_factor_returns(prices)
        assert "value_hml" in result

    def test_empty_prices_returns_empty(self):
        result = build_factor_returns({})
        assert result == {}

    def test_returns_are_daily_not_prices(self):
        prices = {"market": self._make_prices()}
        result = build_factor_returns(prices)
        rets = result.get("market_beta", pd.Series())
        assert len(rets) > 0
        # Daily returns should be small (< 10%)
        assert all(abs(r) < 0.15 for r in rets)


class TestFormatFactorReport:

    def _make_exposures(self):
        return [
            FactorExposure("market_beta", 0.85, 12.3, True, "SPY"),
            FactorExposure("size_smb",    0.20, 2.1, True, "IWM"),
            FactorExposure("value_hml",   0.15, 1.5, False, "IVE"),
        ]

    def test_contains_header(self):
        r = format_factor_report(self._make_exposures())
        assert "FACTOR EXPOSURE" in r

    def test_contains_factor_names(self):
        r = format_factor_report(self._make_exposures())
        assert "market_beta" in r
        assert "size_smb" in r

    def test_contains_beta_values(self):
        r = format_factor_report(self._make_exposures())
        assert "0.8500" in r

    def test_empty_returns_message(self):
        r = format_factor_report([])
        assert "unavailable" in r.lower() or len(r) < 60

    def test_significance_markers_present(self):
        r = format_factor_report(self._make_exposures())
        assert "***" in r or "** " in r
