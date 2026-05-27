"""Unit tests for backtest/sharpe_decomposition.py."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest

from backtest.sharpe_decomposition import (
    compute_sharpe_decomposition,
    format_sharpe_decomposition,
    SharpeDecompositionResult,
    AssetSharpeContrib,
)


def _make_prices(n=500, seed=10) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n)
    m = rng.normal(0.0004, 0.01, n)
    a = 100 * np.cumprod(1 + m + rng.normal(0, 0.005, n))
    b = 100 * np.cumprod(1 + m * 0.7 + rng.normal(0, 0.008, n))
    c = 100 * np.cumprod(1 + rng.normal(0.0001, 0.012, n))
    d = 100 * np.cumprod(1 + rng.normal(0.0001, 0.003, n))
    return pd.DataFrame({"AVUV": a, "AVDV": b, "QMOM": c, "DBMF": d}, index=dates)


_WEIGHTS = {"AVUV": 0.30, "AVDV": 0.30, "QMOM": 0.25, "DBMF": 0.15}


class TestComputeSharpeDecomposition:

    def test_returns_dataclass(self):
        prices = _make_prices()
        result = compute_sharpe_decomposition(prices, _WEIGHTS)
        assert isinstance(result, SharpeDecompositionResult)

    def test_asset_contributions_correct_length(self):
        prices = _make_prices()
        result = compute_sharpe_decomposition(prices, _WEIGHTS)
        assert len(result.asset_contributions) == len(_WEIGHTS)

    def test_contributions_are_dataclass(self):
        prices = _make_prices()
        result = compute_sharpe_decomposition(prices, _WEIGHTS)
        for a in result.asset_contributions:
            assert isinstance(a, AssetSharpeContrib)

    def test_portfolio_sharpe_reasonable(self):
        prices = _make_prices()
        result = compute_sharpe_decomposition(prices, _WEIGHTS)
        assert -5 <= result.portfolio_sharpe <= 5

    def test_sharpe_efficiency_is_float(self):
        prices = _make_prices()
        result = compute_sharpe_decomposition(prices, _WEIGHTS)
        assert isinstance(result.sharpe_efficiency, float)

    def test_sorted_by_portfolio_contrib_descending(self):
        prices = _make_prices()
        result = compute_sharpe_decomposition(prices, _WEIGHTS)
        contribs = [a.portfolio_contrib for a in result.asset_contributions]
        assert contribs == sorted(contribs, reverse=True)

    def test_dominant_contributor_is_valid_ticker(self):
        prices = _make_prices()
        result = compute_sharpe_decomposition(prices, _WEIGHTS)
        assert result.dominant_contributor in _WEIGHTS

    def test_raises_on_single_asset(self):
        dates = pd.bdate_range("2020-01-01", periods=200)
        prices = pd.DataFrame({"A": np.ones(200) * 100}, index=dates)
        with pytest.raises(ValueError):
            compute_sharpe_decomposition(prices, {"A": 1.0})

    def test_raises_on_insufficient_data(self):
        dates = pd.bdate_range("2020-01-01", periods=10)
        prices = pd.DataFrame({
            "A": np.ones(10) * 100,
            "B": np.ones(10) * 100,
        }, index=dates)
        with pytest.raises(ValueError):
            compute_sharpe_decomposition(prices, {"A": 0.5, "B": 0.5})

    def test_window_parameter_works(self):
        prices = _make_prices()
        r1 = compute_sharpe_decomposition(prices, _WEIGHTS, window=252)
        r2 = compute_sharpe_decomposition(prices, _WEIGHTS, window=63)
        assert r1 is not r2  # different objects for different windows

    def test_standalone_sharpe_is_float(self):
        prices = _make_prices()
        result = compute_sharpe_decomposition(prices, _WEIGHTS)
        for a in result.asset_contributions:
            assert isinstance(a.standalone_sharpe, float)

    def test_max_achievable_sharpe_positive(self):
        prices = _make_prices()
        result = compute_sharpe_decomposition(prices, _WEIGHTS)
        assert result.max_achievable_sharpe >= 0


class TestFormatSharpeDecomposition:

    def test_contains_header(self):
        prices = _make_prices()
        result = compute_sharpe_decomposition(prices, _WEIGHTS)
        r = format_sharpe_decomposition(result)
        assert "SHARPE" in r

    def test_contains_ticker_names(self):
        prices = _make_prices()
        result = compute_sharpe_decomposition(prices, _WEIGHTS)
        r = format_sharpe_decomposition(result)
        for t in _WEIGHTS:
            assert t in r

    def test_contains_portfolio_sharpe(self):
        prices = _make_prices()
        result = compute_sharpe_decomposition(prices, _WEIGHTS)
        r = format_sharpe_decomposition(result)
        assert "Portfolio Sharpe" in r

    def test_contains_efficiency_score(self):
        prices = _make_prices()
        result = compute_sharpe_decomposition(prices, _WEIGHTS)
        r = format_sharpe_decomposition(result)
        assert "Efficiency" in r
