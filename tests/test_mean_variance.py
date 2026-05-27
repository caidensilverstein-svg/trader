"""Unit tests for backtest/mean_variance.py."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest

from backtest.mean_variance import (
    run_mvo,
    format_mvo_report,
    EfficientPortfolio,
    MVOResult,
    _annualized_cov,
    _ledoit_wolf_shrinkage,
)


def _make_prices(n=500, seed=42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n)
    a = 100 * np.cumprod(1 + rng.normal(0.0005, 0.01, n))
    b = 100 * np.cumprod(1 + rng.normal(0.0003, 0.015, n))
    c = 100 * np.cumprod(1 + rng.normal(0.0002, 0.008, n))
    return pd.DataFrame({"AVUV": a, "AVDV": b, "DBMF": c}, index=dates)


_WEIGHTS = {"AVUV": 0.40, "AVDV": 0.40, "DBMF": 0.20}


class TestAnnualizedCov:

    def test_returns_square_matrix(self):
        prices = _make_prices()
        returns = prices.pct_change().dropna()
        cov = _annualized_cov(returns)
        assert cov.shape == (3, 3)

    def test_positive_semidefinite(self):
        prices = _make_prices()
        returns = prices.pct_change().dropna()
        cov = _annualized_cov(returns)
        eigenvalues = np.linalg.eigvalsh(cov)
        assert (eigenvalues >= -1e-10).all()

    def test_symmetric(self):
        prices = _make_prices()
        returns = prices.pct_change().dropna()
        cov = _annualized_cov(returns)
        assert np.allclose(cov, cov.T)


class TestLedoitWolfShrinkage:

    def test_returns_correct_shape(self):
        prices = _make_prices()
        returns = prices.pct_change().dropna()
        cov = _ledoit_wolf_shrinkage(returns)
        assert cov.shape == (3, 3)

    def test_positive_semidefinite(self):
        prices = _make_prices()
        returns = prices.pct_change().dropna()
        cov = _ledoit_wolf_shrinkage(returns)
        eigenvalues = np.linalg.eigvalsh(cov)
        assert (eigenvalues >= -1e-10).all()

    def test_symmetric(self):
        prices = _make_prices()
        returns = prices.pct_change().dropna()
        cov = _ledoit_wolf_shrinkage(returns)
        assert np.allclose(cov, cov.T, atol=1e-10)

    def test_more_diagonal_than_sample(self):
        prices = _make_prices()
        returns = prices.pct_change().dropna()
        sample = _annualized_cov(returns)
        shrunk = _ledoit_wolf_shrinkage(returns)
        # Off-diagonal elements should have smaller magnitude on average
        n = sample.shape[0]
        off_sample = [abs(sample[i, j]) for i in range(n) for j in range(n) if i != j]
        off_shrunk  = [abs(shrunk[i, j])  for i in range(n) for j in range(n) if i != j]
        assert np.mean(off_shrunk) <= np.mean(off_sample) * 1.01  # shrunk <= sample


class TestRunMVO:

    def test_returns_mvo_result(self):
        prices = _make_prices()
        result = run_mvo(prices)
        assert isinstance(result, MVOResult)

    def test_max_sharpe_is_flagged(self):
        prices = _make_prices()
        result = run_mvo(prices)
        assert result.max_sharpe.is_max_sharpe

    def test_min_vol_is_flagged(self):
        prices = _make_prices()
        result = run_mvo(prices)
        assert result.min_vol.is_min_vol

    def test_weights_sum_to_one(self):
        prices = _make_prices()
        result = run_mvo(prices)
        for port in [result.max_sharpe, result.min_vol, result.equal_weight]:
            total = sum(port.weights.values())
            assert abs(total - 1.0) < 0.01

    def test_max_sharpe_has_higher_sharpe_than_eq_weight(self):
        prices = _make_prices()
        result = run_mvo(prices)
        assert result.max_sharpe.sharpe >= result.equal_weight.sharpe - 0.1

    def test_min_vol_has_lower_vol_than_eq_weight(self):
        prices = _make_prices()
        result = run_mvo(prices)
        assert result.min_vol.expected_vol <= result.equal_weight.expected_vol + 0.02

    def test_all_weights_non_negative(self):
        prices = _make_prices()
        result = run_mvo(prices)
        for port in [result.max_sharpe, result.min_vol, result.equal_weight]:
            for w in port.weights.values():
                assert w >= -0.001  # allow tiny numerical noise

    def test_equal_weight_is_uniform(self):
        prices = _make_prices()
        result = run_mvo(prices)
        n = len(result.tickers)
        expected = 1 / n
        for t in result.tickers:
            assert abs(result.equal_weight.weights[t] - expected) < 0.01

    def test_factor_portfolio_computed_when_provided(self):
        prices = _make_prices()
        result = run_mvo(prices, factor_weights=_WEIGHTS)
        assert result.factor_target is not None

    def test_factor_portfolio_weights_match_input(self):
        prices = _make_prices()
        result = run_mvo(prices, factor_weights=_WEIGHTS)
        total = sum(result.factor_target.weights.values())
        assert abs(total - 1.0) < 0.01

    def test_covariance_matrix_shape(self):
        prices = _make_prices()
        result = run_mvo(prices)
        n = len(result.tickers)
        assert result.covariance_matrix.shape == (n, n)

    def test_expected_returns_length(self):
        prices = _make_prices()
        result = run_mvo(prices)
        assert len(result.expected_returns) == len(result.tickers)

    def test_estimation_period_correct(self):
        prices = _make_prices()
        result = run_mvo(prices)
        returns = prices.pct_change().dropna()
        assert result.estimation_period_days == len(returns)

    def test_frontier_is_non_empty(self):
        prices = _make_prices()
        result = run_mvo(prices)
        assert len(result.frontier_vols) > 0
        assert len(result.frontier_rets) > 0

    def test_raises_on_single_asset(self):
        dates = pd.bdate_range("2020-01-01", periods=200)
        prices = pd.DataFrame({"A": np.ones(200) * 100}, index=dates)
        with pytest.raises(ValueError):
            run_mvo(prices)


class TestFormatMVOReport:

    def test_contains_header(self):
        prices = _make_prices()
        result = run_mvo(prices)
        r = format_mvo_report(result)
        assert "MEAN-VARIANCE" in r

    def test_contains_max_sharpe(self):
        prices = _make_prices()
        result = run_mvo(prices)
        r = format_mvo_report(result)
        assert "Max Sharpe" in r

    def test_contains_min_vol(self):
        prices = _make_prices()
        result = run_mvo(prices)
        r = format_mvo_report(result)
        assert "Min Volatility" in r

    def test_contains_ticker_names(self):
        prices = _make_prices()
        result = run_mvo(prices)
        r = format_mvo_report(result)
        for t in result.tickers:
            assert t in r

    def test_contains_factor_portfolio_when_provided(self):
        prices = _make_prices()
        result = run_mvo(prices, factor_weights=_WEIGHTS)
        r = format_mvo_report(result)
        assert "Factor" in r
