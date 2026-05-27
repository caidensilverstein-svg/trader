"""Unit tests for core/factor_momentum_signal.py."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest

from core.factor_momentum_signal import (
    compute_momentum_signals,
    compute_momentum_ic,
    format_momentum_signal_report,
    MomentumSignal,
    ICAnalysis,
    BOOST_THRESHOLD,
    PENALTY_THRESHOLD,
)


def _make_prices(n=600, seed=3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n)
    a = 100 * np.cumprod(1 + rng.normal(0.0004, 0.01, n))
    b = 100 * np.cumprod(1 + rng.normal(0.0002, 0.012, n))
    c = 100 * np.cumprod(1 + rng.normal(0.0001, 0.008, n))
    return pd.DataFrame({"AVUV": a, "AVDV": b, "QMOM": c}, index=dates)


class TestComputeMomentumSignals:

    def test_returns_list(self):
        prices = _make_prices()
        result = compute_momentum_signals(prices)
        assert isinstance(result, list)

    def test_returns_momentum_signal_objects(self):
        prices = _make_prices()
        result = compute_momentum_signals(prices)
        for s in result:
            assert isinstance(s, MomentumSignal)

    def test_has_results_for_each_ticker(self):
        prices = _make_prices()
        result = compute_momentum_signals(prices)
        tickers_seen = {s.ticker for s in result}
        for t in prices.columns:
            assert t in tickers_seen

    def test_signal_values_valid(self):
        prices = _make_prices()
        result = compute_momentum_signals(prices)
        valid_signals = {"BOOST", "NEUTRAL", "PENALTY"}
        for s in result:
            assert s.signal in valid_signals

    def test_weight_adj_matches_signal(self):
        prices = _make_prices()
        result = compute_momentum_signals(prices)
        for s in result:
            if s.signal == "BOOST":
                assert s.weight_adj == pytest.approx(0.10)
            elif s.signal == "PENALTY":
                assert s.weight_adj == pytest.approx(-0.20)
            else:
                assert s.weight_adj == pytest.approx(0.0)

    def test_boost_above_threshold(self):
        prices = _make_prices()
        result = compute_momentum_signals(prices)
        for s in result:
            if s.signal == "BOOST":
                assert s.composite_score >= BOOST_THRESHOLD

    def test_penalty_below_threshold(self):
        prices = _make_prices()
        result = compute_momentum_signals(prices)
        for s in result:
            if s.signal == "PENALTY":
                assert s.composite_score <= PENALTY_THRESHOLD

    def test_dates_are_strings(self):
        prices = _make_prices()
        result = compute_momentum_signals(prices)
        for s in result:
            assert isinstance(s.date, str)

    def test_score_6m_is_float(self):
        prices = _make_prices()
        result = compute_momentum_signals(prices)
        for s in result:
            assert isinstance(s.score_6m, float)

    def test_composite_formula(self):
        prices = _make_prices()
        result = compute_momentum_signals(prices)
        for s in result:
            expected = round(0.7 * s.score_6m + 0.3 * s.score_12m, 4)
            assert abs(s.composite_score - expected) < 0.001

    def test_empty_prices_returns_empty(self):
        result = compute_momentum_signals(pd.DataFrame())
        assert result == []


class TestComputeMomentumIC:

    def test_returns_ic_analysis(self):
        prices = _make_prices()
        result = compute_momentum_ic(prices)
        assert isinstance(result, ICAnalysis)

    def test_n_observations_positive(self):
        prices = _make_prices()
        result = compute_momentum_ic(prices)
        assert result.n_observations >= 0

    def test_ic_mean_is_float(self):
        prices = _make_prices()
        result = compute_momentum_ic(prices)
        assert isinstance(result.ic_mean, float)

    def test_pct_positive_in_range(self):
        prices = _make_prices()
        result = compute_momentum_ic(prices)
        if result.n_observations > 0:
            assert 0 <= result.pct_positive_ic <= 100


class TestFormatMomentumSignalReport:

    def test_contains_header(self):
        prices = _make_prices()
        signals = compute_momentum_signals(prices)
        r = format_momentum_signal_report(signals)
        assert "MOMENTUM" in r

    def test_contains_ticker_names(self):
        prices = _make_prices()
        signals = compute_momentum_signals(prices)
        r = format_momentum_signal_report(signals)
        for t in prices.columns:
            assert t in r

    def test_contains_signal_labels(self):
        prices = _make_prices()
        signals = compute_momentum_signals(prices)
        r = format_momentum_signal_report(signals)
        assert "BOOST" in r or "NEUTRAL" in r or "PENALTY" in r

    def test_contains_ic_section_when_provided(self):
        prices = _make_prices()
        signals = compute_momentum_signals(prices)
        ic = compute_momentum_ic(prices)
        r = format_momentum_signal_report(signals, ic=ic)
        if ic.n_observations > 0:
            assert "INFORMATION COEFFICIENT" in r

    def test_empty_signals_returns_message(self):
        r = format_momentum_signal_report([])
        assert "unavailable" in r.lower()
