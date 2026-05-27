"""Unit tests for backtest/position_attribution.py."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest

from backtest.position_attribution import (
    compute_position_attribution,
    attribution_summary,
    format_attribution_report,
    PositionAttribution,
)


def _make_prices(n=500, drift=0.0003, vol=0.01, seed=42) -> pd.Series:
    np.random.seed(seed)
    rets = np.random.normal(drift, vol, n)
    idx = pd.bdate_range("2020-01-01", periods=n)
    return pd.Series(100.0 * np.cumprod(1 + rets), index=idx)


class TestComputePositionAttribution:

    def _make_data(self, n=500):
        tickers = {"AVUV": 0.25, "AVDV": 0.25, "QMOM": 0.15, "DBMF": 0.20, "CTA": 0.15}
        prices = {t: _make_prices(n, seed=i+1) for i, t in enumerate(tickers)}
        spy = _make_prices(n, seed=99)
        return prices, tickers, spy

    def test_returns_list_of_position_attributions(self):
        prices, weights, spy = self._make_data()
        result = compute_position_attribution(prices, weights, spy)
        assert isinstance(result, list)
        assert all(isinstance(p, PositionAttribution) for p in result)

    def test_one_attribution_per_position(self):
        prices, weights, spy = self._make_data()
        result = compute_position_attribution(prices, weights, spy)
        assert len(result) == len(weights)

    def test_sorted_by_contribution_descending(self):
        prices, weights, spy = self._make_data()
        result = compute_position_attribution(prices, weights, spy)
        contribs = [p.contribution for p in result]
        assert contribs == sorted(contribs, reverse=True)

    def test_hit_rate_between_0_and_100(self):
        prices, weights, spy = self._make_data()
        for p in compute_position_attribution(prices, weights, spy):
            assert 0 <= p.hit_rate <= 100

    def test_max_dd_non_positive(self):
        prices, weights, spy = self._make_data()
        for p in compute_position_attribution(prices, weights, spy):
            assert p.max_dd <= 0

    def test_corr_between_minus1_and_1(self):
        prices, weights, spy = self._make_data()
        for p in compute_position_attribution(prices, weights, spy):
            assert -1.0 <= p.corr_to_spy <= 1.0

    def test_contribution_equals_weight_times_return(self):
        prices = {"AVUV": _make_prices(300, seed=5)}
        weights = {"AVUV": 0.30}
        result = compute_position_attribution(prices, weights)
        assert len(result) == 1
        expected = weights["AVUV"] * result[0].total_return
        assert abs(result[0].contribution - expected) < 0.1

    def test_missing_price_data_skipped(self):
        prices = {"AVUV": _make_prices(300)}
        weights = {"AVUV": 0.25, "MISSING": 0.75}
        result = compute_position_attribution(prices, weights)
        tickers = {p.ticker for p in result}
        assert "MISSING" not in tickers

    def test_no_spy_gives_zero_correlation(self):
        prices = {"A": _make_prices(300, seed=1)}
        weights = {"A": 1.0}
        result = compute_position_attribution(prices, weights, spy_prices=None)
        assert result[0].corr_to_spy == 0.0

    def test_daily_vol_positive(self):
        prices, weights, _ = self._make_data()
        for p in compute_position_attribution(prices, weights):
            assert p.daily_vol > 0


class TestAttributionSummary:

    def _make_attrs(self):
        return [
            PositionAttribution("AVUV", 25.0, 15.0, 3.75, 12.0, 55.0, 1.2, -8.0, 0.7),
            PositionAttribution("AVDV", 25.0, 8.0,  2.00, 14.0, 52.0, 0.6, -12.0, 0.6),
            PositionAttribution("DBMF", 20.0, -3.0, -0.60, 8.0, 47.0, -0.4, -5.0, 0.1),
        ]

    def test_n_positions_correct(self):
        s = attribution_summary(self._make_attrs())
        assert s["n_positions"] == 3

    def test_best_worst_correct(self):
        s = attribution_summary(self._make_attrs())
        assert s["best_position"] == "AVUV"
        assert s["worst_position"] == "DBMF"

    def test_total_contribution(self):
        s = attribution_summary(self._make_attrs())
        expected = 3.75 + 2.0 - 0.60
        assert abs(s["total_contribution"] - expected) < 0.01

    def test_empty_returns_empty(self):
        assert attribution_summary([]) == {}


class TestFormatAttributionReport:

    def test_contains_header(self):
        prices = {"AVUV": _make_prices(300, seed=1), "QMOM": _make_prices(300, seed=2)}
        weights = {"AVUV": 0.5, "QMOM": 0.5}
        attrs = compute_position_attribution(prices, weights)
        r = format_attribution_report(attrs)
        assert "ATTRIBUTION" in r

    def test_contains_ticker_names(self):
        prices = {"AVUV": _make_prices(300, seed=1)}
        weights = {"AVUV": 1.0}
        attrs = compute_position_attribution(prices, weights)
        r = format_attribution_report(attrs)
        assert "AVUV" in r

    def test_empty_returns_message(self):
        r = format_attribution_report([])
        assert "unavailable" in r.lower()
