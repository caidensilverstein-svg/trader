"""Unit tests for core/expected_returns.py."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from core.expected_returns import (
    compute_expected_returns,
    portfolio_expected_return,
    format_expected_return_report,
    ETFExpectedReturn,
    FACTOR_PREMIA,
    ETF_FACTOR_LOADINGS,
)


class TestComputeExpectedReturns:

    def test_returns_list(self):
        result = compute_expected_returns(["AVUV", "AVDV", "QMOM"])
        assert isinstance(result, list)

    def test_returns_etf_expected_return_objects(self):
        result = compute_expected_returns(["AVUV", "AVDV"])
        for e in result:
            assert isinstance(e, ETFExpectedReturn)

    def test_correct_number_of_results(self):
        result = compute_expected_returns(["AVUV", "AVDV", "QMOM", "DBMF"])
        assert len(result) == 4

    def test_sorted_by_adjusted_return_descending(self):
        result = compute_expected_returns(["AVUV", "AVDV", "QMOM", "DBMF"])
        rets = [e.adjusted_return for e in result]
        assert rets == sorted(rets, reverse=True)

    def test_equity_etf_higher_return_than_managed_futures(self):
        result = compute_expected_returns(["AVUV", "DBMF"])
        avuv = next(e for e in result if e.ticker == "AVUV")
        dbmf = next(e for e in result if e.ticker == "DBMF")
        assert avuv.adjusted_return > dbmf.adjusted_return

    def test_factor_premium_adds_to_rf(self):
        result = compute_expected_returns(["AVUV"], rf=0.04)
        avuv = result[0]
        assert abs(avuv.expected_return - (0.04 + avuv.factor_premium)) < 0.0001

    def test_bull_regime_adj_positive_for_equity(self):
        result = compute_expected_returns(["AVUV"], regime="BULL")
        avuv = result[0]
        assert avuv.regime_adj > 0

    def test_bear_crisis_regime_adj_negative_for_equity(self):
        result = compute_expected_returns(["AVUV"], regime="BEAR_CRISIS")
        avuv = result[0]
        assert avuv.regime_adj < 0

    def test_dbmf_no_regime_adj(self):
        result = compute_expected_returns(["DBMF"], regime="BEAR_CRISIS")
        dbmf = result[0]
        assert dbmf.regime_adj == 0.0

    def test_factor_breakdown_keys_match_premia(self):
        result = compute_expected_returns(["AVUV"])
        avuv = result[0]
        for factor in avuv.factor_breakdown:
            assert factor in FACTOR_PREMIA

    def test_factor_breakdown_sums_to_factor_premium(self):
        result = compute_expected_returns(["AVUV"])
        avuv = result[0]
        total = sum(avuv.factor_breakdown.values())
        assert abs(total - avuv.factor_premium) < 0.001

    def test_confidence_band_positive(self):
        result = compute_expected_returns(["AVUV", "DBMF"])
        for e in result:
            assert e.confidence_band > 0

    def test_unknown_ticker_excluded(self):
        result = compute_expected_returns(["AVUV", "NOTEXIST"])
        tickers = [e.ticker for e in result]
        assert "NOTEXIST" not in tickers
        assert "AVUV" in tickers

    def test_empty_tickers_returns_empty(self):
        result = compute_expected_returns([])
        assert result == []

    def test_factor_premium_override(self):
        override = dict(FACTOR_PREMIA)
        override["equity_rp"] = 0.10  # double the usual premium
        result_norm  = compute_expected_returns(["AVUV"])
        result_boost = compute_expected_returns(["AVUV"], factor_premia_override=override)
        assert result_boost[0].factor_premium > result_norm[0].factor_premium

    def test_avuv_value_loading_positive(self):
        result = compute_expected_returns(["AVUV"])
        avuv = result[0]
        assert avuv.factor_breakdown.get("value", 0) > 0

    def test_qmom_momentum_loading_dominant(self):
        result = compute_expected_returns(["QMOM"])
        qmom = result[0]
        mom = qmom.factor_breakdown.get("momentum", 0)
        value = qmom.factor_breakdown.get("value", 0)
        assert mom > abs(value)  # momentum dominates


class TestPortfolioExpectedReturn:

    _WEIGHTS = {"AVUV": 0.30, "AVDV": 0.30, "QMOM": 0.25, "DBMF": 0.15}

    def test_returns_dict(self):
        result = portfolio_expected_return(self._WEIGHTS)
        assert isinstance(result, dict)

    def test_contains_portfolio_er(self):
        result = portfolio_expected_return(self._WEIGHTS)
        assert "portfolio_er" in result

    def test_contains_sharpe_estimate(self):
        result = portfolio_expected_return(self._WEIGHTS)
        assert "sharpe_estimate" in result

    def test_portfolio_er_between_components(self):
        result = portfolio_expected_return(self._WEIGHTS)
        # Portfolio ER should be reasonable (between ~5% and ~20%)
        assert 0.03 <= result["portfolio_er"] <= 0.25

    def test_regime_stored(self):
        result = portfolio_expected_return(self._WEIGHTS, regime="SIDEWAYS")
        assert result["regime"] == "SIDEWAYS"

    def test_bear_regime_lower_er_than_bull(self):
        bull  = portfolio_expected_return(self._WEIGHTS, regime="BULL")
        bear  = portfolio_expected_return(self._WEIGHTS, regime="BEAR")
        assert bull["portfolio_er"] > bear["portfolio_er"]

    def test_empty_weights_returns_empty(self):
        result = portfolio_expected_return({})
        assert result == {}


class TestFormatExpectedReturnReport:

    def test_contains_header(self):
        result = compute_expected_returns(["AVUV", "QMOM"])
        r = format_expected_return_report(result)
        assert "EXPECTED RETURN" in r

    def test_contains_ticker_names(self):
        result = compute_expected_returns(["AVUV", "QMOM"])
        r = format_expected_return_report(result)
        assert "AVUV" in r
        assert "QMOM" in r

    def test_contains_factor_breakdown(self):
        result = compute_expected_returns(["AVUV"])
        r = format_expected_return_report(result)
        assert "FACTOR BREAKDOWN" in r

    def test_contains_portfolio_stats_when_provided(self):
        er_list = compute_expected_returns(["AVUV"])
        port = portfolio_expected_return({"AVUV": 1.0})
        r = format_expected_return_report(er_list, port_stats=port)
        assert "PORTFOLIO" in r

    def test_empty_returns_message(self):
        r = format_expected_return_report([])
        assert "unavailable" in r.lower()
