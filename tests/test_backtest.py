"""Unit tests for backtest/engine.py."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest

from backtest.engine import (
    _bsc_scalar,
    _effective_weights,
    _compute_summary,
    format_backtest_report,
)
import config


class TestBSCScalarBacktest:

    def test_high_vol_gives_min(self):
        np.random.seed(1)
        rets = pd.Series(np.random.normal(0, 0.025, 200))  # 25% daily vol
        assert _bsc_scalar(rets) == config.BSC_MIN_SCALAR

    def test_low_vol_gives_max(self):
        np.random.seed(1)
        rets = pd.Series(np.random.normal(0, 0.002, 200))  # 2% daily vol
        assert _bsc_scalar(rets) == config.BSC_MAX_SCALAR

    def test_insufficient_data_gives_min(self):
        rets = pd.Series([0.01, -0.01, 0.02])
        assert _bsc_scalar(rets) == config.BSC_MIN_SCALAR

    def test_returns_float(self):
        rets = pd.Series(np.random.normal(0, 0.01, 200))
        result = _bsc_scalar(rets)
        assert isinstance(result, float)


class TestEffectiveWeightsBacktest:

    def test_bull_full_weights(self):
        w = _effective_weights(1.0, "BULL")
        # BULL mult = 1.0, base total = 0.75
        assert abs(sum(w.values()) - 0.75) < 0.001

    def test_bear_crisis_halved(self):
        bull = _effective_weights(1.0, "BULL")
        crisis = _effective_weights(1.0, "BEAR_CRISIS")
        mult = config.REGIME_ETF_MULT["BEAR_CRISIS"] / config.REGIME_ETF_MULT["BULL"]
        for t in bull:
            assert abs(crisis[t] / bull[t] - mult) < 0.001

    def test_bsc_scales_qmom_only(self):
        w1 = _effective_weights(1.0, "BULL")
        w2 = _effective_weights(0.5, "BULL")
        assert w2["QMOM"] < w1["QMOM"]
        # Non-QMOM tickers should be unchanged by B-SC
        for t in ["AVUV", "AVDV", "DBMF", "CTA"]:
            if t in w1:
                assert abs(w1[t] - w2[t]) < 0.001


class TestComputeSummary:

    def _make_series(self, n: int, ann_return: float) -> pd.Series:
        daily = (1 + ann_return) ** (1 / 252) - 1
        vals = [100.0]
        for _ in range(n - 1):
            vals.append(vals[-1] * (1 + daily))
        idx = pd.date_range("2020-01-01", periods=n, freq="B")
        return pd.Series(vals, index=idx)

    def test_positive_ann_return(self):
        eq = self._make_series(252, 0.10)
        spy = self._make_series(252, 0.10)
        s = _compute_summary(eq, spy, [])
        assert s["strategy"]["ann_return"] > 0

    def test_alpha_positive_when_outperforming(self):
        eq  = self._make_series(252, 0.15)
        spy = self._make_series(252, 0.10)
        s = _compute_summary(eq, spy, [])
        assert s["alpha_ann"] > 0

    def test_alpha_negative_when_underperforming(self):
        eq  = self._make_series(252, 0.05)
        spy = self._make_series(252, 0.12)
        s = _compute_summary(eq, spy, [])
        assert s["alpha_ann"] < 0

    def test_max_dd_negative(self):
        idx = pd.date_range("2020-01-01", periods=100, freq="B")
        vals = [100, 110, 120, 100, 90, 95, 100] + [100] * 93
        eq = pd.Series(vals[:100], index=idx)
        spy = pd.Series(vals[:100], index=idx)
        s = _compute_summary(eq, spy, [])
        assert s["strategy"]["max_dd"] < 0


class TestFormatReport:

    def test_no_error_in_output(self):
        eq  = pd.Series([100_000, 101_000, 102_000],
                        index=pd.date_range("2020-01-01", periods=3))
        spy = pd.Series([100_000, 100_500, 101_000],
                        index=pd.date_range("2020-01-01", periods=3))
        result = {
            "equity_curve": eq,
            "spy_curve": spy,
            "trade_log": [],
            "summary": _compute_summary(eq, spy, []) | {
                "start": "2020-01-01", "end": "2020-01-03",
                "initial_capital": 100_000, "rebalance_count": 0,
            },
        }
        report = format_backtest_report(result)
        assert "BACKTEST RESULTS" in report
        assert "Strategy" in report

    def test_error_case(self):
        report = format_backtest_report({"error": "test error"})
        assert "BACKTEST ERROR" in report
        assert "test error" in report
