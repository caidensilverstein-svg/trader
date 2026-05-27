"""Unit tests for backtest/rolling_metrics.py."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest

from backtest.rolling_metrics import (
    compute_rolling_metrics,
    rolling_stability_score,
    format_rolling_report,
)


def _make_equity(n=500, drift=0.0003, seed=42) -> pd.Series:
    np.random.seed(seed)
    rets = np.random.normal(drift, 0.01, n)
    dates = pd.bdate_range("2020-01-01", periods=n)
    return pd.Series(100_000 * np.cumprod(1 + rets), index=dates)


class TestComputeRollingMetrics:

    def test_returns_dataframe_and_dict(self):
        eq = _make_equity(500)
        df, summary = compute_rolling_metrics(eq)
        assert isinstance(df, pd.DataFrame)
        assert isinstance(summary, dict)

    def test_dataframe_has_required_columns(self):
        eq = _make_equity(500)
        df, _ = compute_rolling_metrics(eq)
        for col in ("ret_63d", "ret_252d", "sharpe_63d", "sharpe_252d", "vol_63d"):
            assert col in df.columns

    def test_insufficient_data_returns_empty(self):
        eq = _make_equity(30)
        df, summary = compute_rolling_metrics(eq)
        assert df.empty
        assert summary == {}

    def test_rolling_sharpe_finite_for_long_series(self):
        eq = _make_equity(600)
        df, _ = compute_rolling_metrics(eq)
        sharpe = df["sharpe_252d"].dropna()
        assert len(sharpe) > 0
        assert all(np.isfinite(v) for v in sharpe)

    def test_vol_strictly_positive(self):
        eq = _make_equity(500)
        df, _ = compute_rolling_metrics(eq)
        vol = df["vol_63d"].dropna()
        assert all(v > 0 for v in vol)

    def test_positive_drift_gives_positive_avg_return(self):
        eq = _make_equity(500, drift=0.001)
        df, _ = compute_rolling_metrics(eq)
        ret = df["ret_252d"].dropna()
        assert float(ret.mean()) > 0

    def test_summary_keys_present(self):
        eq = _make_equity(500)
        _, summary = compute_rolling_metrics(eq)
        assert "sharpe_252d" in summary or "ret_252d" in summary

    def test_calmar_252d_column_present(self):
        eq = _make_equity(500)
        df, _ = compute_rolling_metrics(eq)
        assert "calmar_252d" in df.columns

    def test_returns_indexed_by_date(self):
        eq = _make_equity(400)
        df, _ = compute_rolling_metrics(eq)
        assert isinstance(df.index, pd.DatetimeIndex)

    def test_negative_drift_gives_negative_avg_return(self):
        eq = _make_equity(500, drift=-0.001)
        df, _ = compute_rolling_metrics(eq)
        ret = df["ret_252d"].dropna()
        assert float(ret.mean()) < 0


class TestRollingStabilityScore:

    def _make_df_with_sharpe(self, values) -> pd.DataFrame:
        dates = pd.bdate_range("2020-01-01", periods=len(values))
        return pd.DataFrame({"sharpe_252d": values}, index=dates)

    def test_returns_dict_with_expected_keys(self):
        df = self._make_df_with_sharpe([0.5] * 100)
        result = rolling_stability_score(df)
        for key in ("stability_score", "pct_positive_sharpe", "min_sharpe_252d"):
            assert key in result

    def test_constant_positive_sharpe_high_stability(self):
        df = self._make_df_with_sharpe([1.0] * 100)
        result = rolling_stability_score(df)
        assert result["pct_positive_sharpe"] == 100.0

    def test_mixed_sharpe_lower_pct_positive(self):
        vals = [1.0] * 50 + [-0.5] * 50
        df = self._make_df_with_sharpe(vals)
        result = rolling_stability_score(df)
        assert result["pct_positive_sharpe"] == 50.0

    def test_min_sharpe_correct(self):
        df = self._make_df_with_sharpe([1.0, 0.5, -2.0, 0.8])
        result = rolling_stability_score(df)
        assert abs(result["min_sharpe_252d"] - (-2.0)) < 0.01

    def test_empty_df_returns_empty(self):
        result = rolling_stability_score(pd.DataFrame())
        assert result == {}

    def test_score_between_0_and_100(self):
        eq = _make_equity(600)
        df, _ = compute_rolling_metrics(eq)
        result = rolling_stability_score(df)
        assert 0 <= result["stability_score"] <= 100


class TestFormatRollingReport:

    def test_contains_header(self):
        eq = _make_equity(500)
        df, summary = compute_rolling_metrics(eq)
        r = format_rolling_report(df, summary)
        assert "ROLLING PERFORMANCE" in r

    def test_contains_sharpe_stability(self):
        eq = _make_equity(600)
        df, summary = compute_rolling_metrics(eq)
        r = format_rolling_report(df, summary)
        assert "SHARPE STABILITY" in r

    def test_insufficient_data_returns_message(self):
        r = format_rolling_report(pd.DataFrame(), {})
        assert "Insufficient" in r

    def test_contains_date_entries(self):
        eq = _make_equity(500)
        df, summary = compute_rolling_metrics(eq)
        r = format_rolling_report(df, summary)
        assert "2020" in r or "2021" in r
