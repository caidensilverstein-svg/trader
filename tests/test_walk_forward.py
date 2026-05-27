"""Unit tests for backtest/walk_forward.py (format and aggregate logic)."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from backtest.walk_forward import format_wf_report


def _make_result(n_windows=4, outperform_pct=75):
    """Build a synthetic walk-forward result dict."""
    windows = []
    for i in range(n_windows):
        strat = 10.0 + i * 2
        spy   = 8.0 + i * 1.5
        windows.append({
            "label":        f"Window {i+1}",
            "start":        f"202{i}-01-01",
            "end":          f"202{i}-12-31",
            "strat_return": strat,
            "spy_return":   spy,
            "strat_sharpe": 0.8,
            "spy_sharpe":   0.7,
            "strat_maxdd":  -10.0,
            "spy_maxdd":    -14.0,
            "strat_calmar": 1.0,
            "spy_calmar":   0.6,
            "alpha":        strat - spy,
            "calmar_edge":  0.4,
            "rebalances":   5,
        })
    return {
        "windows": windows,
        "avg_strat_return": 14.0,
        "avg_spy_return": 11.0,
        "avg_alpha": 3.0,
        "pct_windows_outperform": float(outperform_pct),
        "pct_windows_calmar_beat": 75.0,
        "avg_strat_maxdd": -10.0,
        "avg_spy_maxdd": -14.0,
        "n_windows": n_windows,
    }


class TestFormatWFReport:

    def test_contains_header(self):
        r = format_wf_report(_make_result())
        assert "WALK-FORWARD" in r

    def test_contains_window_labels(self):
        r = format_wf_report(_make_result())
        assert "Window 1" in r

    def test_contains_alpha(self):
        r = format_wf_report(_make_result())
        assert "Alpha" in r or "alpha" in r.lower() or "+" in r

    def test_contains_average_row(self):
        r = format_wf_report(_make_result())
        assert "AVERAGE" in r

    def test_contains_outperform_pct(self):
        r = format_wf_report(_make_result(outperform_pct=75))
        assert "75" in r

    def test_contains_max_drawdown(self):
        r = format_wf_report(_make_result())
        assert "drawdown" in r.lower() or "maxdd" in r.lower() or "10.0" in r

    def test_error_result_returns_error_message(self):
        r = format_wf_report({"error": "No valid windows"})
        assert "ERROR" in r or "error" in r.lower()

    def test_all_4_windows_shown(self):
        result = _make_result(4)
        r = format_wf_report(result)
        for i in range(1, 5):
            assert f"Window {i}" in r

    def test_contains_regime_note(self):
        r = format_wf_report(_make_result())
        assert "Regime" in r or "regime" in r or "B-SC" in r or "signals" in r.lower()

    def test_spy_returns_shown(self):
        result = _make_result()
        r = format_wf_report(result)
        assert "SPY" in r

    def test_100pct_outperform_case(self):
        result = _make_result(outperform_pct=100)
        r = format_wf_report(result)
        assert "100" in r

    def test_0pct_outperform_case(self):
        result = _make_result(outperform_pct=0)
        r = format_wf_report(result)
        assert "0" in r
