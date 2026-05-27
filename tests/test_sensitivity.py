"""Unit tests for backtest/sensitivity.py."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from backtest.sensitivity import (
    SensitivityResult,
    fragile_parameters,
    format_sensitivity_report,
)


class TestFragileParameters:

    def _make_results(self, deltas: list) -> dict:
        """Build synthetic sensitivity results with given calmar_delta values."""
        param_results = [
            SensitivityResult(
                param_name="REBALANCE_DRIFT_THRESHOLD",
                param_value=v,
                calmar=0.3,
                max_dd=-0.15,
                total_return=0.5,
                n_rebalances=40,
                calmar_delta=d,
                dd_delta=0.0,
            )
            for v, d in zip([0.03, 0.05, 0.07], deltas)
        ]
        return {"REBALANCE_DRIFT_THRESHOLD": param_results}

    def test_robust_when_all_deltas_small(self):
        results = self._make_results([0.05, 0.0, -0.05])
        assert fragile_parameters(results) == []

    def test_fragile_when_one_delta_large(self):
        results = self._make_results([0.30, 0.0, -0.05])  # 30% change
        assert "REBALANCE_DRIFT_THRESHOLD" in fragile_parameters(results)

    def test_custom_threshold(self):
        results = self._make_results([0.15, 0.0, -0.05])
        # At default threshold 0.20, this is robust
        assert fragile_parameters(results, threshold=0.20) == []
        # At stricter threshold 0.10, this is fragile
        assert "REBALANCE_DRIFT_THRESHOLD" in fragile_parameters(results, threshold=0.10)

    def test_multiple_params(self):
        r1 = SensitivityResult("A", 1.0, 0.3, -0.15, 0.5, 40, 0.30, 0.0)
        r2 = SensitivityResult("B", 1.0, 0.3, -0.15, 0.5, 40, 0.05, 0.0)
        results = {"A": [r1], "B": [r2]}
        fragile = fragile_parameters(results)
        assert "A" in fragile
        assert "B" not in fragile


class TestFormatSensitivityReport:

    def _make_results(self) -> dict:
        return {
            "REBALANCE_DRIFT_THRESHOLD": [
                SensitivityResult("REBALANCE_DRIFT_THRESHOLD", 0.03, 0.335,
                                  -0.199, 0.5, 55, 0.167, 0.027),
                SensitivityResult("REBALANCE_DRIFT_THRESHOLD", 0.05, 0.287,
                                  -0.226, 0.4, 40, 0.000, 0.000),
            ]
        }

    def test_report_contains_header(self):
        report = format_sensitivity_report(self._make_results(), 0.287)
        assert "PARAMETER SENSITIVITY ANALYSIS" in report

    def test_report_contains_baseline_calmar(self):
        report = format_sensitivity_report(self._make_results(), 0.287)
        assert "0.287" in report

    def test_report_contains_param_name(self):
        report = format_sensitivity_report(self._make_results(), 0.287)
        assert "REBALANCE_DRIFT_THRESHOLD" in report

    def test_report_shows_calmar_values(self):
        report = format_sensitivity_report(self._make_results(), 0.287)
        assert "0.335" in report
        assert "0.287" in report

    def test_robust_label_when_stable(self):
        results = {
            "REBALANCE_DRIFT_THRESHOLD": [
                SensitivityResult("REBALANCE_DRIFT_THRESHOLD", 0.05,
                                  0.287, -0.226, 0.4, 40, 0.00, 0.0)
            ]
        }
        report = format_sensitivity_report(results, 0.287)
        assert "ROBUST" in report

    def test_fragile_label_when_volatile(self):
        results = {
            "REBALANCE_DRIFT_THRESHOLD": [
                SensitivityResult("REBALANCE_DRIFT_THRESHOLD", 0.03,
                                  0.5, -0.10, 0.6, 60, 0.75, 0.05)
            ]
        }
        report = format_sensitivity_report(results, 0.287)
        assert "FRAGILE" in report


class TestSensitivityResult:

    def test_dataclass_fields(self):
        r = SensitivityResult(
            param_name="X", param_value=0.05, calmar=0.3,
            max_dd=-0.15, total_return=0.5, n_rebalances=40,
        )
        assert r.param_name == "X"
        assert r.calmar_delta == 0.0   # default
        assert r.dd_delta == 0.0      # default
