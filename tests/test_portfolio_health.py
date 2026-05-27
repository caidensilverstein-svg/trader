"""Unit tests for core/portfolio_health.py."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from core.portfolio_health import (
    run_health_checks,
    format_health_report,
    HealthCheck,
    PortfolioHealthReport,
)


def _good_inputs() -> dict:
    return dict(
        drawdown_pct=-0.05,
        var_95_pct=0.008,
        regime="BULL",
        hhi_sector=1200,
        avg_correlation=0.35,
        portfolio_sharpe=0.74,
        momentum_penalties=0,
        days_to_exit=0.5,
        vol_scalar=0.80,
    )


def _bad_inputs() -> dict:
    return dict(
        drawdown_pct=-0.22,
        var_95_pct=0.025,
        regime="BEAR_CRISIS",
        hhi_sector=3000,
        avg_correlation=0.80,
        portfolio_sharpe=-0.10,
        momentum_penalties=3,
        days_to_exit=8.0,
        vol_scalar=0.30,
    )


class TestRunHealthChecks:

    def test_returns_health_report(self):
        result = run_health_checks(**_good_inputs())
        assert isinstance(result, PortfolioHealthReport)

    def test_good_inputs_return_green(self):
        result = run_health_checks(**_good_inputs())
        assert result.overall_status == "GREEN"

    def test_bad_inputs_return_red_or_halt(self):
        result = run_health_checks(**_bad_inputs())
        assert result.overall_status in ("RED", "HALT")

    def test_n_green_plus_yellow_plus_red_eq_total(self):
        result = run_health_checks(**_good_inputs())
        assert result.n_green + result.n_yellow + result.n_red == len(result.checks)

    def test_9_checks_returned(self):
        result = run_health_checks(**_good_inputs())
        assert len(result.checks) == 9

    def test_each_check_has_valid_status(self):
        result = run_health_checks(**_good_inputs())
        for c in result.checks:
            assert c.status in ("GREEN", "YELLOW", "RED")

    def test_halt_when_many_reds(self):
        result = run_health_checks(**_bad_inputs())
        assert result.overall_status == "HALT"

    def test_bear_crisis_regime_gives_red(self):
        inp = _good_inputs()
        inp["regime"] = "BEAR_CRISIS"
        result = run_health_checks(**inp)
        regime_check = next(c for c in result.checks if c.name == "Market Regime")
        assert regime_check.status == "RED"

    def test_deep_drawdown_gives_red(self):
        inp = _good_inputs()
        inp["drawdown_pct"] = -0.22  # -22%
        result = run_health_checks(**inp)
        dd_check = next(c for c in result.checks if "Drawdown" in c.name)
        assert dd_check.status == "RED"

    def test_high_sector_concentration_gives_red(self):
        inp = _good_inputs()
        inp["hhi_sector"] = 3000
        result = run_health_checks(**inp)
        hhi_check = next(c for c in result.checks if "HHI" in c.name)
        assert hhi_check.status == "RED"

    def test_good_sharpe_gives_green(self):
        result = run_health_checks(**_good_inputs())
        sh_check = next(c for c in result.checks if "Sharpe" in c.name)
        assert sh_check.status == "GREEN"

    def test_low_sharpe_gives_red(self):
        inp = _good_inputs()
        inp["portfolio_sharpe"] = -0.5
        result = run_health_checks(**inp)
        sh_check = next(c for c in result.checks if "Sharpe" in c.name)
        assert sh_check.status == "RED"

    def test_priority_action_reflects_worst_check(self):
        result = run_health_checks(**_bad_inputs())
        assert len(result.priority_action) > 0
        assert "RED" in result.priority_action or "YELLOW" in result.priority_action

    def test_all_green_has_green_priority(self):
        result = run_health_checks(**_good_inputs())
        assert "GREEN" in result.priority_action or "normal" in result.priority_action.lower()

    def test_bsc_scalar_low_gives_red(self):
        inp = _good_inputs()
        inp["vol_scalar"] = 0.30
        result = run_health_checks(**inp)
        bsc = next(c for c in result.checks if "B-SC" in c.name or "Scalar" in c.name)
        assert bsc.status == "RED"

    def test_bear_regime_gives_yellow(self):
        inp = _good_inputs()
        inp["regime"] = "BEAR"
        result = run_health_checks(**inp)
        reg = next(c for c in result.checks if "Regime" in c.name)
        assert reg.status == "YELLOW"


class TestFormatHealthReport:

    def test_contains_overall_status(self):
        report = run_health_checks(**_good_inputs())
        r = format_health_report(report)
        assert "OVERALL STATUS" in r

    def test_contains_green_for_good_inputs(self):
        report = run_health_checks(**_good_inputs())
        r = format_health_report(report)
        assert "GREEN" in r

    def test_contains_check_categories(self):
        report = run_health_checks(**_good_inputs())
        r = format_health_report(report)
        assert "Drawdown" in r or "Risk" in r

    def test_contains_action_for_red(self):
        report = run_health_checks(**_bad_inputs())
        r = format_health_report(report)
        assert "Action" in r

    def test_contains_priority_action(self):
        report = run_health_checks(**_good_inputs())
        r = format_health_report(report)
        assert "Priority" in r
