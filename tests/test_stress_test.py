"""Unit tests for backtest/stress_test.py."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest

from backtest.stress_test import (
    run_stress_scenarios,
    stress_test_summary,
    format_stress_report,
    StressScenario,
    HISTORICAL_SCENARIOS,
)


def _make_equity(n=2000, seed=42) -> pd.Series:
    np.random.seed(seed)
    rets = np.random.normal(0.0003, 0.01, n)
    dates = pd.bdate_range("2018-01-01", periods=n)
    return pd.Series(100_000 * np.cumprod(1 + rets), index=dates)


class TestRunStressScenarios:

    def test_returns_list_of_stress_scenarios(self):
        eq = _make_equity()
        result = run_stress_scenarios(eq)
        assert isinstance(result, list)
        assert all(isinstance(s, StressScenario) for s in result)

    def test_number_of_scenarios_matches_definitions(self):
        eq = _make_equity()
        result = run_stress_scenarios(eq)
        assert len(result) == len(HISTORICAL_SCENARIOS)

    def test_sorted_by_estimated_loss(self):
        eq = _make_equity()
        result = run_stress_scenarios(eq)
        losses = [s.estimated_loss_usd for s in result]
        assert losses == sorted(losses)

    def test_worst_scenario_is_gfc(self):
        eq = _make_equity()
        result = run_stress_scenarios(eq)
        # GFC -56.8% should be worst
        assert result[0].name == "GFC 2008"

    def test_estimated_losses_negative_for_scenarios_without_actual_data(self):
        eq = _make_equity()
        for s in run_stress_scenarios(eq):
            if s.portfolio_return_pct is None:
                # Estimated from SPY beta -- must be negative for crisis scenarios
                assert s.estimated_loss_usd < 0

    def test_ratio_less_than_1_for_portfolio_vs_spy(self):
        # Portfolio with beta=0.75 should lose less than SPY
        eq = _make_equity()
        for s in run_stress_scenarios(eq, beta=0.75):
            if s.drawdown_vs_spy is not None:
                # ratio should be < 1 (portfolio lost less)
                assert s.drawdown_vs_spy < 1.0

    def test_covid_scenario_present(self):
        eq = _make_equity()
        result = run_stress_scenarios(eq)
        names = [s.name for s in result]
        assert "COVID Crash" in names

    def test_covid_uses_actual_backtest_data(self):
        eq = _make_equity()
        result = run_stress_scenarios(eq)
        covid = next(s for s in result if s.name == "COVID Crash")
        # Backtest covers 2018-2026, so COVID period should have actual data
        assert covid.portfolio_return_pct is not None

    def test_gfc_uses_estimated_not_actual(self):
        # Backtest starts 2018, GFC was 2007-2009, so must use estimate
        eq = _make_equity()
        result = run_stress_scenarios(eq)
        gfc = next(s for s in result if s.name == "GFC 2008")
        assert gfc.portfolio_return_pct is None

    def test_initial_value_scales_loss(self):
        eq = _make_equity()
        r1 = run_stress_scenarios(eq, initial_value=100_000)
        r2 = run_stress_scenarios(eq, initial_value=200_000)
        # GFC loss should be double
        gfc1 = next(s for s in r1 if s.name == "GFC 2008")
        gfc2 = next(s for s in r2 if s.name == "GFC 2008")
        assert abs(gfc2.estimated_loss_usd / gfc1.estimated_loss_usd - 2.0) < 0.01

    def test_empty_equity_still_runs(self):
        eq = pd.Series([100_000] * 30, index=pd.bdate_range("2020-01-01", periods=30))
        result = run_stress_scenarios(eq)
        assert isinstance(result, list)


class TestStressTestSummary:

    def _make_scenarios(self):
        return [
            StressScenario("GFC", "2007-2009", -56.8, 517, None, -42_600, 0.75),
            StressScenario("COVID", "2020", -33.9, 33, -22.5, -22_500, 0.66),
            StressScenario("Flash", "2010", -9.2, 1, -6.5, -6_500, 0.71),
        ]

    def test_worst_scenario_correct(self):
        smry = stress_test_summary(self._make_scenarios())
        assert smry["worst_scenario"] == "GFC"

    def test_n_scenarios_correct(self):
        smry = stress_test_summary(self._make_scenarios())
        assert smry["n_scenarios"] == 3

    def test_worst_loss_pct_correct(self):
        smry = stress_test_summary(self._make_scenarios())
        assert smry["worst_loss_pct"] < 0

    def test_empty_returns_empty(self):
        assert stress_test_summary([]) == {}


class TestFormatStressReport:

    def test_contains_header(self):
        eq = _make_equity()
        scenarios = run_stress_scenarios(eq)
        r = format_stress_report(scenarios)
        assert "STRESS TEST" in r

    def test_contains_scenario_names(self):
        eq = _make_equity()
        scenarios = run_stress_scenarios(eq)
        r = format_stress_report(scenarios)
        assert "COVID" in r
        assert "GFC" in r

    def test_contains_spy_returns(self):
        eq = _make_equity()
        scenarios = run_stress_scenarios(eq)
        r = format_stress_report(scenarios)
        assert "-33" in r or "-56" in r

    def test_empty_returns_message(self):
        r = format_stress_report([])
        assert "No" in r

    def test_contains_worst_scenario(self):
        eq = _make_equity()
        scenarios = run_stress_scenarios(eq)
        r = format_stress_report(scenarios)
        assert "Worst scenario" in r
