"""Unit tests for core/sector_exposure.py."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from core.sector_exposure import (
    compute_sector_exposure,
    sector_concentration_score,
    format_sector_report,
    SectorStats,
    ETF_SECTOR_WEIGHTS,
    SPY_SECTOR_WEIGHTS,
)


class TestComputeSectorExposure:

    def test_returns_list_and_hhi(self):
        weights = {"AVUV": 0.30, "AVDV": 0.30, "QMOM": 0.15,
                   "DBMF": 0.15, "CTA": 0.10}
        result, hhi = compute_sector_exposure(weights)
        assert isinstance(result, list)
        assert isinstance(hhi, float)

    def test_portfolio_weights_sum_to_100(self):
        weights = {"AVUV": 0.30, "AVDV": 0.30, "QMOM": 0.15,
                   "DBMF": 0.15, "CTA": 0.10}
        result, _ = compute_sector_exposure(weights)
        total = sum(s.portfolio_weight for s in result)
        assert abs(total - 100.0) < 1.0  # may differ slightly due to rounding

    def test_hhi_between_0_and_100(self):
        weights = {"AVUV": 0.50, "QMOM": 0.50}
        _, hhi = compute_sector_exposure(weights)
        assert 0 <= hhi <= 10_000  # HHI max is 10000 (monopoly = one sector)

    def test_all_one_etf_concentrated(self):
        weights = {"QMOM": 1.0}
        result, hhi = compute_sector_exposure(weights)
        # QMOM is tech-heavy; should have high HHI
        assert hhi > 1000

    def test_spy_weight_from_benchmark(self):
        weights = {"AVUV": 1.0}
        result, _ = compute_sector_exposure(weights)
        fin = next((s for s in result if s.sector == "Financials"), None)
        assert fin is not None
        # AVUV has 28% financials, SPY has 13%
        assert fin.is_overweight

    def test_tech_underweight_vs_spy_for_value_tilt(self):
        # AVUV + AVDV are value-tilted, low tech
        weights = {"AVUV": 0.5, "AVDV": 0.5}
        result, _ = compute_sector_exposure(weights)
        tech = next((s for s in result if s.sector == "Info Technology"), None)
        if tech:
            assert not tech.is_overweight  # value tilt = underweight tech

    def test_managed_futures_not_in_equity_hhi(self):
        # DBMF and CTA should not affect equity HHI
        weights = {"DBMF": 0.5, "CTA": 0.5}
        _, hhi = compute_sector_exposure(weights)
        assert hhi == 0.0  # all managed futures, no equity sector HHI

    def test_sorted_by_portfolio_weight_descending(self):
        weights = {"AVUV": 0.30, "AVDV": 0.30, "QMOM": 0.15,
                   "DBMF": 0.15, "CTA": 0.10}
        result, _ = compute_sector_exposure(weights)
        pw = [s.portfolio_weight for s in result]
        assert pw == sorted(pw, reverse=True)

    def test_empty_weights_returns_empty(self):
        result, hhi = compute_sector_exposure({})
        assert result == [] or all(s.portfolio_weight == 0 for s in result)

    def test_active_bet_correct(self):
        # Single ETF, check active bet = portfolio - spy
        weights = {"AVUV": 1.0}
        result, _ = compute_sector_exposure(weights)
        for s in result:
            assert abs(s.active_bet - (s.portfolio_weight - s.spy_weight)) < 0.01


class TestSectorConcentrationScore:

    def test_low_hhi_diversified(self):
        assert "LOW" in sector_concentration_score(1000)

    def test_moderate_hhi(self):
        assert "MODERATE" in sector_concentration_score(2000)

    def test_high_hhi_concentrated(self):
        assert "HIGH" in sector_concentration_score(3000)


class TestFormatSectorReport:

    def test_contains_header(self):
        weights = {"AVUV": 0.30, "AVDV": 0.30, "QMOM": 0.15,
                   "DBMF": 0.15, "CTA": 0.10}
        result, hhi = compute_sector_exposure(weights)
        r = format_sector_report(result, hhi)
        assert "SECTOR" in r

    def test_contains_sector_names(self):
        weights = {"AVUV": 0.50, "QMOM": 0.50}
        result, hhi = compute_sector_exposure(weights)
        r = format_sector_report(result, hhi)
        assert "Financials" in r

    def test_contains_hhi(self):
        weights = {"AVUV": 0.50, "QMOM": 0.50}
        result, hhi = compute_sector_exposure(weights)
        r = format_sector_report(result, hhi)
        assert "HHI" in r

    def test_empty_returns_message(self):
        r = format_sector_report([], 0)
        assert "unavailable" in r.lower()
