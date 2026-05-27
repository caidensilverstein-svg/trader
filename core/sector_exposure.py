"""
Sector exposure decomposition for the ETF sleeve.

Estimates portfolio sector exposure by blending known ETF sector weights.
ETF sector composition is based on published iShares/Avantis fact sheets.

Returns a breakdown of:
  - Estimated sector weights (% of total portfolio)
  - Sector concentration (Herfindahl index)
  - Comparison to SPY sector weights (benchmark)
  - Active sector bets (portfolio vs benchmark)

Academic basis: Barra (1998) factor model for sector risk decomposition.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# Approximate sector weights for each ETF (based on published fact sheets)
# Values are fractions (not percentages)
ETF_SECTOR_WEIGHTS: Dict[str, Dict[str, float]] = {
    "AVUV": {  # Avantis US Small Cap Value -- sector tilted
        "Financials":       0.28,
        "Industrials":      0.19,
        "Consumer Disc":    0.12,
        "Energy":           0.08,
        "Real Estate":      0.07,
        "Materials":        0.07,
        "Health Care":      0.06,
        "Info Technology":  0.05,
        "Consumer Staples": 0.04,
        "Utilities":        0.03,
        "Communication":    0.01,
    },
    "AVDV": {  # Avantis Intl Small Cap Value -- international
        "Financials":       0.32,
        "Industrials":      0.21,
        "Materials":        0.10,
        "Consumer Disc":    0.09,
        "Consumer Staples": 0.07,
        "Real Estate":      0.05,
        "Energy":           0.05,
        "Health Care":      0.04,
        "Info Technology":  0.04,
        "Utilities":        0.02,
        "Communication":    0.01,
    },
    "QMOM": {  # Alpha Architect Quality Momentum
        "Info Technology":  0.30,
        "Financials":       0.18,
        "Health Care":      0.15,
        "Consumer Disc":    0.12,
        "Industrials":      0.10,
        "Communication":    0.07,
        "Consumer Staples": 0.04,
        "Energy":           0.02,
        "Materials":        0.01,
        "Real Estate":      0.01,
        "Utilities":        0.00,
    },
    "DBMF": {  # Managed futures -- non-equity, all sectors approx 0
        "Managed Futures":  1.00,
    },
    "CTA":  {  # CTA trend following -- non-equity
        "Managed Futures":  1.00,
    },
    "CASH": {
        "Cash":             1.00,
    },
}

# SPY sector weights (approx, as of 2025)
SPY_SECTOR_WEIGHTS: Dict[str, float] = {
    "Info Technology":  0.32,
    "Financials":       0.13,
    "Health Care":      0.12,
    "Consumer Disc":    0.11,
    "Communication":    0.09,
    "Industrials":      0.08,
    "Consumer Staples": 0.06,
    "Energy":           0.04,
    "Materials":        0.02,
    "Real Estate":      0.02,
    "Utilities":        0.01,
}


@dataclass
class SectorStats:
    sector:            str
    portfolio_weight:  float    # % of total portfolio in this sector
    spy_weight:        float    # SPY benchmark weight
    active_bet:        float    # portfolio_weight - spy_weight (active overweight)
    is_overweight:     bool


def compute_sector_exposure(
    etf_weights: Dict[str, float],
) -> Tuple[List[SectorStats], float]:
    """
    Compute portfolio sector exposure.

    Parameters
    ----------
    etf_weights : {ticker: portfolio_weight_fraction} (e.g. {"AVUV": 0.20, ...})

    Returns
    -------
    (sector_stats_list, herfindahl_index)
    """
    # Aggregate sector weights across ETFs
    sector_totals: Dict[str, float] = {}

    for ticker, port_weight in etf_weights.items():
        etf_sectors = ETF_SECTOR_WEIGHTS.get(ticker, {})
        for sector, etf_sector_weight in etf_sectors.items():
            sector_totals[sector] = sector_totals.get(sector, 0.0) + (
                port_weight * etf_sector_weight
            )

    # Normalize to ensure sum = 1
    total = sum(sector_totals.values())
    if total > 0:
        sector_totals = {k: v / total for k, v in sector_totals.items()}

    # Herfindahl-Hirschman Index (concentration)
    equity_sectors = {k: v for k, v in sector_totals.items()
                      if k not in ("Managed Futures", "Cash")}
    equity_total = sum(equity_sectors.values())
    if equity_total > 0:
        eq_normalized = {k: v / equity_total for k, v in equity_sectors.items()}
        hhi = sum(w ** 2 for w in eq_normalized.values()) * 10_000
    else:
        hhi = 0.0

    # Build SectorStats list
    all_sectors = set(list(sector_totals.keys()) + list(SPY_SECTOR_WEIGHTS.keys()))
    results = []
    for sector in sorted(all_sectors):
        port_w = sector_totals.get(sector, 0.0) * 100
        spy_w  = SPY_SECTOR_WEIGHTS.get(sector, 0.0) * 100
        active = port_w - spy_w
        results.append(SectorStats(
            sector=sector,
            portfolio_weight=round(port_w, 1),
            spy_weight=round(spy_w, 1),
            active_bet=round(active, 1),
            is_overweight=(active > 0),
        ))

    results = sorted(results, key=lambda s: s.portfolio_weight, reverse=True)
    return results, round(hhi, 1)


def sector_concentration_score(hhi: float) -> str:
    """Classify sector concentration based on HHI."""
    if hhi < 1500:
        return "LOW (well diversified)"
    elif hhi < 2500:
        return "MODERATE"
    else:
        return "HIGH (concentrated)"


def format_sector_report(stats: List[SectorStats], hhi: float) -> str:
    """Format sector exposure as ASCII table."""
    if not stats:
        return "Sector exposure data unavailable."

    lines = [
        "=" * 75,
        "SECTOR EXPOSURE DECOMPOSITION",
        "(ETF sleeve only; managed futures/cash allocated separately)",
        "=" * 75,
        f"{'Sector':<20} {'Portfolio':>10} {'SPY Bench':>10} {'Active Bet':>11} {'vs Bench'}",
        "-" * 65,
    ]
    for s in stats:
        overweight = "OVER " if s.is_overweight else "UNDER"
        if s.portfolio_weight == 0 and s.spy_weight == 0:
            continue
        lines.append(
            f"{s.sector:<20} {s.portfolio_weight:>8.1f}% {s.spy_weight:>9.1f}% "
            f"{s.active_bet:>+9.1f}% {overweight}"
        )
    lines += [
        "",
        f"Sector HHI (equity portion): {hhi:.1f} -- {sector_concentration_score(hhi)}",
        "(HHI < 1500 = diversified, 1500-2500 = moderate, > 2500 = concentrated)",
        "",
        "Active bets reflect factor tilts:",
        "  OVER: Financials (value tilt from AVUV/AVDV)",
        "  OVER: Industrials (small-cap value tilt)",
        "  UNDER: Info Technology (avoid growth stocks)",
        "  UNDER: Communication (avoid growth/mega-cap)",
        "=" * 75,
    ]
    return "\n".join(lines)
