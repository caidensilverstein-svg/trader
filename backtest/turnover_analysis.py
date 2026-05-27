"""
Portfolio turnover analysis.

Measures how frequently the portfolio rebalances and quantifies the
cost impact. Key metrics:
  - Annual turnover rate (% of portfolio value traded per year)
  - Average hold time by position
  - Rebalancing frequency (drift-triggered vs scheduled)
  - Transaction cost drag estimate (at 3 bps per side)
  - Tax efficiency score (lower turnover = fewer taxable events)

Academic basis: Carhart (1997) on turnover-adjusted returns;
Grossman & Stiglitz (1980) on informed vs noise trading.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TRANSACTION_COST_BPS = 3.0   # 3 bps per side (6 bps round-trip)


@dataclass
class TurnoverStats:
    annual_turnover_pct:    float   # % of portfolio traded per year
    avg_hold_days:          float   # average days between rebalances
    n_rebalance_events:     int     # total rebalance count
    cost_drag_bps_annual:   float   # annual transaction cost drag (bps)
    cost_drag_pct:          float   # annual cost as % return
    tax_efficiency_score:   float   # 0-100 (100 = buy-and-hold)
    dominant_trigger:       str     # "drift" or "scheduled"


def compute_turnover_from_weights(
    weight_history: pd.DataFrame,
    rebalance_dates: Optional[List] = None,
) -> TurnoverStats:
    """
    Compute turnover from a history of portfolio weights.

    Parameters
    ----------
    weight_history : DataFrame where each row is a date and columns are tickers
                     (values are fractional weights, sum ~1.0)
    rebalance_dates: Optional list of dates when rebalancing occurred

    Returns
    -------
    TurnoverStats
    """
    if len(weight_history) < 2:
        return TurnoverStats(0.0, 0.0, 0, 0.0, 0.0, 100.0, "none")

    # Turnover = sum of absolute weight changes / 2 per rebalance
    diffs = weight_history.diff().abs().dropna()
    turnover_per_event = diffs.sum(axis=1) / 2  # half the abs changes = one side

    if rebalance_dates:
        n_events = len(rebalance_dates)
    else:
        n_events = int((turnover_per_event > 0.01).sum())

    # Annual turnover: average event turnover * events per year
    n_years = len(weight_history) / 252.0
    events_per_year = n_events / n_years if n_years > 0 else 0
    avg_event_turnover = float(turnover_per_event[turnover_per_event > 0.01].mean()) if n_events > 0 else 0
    annual_turnover = avg_event_turnover * events_per_year * 100

    avg_hold = 252 / events_per_year if events_per_year > 0 else float("inf")

    # Cost drag
    cost_drag_bps = annual_turnover * TRANSACTION_COST_BPS * 2 / 100
    cost_drag_pct = cost_drag_bps / 100

    # Tax efficiency: higher turnover = lower score
    # Score of 100 at 0% turnover, score of 0 at >200% turnover
    tax_score = max(0.0, min(100.0, (1 - annual_turnover / 200) * 100))

    # Dominant trigger (heuristic: if events > 12/year, likely drift-triggered)
    dominant = "drift" if events_per_year > 12 else "scheduled"

    return TurnoverStats(
        annual_turnover_pct=round(annual_turnover, 1),
        avg_hold_days=round(avg_hold, 1) if avg_hold != float("inf") else 999.0,
        n_rebalance_events=n_events,
        cost_drag_bps_annual=round(cost_drag_bps, 2),
        cost_drag_pct=round(cost_drag_pct, 4),
        tax_efficiency_score=round(tax_score, 1),
        dominant_trigger=dominant,
    )


def compute_turnover_from_trade_log(
    trade_log: List[dict],
    initial_value: float = 100_000.0,
    n_days: int = 2110,  # ~2018-2026
) -> TurnoverStats:
    """
    Compute turnover from trade log events.
    Sums the dollar value of all BUY trades (one side).
    """
    if not trade_log:
        return TurnoverStats(0.0, 0.0, 0, 0.0, 0.0, 100.0, "none")

    total_buys = 0.0
    rebalance_dates = set()

    for t in trade_log:
        action = t.get("action", "").upper()
        if action in ("BUY", "REBALANCE"):
            qty   = abs(t.get("qty", t.get("quantity", 0)))
            price = t.get("price", t.get("filled_price", 0))
            total_buys += qty * price
            date = t.get("date", t.get("timestamp", ""))[:10]
            rebalance_dates.add(date)

    n_years = n_days / 252.0
    annual_buys = total_buys / n_years
    annual_turnover = (annual_buys / initial_value) * 100

    n_events = len(rebalance_dates)
    events_per_year = n_events / n_years if n_years > 0 else 0
    avg_hold = 252 / events_per_year if events_per_year > 0 else 999.0

    cost_drag_bps = annual_turnover * TRANSACTION_COST_BPS * 2 / 100
    cost_drag_pct = cost_drag_bps / 100
    tax_score = max(0.0, min(100.0, (1 - annual_turnover / 200) * 100))
    dominant = "drift" if events_per_year > 12 else "scheduled"

    return TurnoverStats(
        annual_turnover_pct=round(annual_turnover, 1),
        avg_hold_days=round(avg_hold, 1),
        n_rebalance_events=n_events,
        cost_drag_bps_annual=round(cost_drag_bps, 2),
        cost_drag_pct=round(cost_drag_pct, 4),
        tax_efficiency_score=round(tax_score, 1),
        dominant_trigger=dominant,
    )


def format_turnover_report(stats: TurnoverStats) -> str:
    """Format turnover analysis as ASCII report."""
    lines = [
        "=" * 60,
        "PORTFOLIO TURNOVER ANALYSIS",
        "=" * 60,
        f"Annual Turnover Rate:       {stats.annual_turnover_pct:.1f}%",
        f"Average Hold Duration:      {stats.avg_hold_days:.0f} trading days",
        f"Rebalance Events:           {stats.n_rebalance_events}",
        f"Dominant Rebalance Trigger: {stats.dominant_trigger.upper()}",
        "",
        "TRANSACTION COST ANALYSIS:",
        f"  Cost per side:            {TRANSACTION_COST_BPS:.0f} bps",
        f"  Round-trip cost:          {TRANSACTION_COST_BPS * 2:.0f} bps",
        f"  Annual cost drag:         {stats.cost_drag_bps_annual:.2f} bps",
        f"  Cost as % of return:      {stats.cost_drag_pct:.4f}%",
        "",
        "TAX EFFICIENCY:",
        f"  Tax Efficiency Score:     {stats.tax_efficiency_score:.0f}/100",
        f"  {'HIGHLY TAX-EFFICIENT' if stats.tax_efficiency_score > 80 else 'MODERATE TAX EFFICIENCY' if stats.tax_efficiency_score > 60 else 'LOW TAX EFFICIENCY'}",
        "",
        "BENCHMARKS:",
        "  Buy-and-hold ETF:         ~5-10% annual turnover",
        "  This strategy:            ~15-30% (drift-triggered rebalancing)",
        "  Active stock picker:      100-300% typical",
        "=" * 60,
    ]
    return "\n".join(lines)
