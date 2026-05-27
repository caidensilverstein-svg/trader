"""
Drawdown anatomy analysis.

Identifies and characterizes each distinct drawdown period in an equity curve:
  - Start date (when drawdown begins)
  - Trough date (worst point)
  - Recovery date (when full recovery occurs, or None if ongoing)
  - Depth (maximum % decline from peak)
  - Duration in trading days (trough - peak)
  - Recovery time in trading days (recovery - trough, or None)
  - Cause (regime at trough, if trade log provided)

Academic basis: Magdon-Ismail & Atiya (2004)
"Maximum Drawdown for Geometric Brownian Motion"
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class DrawdownPeriod:
    peak_idx:      int
    trough_idx:    int
    recovery_idx:  Optional[int]
    peak_date:     str
    trough_date:   str
    recovery_date: Optional[str]
    depth_pct:     float        # negative, e.g. -7.6
    duration_days: int          # peak to trough
    recovery_days: Optional[int]  # trough to full recovery (None if ongoing)
    peak_value:    float
    trough_value:  float


def find_drawdown_periods(
    equity_curve: pd.Series,
    min_depth_pct: float = -1.0,  # only report drawdowns >= 1%
) -> List[DrawdownPeriod]:
    """
    Identify all distinct drawdown periods in an equity curve.

    A drawdown period begins when the portfolio falls below its
    all-time-high (as of that point) and ends when it recovers.

    Parameters
    ----------
    equity_curve  : Daily portfolio values (indexed by date)
    min_depth_pct : Minimum drawdown depth to report (e.g. -1.0 = ignore < 1%)

    Returns
    -------
    List of DrawdownPeriod objects, sorted by depth (worst first)
    """
    if len(equity_curve) < 5:
        return []

    values = equity_curve.values
    dates  = [str(d)[:10] for d in equity_curve.index]
    n      = len(values)

    periods = []
    i = 0

    while i < n:
        # Find next peak (local or all-time high)
        peak_val = values[i]
        peak_idx = i

        # Walk forward until we start declining
        j = i + 1
        while j < n and values[j] >= peak_val:
            if values[j] > peak_val:
                peak_val = values[j]
                peak_idx = j
            j += 1

        if j >= n:
            break

        # Now in a drawdown; find the trough
        trough_val = values[j]
        trough_idx = j
        k = j + 1
        while k < n and values[k] < peak_val:
            if values[k] < trough_val:
                trough_val = values[k]
                trough_idx = k
            k += 1

        depth = (trough_val / peak_val - 1.0) * 100

        if depth <= min_depth_pct:  # depth is negative, min_depth is also negative
            # Find recovery
            recovery_idx = None
            for m in range(trough_idx + 1, n):
                if values[m] >= peak_val:
                    recovery_idx = m
                    break

            recovery_date  = dates[recovery_idx] if recovery_idx else None
            recovery_days  = (recovery_idx - trough_idx) if recovery_idx else None

            periods.append(DrawdownPeriod(
                peak_idx      = peak_idx,
                trough_idx    = trough_idx,
                recovery_idx  = recovery_idx,
                peak_date     = dates[peak_idx],
                trough_date   = dates[trough_idx],
                recovery_date = recovery_date,
                depth_pct     = round(depth, 2),
                duration_days = trough_idx - peak_idx,
                recovery_days = recovery_days,
                peak_value    = round(peak_val, 2),
                trough_value  = round(trough_val, 2),
            ))

        # Continue from after the trough
        i = trough_idx + 1

    return sorted(periods, key=lambda d: d.depth_pct)


def drawdown_statistics(periods: List[DrawdownPeriod]) -> Dict[str, float]:
    """
    Aggregate statistics across all drawdown periods.

    Returns
    -------
    dict : {stat_name: value}
    """
    if not periods:
        return {}

    depths     = [p.depth_pct for p in periods]
    durations  = [p.duration_days for p in periods]
    recoveries = [p.recovery_days for p in periods if p.recovery_days is not None]

    return {
        "n_drawdowns":       len(periods),
        "worst_dd_pct":      round(min(depths), 2),
        "avg_dd_pct":        round(float(np.mean(depths)), 2),
        "max_duration_days": max(durations),
        "avg_duration_days": round(float(np.mean(durations)), 1),
        "avg_recovery_days": round(float(np.mean(recoveries)), 1) if recoveries else None,
        "pct_recovered":     round(len(recoveries) / len(periods) * 100, 1),
    }


def format_drawdown_report(
    periods: List[DrawdownPeriod],
    n_show: int = 10,
) -> str:
    """Format the worst N drawdown periods as an ASCII table."""
    lines = [
        "=" * 80,
        "DRAWDOWN PERIOD ANALYSIS (Worst to Least)",
        "=" * 80,
        f"{'#':<3} {'Peak':>12} {'Trough':>12} {'Recovery':>12} {'Depth':>7} {'Dur':>5} {'Recov':>6}",
        "-" * 55,
    ]
    worst = sorted(periods, key=lambda d: d.depth_pct)[:n_show]
    for i, p in enumerate(worst, 1):
        recov_str = str(p.recovery_days) if p.recovery_days else "---"
        recov_d   = p.recovery_date or "ongoing"
        lines.append(
            f"{i:<3} {p.peak_date:>12} {p.trough_date:>12} {recov_d:>12} "
            f"{p.depth_pct:>7.2f}% {p.duration_days:>5}d {recov_str:>5}d"
        )
    lines += ["", "=" * 80]
    return "\n".join(lines)
