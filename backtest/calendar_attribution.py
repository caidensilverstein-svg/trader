"""
Calendar-year performance attribution.

Breaks down the equity curve into individual calendar years and computes
per-year return, max drawdown, Sharpe, and best/worst month. Also
computes a benchmark comparison if provided (default: buy-and-hold start
value since we may not have SPY data here, so benchmark is optional).

Academic basis: Common practitioner attribution (GIPS, CFA Institute 2020).
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class YearStats:
    year:          int
    annual_return: float   # % e.g. 12.3
    max_drawdown:  float   # % e.g. -8.5
    sharpe:        float   # annualised (no risk-free rate)
    best_month:    float   # % e.g. 4.2
    worst_month:   float   # % e.g. -3.1
    n_days:        int
    start_value:   float
    end_value:     float
    benchmark_return: Optional[float] = None  # % if benchmark provided


def compute_calendar_attribution(
    equity_curve: pd.Series,
    benchmark_curve: Optional[pd.Series] = None,
) -> List[YearStats]:
    """
    Compute per-calendar-year performance statistics.

    Parameters
    ----------
    equity_curve    : Daily portfolio values indexed by date
    benchmark_curve : Optional benchmark daily values (same index)

    Returns
    -------
    List of YearStats, one per calendar year, sorted chronologically
    """
    if len(equity_curve) < 20:
        return []

    equity_curve = equity_curve.copy()
    equity_curve.index = pd.to_datetime(equity_curve.index)

    results = []

    for year, group in equity_curve.groupby(equity_curve.index.year):
        if len(group) < 5:
            continue

        start_val = float(group.iloc[0])
        end_val   = float(group.iloc[-1])
        annual_ret = (end_val / start_val - 1.0) * 100

        # Daily returns
        daily_rets = group.pct_change().dropna()
        vol = float(daily_rets.std()) * np.sqrt(252)
        sharpe = (float(daily_rets.mean()) * 252) / vol if vol > 0 else 0.0

        # Max drawdown within year
        cummax = group.cummax()
        dd = (group / cummax - 1.0)
        max_dd = float(dd.min()) * 100

        # Monthly best/worst
        monthly = group.resample("ME").last().pct_change().dropna() * 100
        best_m  = float(monthly.max()) if len(monthly) > 0 else 0.0
        worst_m = float(monthly.min()) if len(monthly) > 0 else 0.0

        # Benchmark
        bench_ret = None
        if benchmark_curve is not None:
            bc = benchmark_curve.copy()
            bc.index = pd.to_datetime(bc.index)
            bench_year = bc[bc.index.year == year]
            if len(bench_year) >= 2:
                bench_ret = round(
                    (float(bench_year.iloc[-1]) / float(bench_year.iloc[0]) - 1.0) * 100, 1
                )

        results.append(YearStats(
            year=int(year),
            annual_return=round(annual_ret, 1),
            max_drawdown=round(max_dd, 1),
            sharpe=round(sharpe, 2),
            best_month=round(best_m, 1),
            worst_month=round(worst_m, 1),
            n_days=len(group),
            start_value=round(start_val, 2),
            end_value=round(end_val, 2),
            benchmark_return=bench_ret,
        ))

    return sorted(results, key=lambda s: s.year)


def calendar_summary_stats(years: List[YearStats]) -> Dict[str, float]:
    """Aggregate stats across all calendar years."""
    if not years:
        return {}

    returns = [y.annual_return for y in years]
    dds     = [y.max_drawdown  for y in years]

    positive_years = sum(1 for r in returns if r > 0)
    negative_years = sum(1 for r in returns if r <= 0)
    best_year  = max(years, key=lambda y: y.annual_return)
    worst_year = min(years, key=lambda y: y.annual_return)

    return {
        "n_years":        len(years),
        "positive_years": positive_years,
        "negative_years": negative_years,
        "pct_positive":   round(positive_years / len(years) * 100, 1),
        "avg_annual_ret": round(float(np.mean(returns)), 1),
        "median_annual_ret": round(float(np.median(returns)), 1),
        "best_year_ret":  best_year.annual_return,
        "best_year":      best_year.year,
        "worst_year_ret": worst_year.annual_return,
        "worst_year":     worst_year.year,
        "avg_max_dd":     round(float(np.mean(dds)), 1),
        "std_annual_ret": round(float(np.std(returns, ddof=1)), 1),
    }


def format_calendar_report(years: List[YearStats]) -> str:
    """Format calendar attribution as ASCII table."""
    if not years:
        return "No calendar data available."

    summary = calendar_summary_stats(years)
    has_bench = any(y.benchmark_return is not None for y in years)

    header = (
        f"{'Year':<6} {'Return':>8} {'Max DD':>8} {'Sharpe':>7} "
        f"{'Best Mo':>8} {'Worst Mo':>9}"
    )
    if has_bench:
        header += f" {'Bench':>8} {'Alpha':>7}"

    lines = [
        "=" * 75,
        "CALENDAR YEAR PERFORMANCE ATTRIBUTION",
        "=" * 75,
        header,
        "-" * (75 if not has_bench else 92),
    ]

    for y in years:
        sign = "+" if y.annual_return >= 0 else ""
        row = (
            f"{y.year:<6} {sign}{y.annual_return:>7.1f}% {y.max_drawdown:>7.1f}% "
            f"{y.sharpe:>7.2f} {y.best_month:>+7.1f}% {y.worst_month:>+8.1f}%"
        )
        if has_bench and y.benchmark_return is not None:
            alpha = y.annual_return - y.benchmark_return
            row += f" {y.benchmark_return:>+7.1f}% {alpha:>+6.1f}%"
        lines.append(row)

    lines += [
        "-" * 75,
        f"{'AVG':<6} {summary['avg_annual_ret']:>+7.1f}%          "
        f"        Positive years: {summary['positive_years']}/{summary['n_years']} "
        f"({summary['pct_positive']:.0f}%)",
        "",
        f"Best:  {summary['best_year']} ({summary['best_year_ret']:+.1f}%)    "
        f"Worst: {summary['worst_year']} ({summary['worst_year_ret']:+.1f}%)",
        f"Avg Ann Return: {summary['avg_annual_ret']:+.1f}%    "
        f"Std Dev: {summary['std_annual_ret']:.1f}%    "
        f"Avg Max DD: {summary['avg_max_dd']:.1f}%",
        "=" * 75,
    ]
    return "\n".join(lines)
