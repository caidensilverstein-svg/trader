"""
Rolling performance metrics analysis.

Computes rolling Sharpe ratio, rolling Calmar ratio, rolling annual return,
and rolling volatility over configurable windows (63/126/252 trading days).

Rolling windows reveal:
  - Performance stability over time
  - Regime-driven performance changes
  - Whether strong returns are concentrated or persistent

Academic basis: Lo (2002) "The Statistics of Sharpe Ratios"
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

WINDOWS = {
    "63d":  63,   # 1 quarter
    "126d": 126,  # 1 half-year
    "252d": 252,  # 1 year
}


@dataclass
class RollingSnapshot:
    """Point-in-time rolling metric snapshot."""
    date:     str
    ret_63:   Optional[float]   # annualised return (63-day window)
    ret_252:  Optional[float]   # annualised return (252-day window)
    sharpe_63:  Optional[float]
    sharpe_252: Optional[float]
    vol_63:     Optional[float]  # annualised vol %
    calmar_252: Optional[float]  # rolling Calmar (rolling 252d return / 252d max DD)


def compute_rolling_metrics(
    equity_curve: pd.Series,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Compute rolling performance metrics for the equity curve.

    Parameters
    ----------
    equity_curve : Daily portfolio values indexed by date

    Returns
    -------
    (df, summary) where df has columns for each rolling metric and
    summary is {metric_name: current_value} for the most recent window
    """
    if len(equity_curve) < 63:
        return pd.DataFrame(), {}

    eq = equity_curve.copy()
    eq.index = pd.to_datetime(eq.index)
    daily_ret = eq.pct_change()

    def _rolling_sharpe(rets: pd.Series, window: int) -> pd.Series:
        mu  = rets.rolling(window).mean() * 252
        vol = rets.rolling(window).std() * np.sqrt(252)
        return (mu / vol).replace([np.inf, -np.inf], np.nan)

    def _rolling_annual_ret(eq_: pd.Series, window: int) -> pd.Series:
        return eq_.pct_change(window) * (252 / window) * 100

    def _rolling_vol(rets: pd.Series, window: int) -> pd.Series:
        return rets.rolling(window).std() * np.sqrt(252) * 100

    def _rolling_calmar(eq_: pd.Series, window: int) -> pd.Series:
        roll_ret = eq_.pct_change(window) * (252 / window) * 100
        roll_dd  = pd.Series(index=eq_.index, dtype=float)
        for i in range(window, len(eq_)):
            seg = eq_.iloc[i - window: i + 1]
            cummax = seg.cummax()
            dd = (seg / cummax - 1.0).min() * 100
            roll_dd.iloc[i] = dd
        calmar = roll_ret / roll_dd.abs()
        return calmar.replace([np.inf, -np.inf], np.nan)

    df = pd.DataFrame(index=eq.index)
    df["ret_63d"]    = _rolling_annual_ret(eq, 63)
    df["ret_252d"]   = _rolling_annual_ret(eq, 252)
    df["sharpe_63d"] = _rolling_sharpe(daily_ret, 63)
    df["sharpe_252d"]= _rolling_sharpe(daily_ret, 252)
    df["vol_63d"]    = _rolling_vol(daily_ret, 63)

    # Calmar is expensive; compute only 252-day version
    df["calmar_252d"] = _rolling_calmar(eq, 252)

    # Round for storage efficiency
    df = df.round(3)

    # Summary: most recent non-NaN values
    summary = {}
    last = df.dropna(how="all").iloc[-1] if len(df.dropna(how="all")) > 0 else pd.Series()
    for col in df.columns:
        val = last.get(col, np.nan)
        summary[col] = round(float(val), 3) if not np.isnan(val) else None

    return df, summary


def rolling_stability_score(df: pd.DataFrame) -> Dict[str, float]:
    """
    Score (0-100) how stable the rolling Sharpe has been.
    Higher = more consistent (lower std of rolling Sharpe).

    Returns dict with 'stability_score', 'pct_positive_sharpe', 'min_sharpe_252d'.
    """
    if "sharpe_252d" not in df.columns:
        return {}

    s = df["sharpe_252d"].dropna()
    if len(s) < 2:
        return {}

    # Coefficient of variation of Sharpe (inverted and scaled)
    mean_s = float(s.mean())
    std_s  = float(s.std())
    cv     = std_s / abs(mean_s) if mean_s != 0 else 99
    stability = max(0.0, min(100.0, (1.0 - cv) * 100))

    pct_pos = float((s > 0).mean()) * 100
    min_s   = float(s.min())

    return {
        "stability_score":    round(stability, 1),
        "pct_positive_sharpe": round(pct_pos, 1),
        "min_sharpe_252d":    round(min_s, 2),
        "mean_sharpe_252d":   round(mean_s, 2),
        "std_sharpe_252d":    round(std_s, 2),
    }


def format_rolling_report(df: pd.DataFrame, summary: Dict[str, float]) -> str:
    """Format rolling metrics as a summary table (most recent 12 quarters)."""
    if df.empty:
        return "Insufficient data for rolling metrics."

    # Sample at quarter-end frequency
    quarterly = df.resample("QE").last()
    # Keep last 12 quarters
    quarterly = quarterly.tail(12)

    stability = rolling_stability_score(df)

    lines = [
        "=" * 80,
        "ROLLING PERFORMANCE METRICS (252-Day Windows)",
        "=" * 80,
        f"{'Date':<12} {'Ann Ret 63d':>12} {'Ann Ret 252d':>13} "
        f"{'Sharpe 63d':>11} {'Sharpe 252d':>12} {'Vol 63d':>8}",
        "-" * 70,
    ]

    for date, row in quarterly.iterrows():
        def _fmt(v, pct=True):
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return "    ---"
            suffix = "%" if pct else " "
            return f"{v:>+6.1f}{suffix}"

        lines.append(
            f"{str(date)[:10]:<12} {_fmt(row.get('ret_63d')):>12} "
            f"{_fmt(row.get('ret_252d')):>13} "
            f"{_fmt(row.get('sharpe_63d'), pct=False):>11} "
            f"{_fmt(row.get('sharpe_252d'), pct=False):>12} "
            f"{_fmt(row.get('vol_63d')):>8}"
        )

    lines += ["", "-" * 80, "CURRENT ROLLING VALUES (most recent window):"]
    for k, v in summary.items():
        if v is not None:
            lines.append(f"  {k:<20}: {v:+.3f}")

    if stability:
        lines += [
            "",
            "SHARPE STABILITY:",
            f"  Stability Score   : {stability.get('stability_score', 0):.1f}/100 "
            f"(higher = more consistent)",
            f"  % Time Sharpe > 0 : {stability.get('pct_positive_sharpe', 0):.1f}%",
            f"  Min 252d Sharpe   : {stability.get('min_sharpe_252d', 0):+.2f}",
            f"  Mean 252d Sharpe  : {stability.get('mean_sharpe_252d', 0):+.2f}",
        ]

    lines += ["", "=" * 80]
    return "\n".join(lines)
