"""
Portfolio correlation analysis.

Computes rolling correlations between ETF components to verify diversification
and identify periods where factors converge (risk-on/risk-off).

Key insight: when correlations spike toward 1.0, diversification breaks down
and the portfolio behaves like a single bet. Regime detection should catch this,
but explicit correlation monitoring is a secondary check.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import config

logger = logging.getLogger(__name__)


def compute_correlation_matrix(
    start: str = "2022-01-01",
    end: Optional[str] = None,
    window: int = 63,          # rolling 63-day (quarterly) window
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute full and rolling correlation matrices for the ETF portfolio.

    Parameters
    ----------
    start  : Start date
    end    : End date (defaults to today)
    window : Rolling window in trading days

    Returns
    -------
    (full_corr, rolling_avg_corr) : Full period correlation matrix and
                                    time series of average pairwise correlation
    """
    if end is None:
        from datetime import datetime
        end = datetime.today().strftime("%Y-%m-%d")

    tickers = list(config.ETF_TARGET_WEIGHTS.keys())
    raw = yf.download(tickers + ["SPY"], start=start, end=end,
                      auto_adjust=True, progress=False)

    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw

    returns = prices.pct_change().dropna()

    # Full period correlation matrix
    full_corr = returns[tickers].corr()

    # Rolling average pairwise correlation
    pairs = [(a, b) for i, a in enumerate(tickers) for b in tickers[i+1:]]
    rolling_cors = {}
    for (a, b) in pairs:
        if a in returns.columns and b in returns.columns:
            rolling_cors[f"{a}-{b}"] = returns[a].rolling(window).corr(returns[b])

    if rolling_cors:
        rolling_df = pd.DataFrame(rolling_cors)
        avg_pairwise = rolling_df.mean(axis=1).dropna()
    else:
        avg_pairwise = pd.Series(dtype=float)

    return full_corr, avg_pairwise


def diversification_score(corr_matrix: pd.DataFrame) -> float:
    """
    Compute a diversification score [0, 1] from correlation matrix.

    Score = 1 - avg(off-diagonal correlations)
    0 = perfectly correlated (no diversification)
    1 = perfectly uncorrelated (maximum diversification)
    """
    n = len(corr_matrix)
    if n < 2:
        return 0.0
    # Average of upper triangle (off-diagonal)
    upper = []
    tickers = list(corr_matrix.index)
    for i in range(n):
        for j in range(i + 1, n):
            upper.append(float(corr_matrix.iloc[i, j]))
    avg_corr = np.mean(upper) if upper else 0.0
    return max(0.0, 1.0 - avg_corr)


def format_correlation_report(corr: pd.DataFrame, avg_pairwise: pd.Series) -> str:
    """ASCII-safe correlation report."""
    lines = [
        "=" * 72,
        "PORTFOLIO CORRELATION ANALYSIS",
        "=" * 72,
        "",
        "FULL-PERIOD CORRELATION MATRIX",
        "-" * 40,
    ]

    tickers = list(corr.index)
    # Header row
    header = f"{'':>6}" + "".join(f"{t:>7}" for t in tickers)
    lines.append(header)
    for t in tickers:
        row = f"{t:>6}" + "".join(f"{corr.loc[t, u]:>7.2f}" for u in tickers)
        lines.append(row)

    div_score = diversification_score(corr)
    lines += [
        "",
        f"Diversification Score: {div_score:.3f} (1.0 = fully uncorrelated)",
        f"Average pairwise correlation: {1 - div_score:.3f}",
        "",
    ]

    if not avg_pairwise.empty:
        recent_corr = float(avg_pairwise.iloc[-1])
        max_corr    = float(avg_pairwise.max())
        min_corr    = float(avg_pairwise.min())
        lines += [
            "ROLLING 63-DAY AVERAGE PAIRWISE CORRELATION",
            "-" * 40,
            f"Current (most recent): {recent_corr:.3f}",
            f"Maximum (worst case):  {max_corr:.3f}",
            f"Minimum (best case):   {min_corr:.3f}",
            "",
        ]
        if recent_corr > 0.7:
            lines.append("WARNING: High correlation detected -- diversification may be limited")
        elif recent_corr > 0.5:
            lines.append("CAUTION: Moderate correlation -- monitor closely")
        else:
            lines.append("OK: Low correlation -- good diversification")

    lines += ["", "=" * 72]
    return "\n".join(lines)
