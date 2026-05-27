"""
Bootstrap confidence intervals for backtest statistics.

Stationary bootstrap resampling of daily returns to construct non-parametric
confidence intervals around Sharpe, Calmar, and max drawdown.

Methodology: Politis & Romano (1994) Stationary Bootstrap, with automatic
block length selection via Patton-Politis-White (2009).

Why bootstrap instead of parametric CIs?
  - Returns are fat-tailed and serially correlated
  - Asymptotic normality does not hold for max drawdown
  - Bootstrap captures realistic return path dependence

Usage:
    from backtest.bootstrap import bootstrap_metrics, format_ci_report
    equity = result["equity_curve"]
    cis    = bootstrap_metrics(equity, n_boot=500)
    print(format_ci_report(cis))
"""

import logging
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _optimal_block_length(returns: np.ndarray) -> int:
    """
    Heuristic block length for stationary bootstrap.
    PPW (2009) rule-of-thumb: b = 1.75 * n^(1/3) for financial returns.
    """
    n = len(returns)
    b = max(1, int(1.75 * (n ** (1 / 3))))
    logger.debug("Bootstrap block length: %d (n=%d)", b, n)
    return b


def _stationary_bootstrap_sample(
    returns: np.ndarray,
    block_length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generate one stationary bootstrap resample.

    Geometric block length: each block ends with prob 1/block_length.
    This preserves serial correlation within blocks while ensuring
    the resampled series is stationary.
    """
    n = len(returns)
    sample = np.empty(n)
    i = 0
    while i < n:
        start = rng.integers(0, n)
        geom  = rng.geometric(p=1.0 / block_length)
        length = min(geom, n - i)
        indices = [(start + j) % n for j in range(length)]
        sample[i: i + length] = returns[indices]
        i += length
    return sample[:n]


def _compute_stats(returns: np.ndarray, trading_days: int = 252) -> Dict[str, float]:
    """Compute core performance statistics from a returns array."""
    if len(returns) < 2:
        return {}

    # Annualized return and volatility
    cum_return = float(np.prod(1 + returns) - 1)
    years      = len(returns) / trading_days
    ann_return = float((1 + cum_return) ** (1 / max(years, 0.01)) - 1)
    ann_vol    = float(np.std(returns, ddof=1) * (trading_days ** 0.5))

    sharpe = ann_return / ann_vol if ann_vol > 0 else 0.0

    # Max drawdown from equity curve
    equity = np.cumprod(1 + returns) * 100_000
    peak   = np.maximum.accumulate(equity)
    dd     = (equity / peak) - 1.0
    max_dd = float(dd.min())

    calmar = ann_return / abs(max_dd) if max_dd < 0 else float("inf")

    return {
        "ann_return": ann_return,
        "ann_vol":    ann_vol,
        "sharpe":     sharpe,
        "max_dd":     max_dd,
        "calmar":     min(calmar, 10.0),  # cap at 10x for outlier protection
    }


def bootstrap_metrics(
    equity_curve: pd.Series,
    n_boot: int = 500,
    confidence: float = 0.90,
    seed: int = 42,
) -> Dict[str, Dict[str, float]]:
    """
    Compute bootstrap confidence intervals for portfolio statistics.

    Parameters
    ----------
    equity_curve : Daily portfolio equity values (pd.Series)
    n_boot       : Number of bootstrap samples (500 is sufficient for 90% CI)
    confidence   : Confidence level (default 0.90)
    seed         : Random seed for reproducibility

    Returns
    -------
    dict : {metric_name: {"point": float, "lower": float, "upper": float,
                          "ci_width": float, "significant": bool}}
    """
    returns = equity_curve.pct_change().dropna().values

    if len(returns) < 50:
        logger.warning("Insufficient data for bootstrap (%d obs)", len(returns))
        return {}

    rng          = np.random.default_rng(seed)
    block_length = _optimal_block_length(returns)
    point_stats  = _compute_stats(returns)

    if not point_stats:
        return {}

    # Collect bootstrap statistics
    boot_stats: Dict[str, List[float]] = {k: [] for k in point_stats}

    logger.info("Running %d bootstrap iterations (block=%d)...", n_boot, block_length)
    for _ in range(n_boot):
        sample = _stationary_bootstrap_sample(returns, block_length, rng)
        stats  = _compute_stats(sample)
        for k, v in stats.items():
            if np.isfinite(v):
                boot_stats[k].append(v)

    alpha = 1.0 - confidence
    lo, hi = alpha / 2, 1.0 - alpha / 2

    results = {}
    for metric, point in point_stats.items():
        samples = np.array(boot_stats[metric])
        if len(samples) < 10:
            continue
        lower = float(np.quantile(samples, lo))
        upper = float(np.quantile(samples, hi))
        results[metric] = {
            "point":       round(point, 4),
            "lower":       round(lower, 4),
            "upper":       round(upper, 4),
            "ci_width":    round(upper - lower, 4),
            "significant": (lower > 0 and metric in ("sharpe", "calmar", "ann_return")),
        }

    return results


def format_ci_report(
    cis: Dict[str, Dict[str, float]],
    confidence: float = 0.90,
) -> str:
    """Format bootstrap CI results as an ASCII table."""
    pct = int(confidence * 100)
    lines = [
        "=" * 65,
        f"BOOTSTRAP CONFIDENCE INTERVALS ({pct}%, Stationary Bootstrap)",
        "=" * 65,
        f"{'Metric':<18} {'Point Est':>10} {f'{pct}% CI Lower':>12} {f'{pct}% CI Upper':>12}  Sig",
        "-" * 65,
    ]
    for metric, d in cis.items():
        fmt  = ".1%" if metric in ("ann_return", "ann_vol", "max_dd") else ".3f"
        sig  = "YES" if d["significant"] else "---"
        point = f"{d['point']:{fmt}}"
        lower = f"{d['lower']:{fmt}}"
        upper = f"{d['upper']:{fmt}}"
        lines.append(f"{metric:<18} {point:>10} {lower:>12} {upper:>12}  {sig}")
    lines += ["", "=" * 65]
    return "\n".join(lines)
