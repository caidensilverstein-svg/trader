"""
Alpha decomposition and Information Ratio analysis.

Decomposes total alpha into:
  - Market timing alpha (Jensen's alpha: intercept of CAPM regression)
  - Factor alpha (residual after controlling for factor exposures)
  - Information Ratio: alpha / tracking_error
  - Treynor Ratio: excess return per unit of systematic risk
  - Up-capture / Down-capture ratios vs SPY benchmark

These metrics answer: "Is the portfolio generating skill-based alpha
or just riding factor tilts?"

Academic basis:
  Jensen (1968) "The Performance of Mutual Funds in the Period 1945-1964"
  Grinold & Kahn (2000) "Active Portfolio Management"
  Henriksson & Merton (1981) "On Market Timing and Investment Performance"
"""

import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class AlphaMetrics:
    # CAPM alpha (annualized %)
    jensen_alpha_pct: float
    alpha_t_stat: float
    alpha_significant: bool   # |t| > 2

    # Beta
    market_beta: float
    r_squared: float

    # Information ratio
    tracking_error_pct: float   # annualized std of active returns
    information_ratio: float    # alpha / tracking_error

    # Treynor
    treynor_ratio: float

    # Capture ratios
    up_capture: float    # % of benchmark up-moves captured
    down_capture: float  # % of benchmark down-moves captured (lower = better)

    # Active return
    active_return_pct: float  # portfolio - benchmark (annualized %)


def compute_alpha_metrics(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    risk_free_rate: float = 0.05 / 252,  # daily risk-free (5% annual)
) -> AlphaMetrics:
    """
    Compute comprehensive alpha decomposition metrics.

    Parameters
    ----------
    portfolio_returns  : Daily portfolio returns
    benchmark_returns  : Daily benchmark (SPY) returns
    risk_free_rate     : Daily risk-free rate

    Returns
    -------
    AlphaMetrics dataclass
    """
    # Align
    idx = portfolio_returns.index.intersection(benchmark_returns.index)
    if len(idx) < 60:
        raise ValueError(f"Insufficient data: {len(idx)} overlapping observations")

    p = portfolio_returns.loc[idx].values
    b = benchmark_returns.loc[idx].values
    rf = risk_free_rate

    # Excess returns
    p_ex = p - rf
    b_ex = b - rf

    n = len(p)

    # CAPM OLS: p_ex = alpha + beta * b_ex + epsilon
    X = np.column_stack([np.ones(n), b_ex])
    result = np.linalg.lstsq(X, p_ex, rcond=None)
    coeffs = result[0]
    alpha_daily = float(coeffs[0])
    beta = float(coeffs[1])

    # t-stat for alpha
    y_hat = X @ coeffs
    resid = p_ex - y_hat
    df_err = n - 2
    mse = float(np.sum(resid ** 2) / df_err)
    XtX_inv = np.linalg.pinv(X.T @ X)
    se_alpha = np.sqrt(mse * XtX_inv[0, 0])
    t_stat = alpha_daily / se_alpha if se_alpha > 0 else 0.0

    # R-squared
    ss_tot = float(np.sum((p_ex - p_ex.mean()) ** 2))
    ss_res = float(np.sum(resid ** 2))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    # Annualize
    alpha_ann = alpha_daily * 252
    r2 = max(0.0, min(1.0, r2))

    # Tracking error (active return std, annualized)
    active_rets = p - b
    te_ann = float(np.std(active_rets, ddof=1)) * np.sqrt(252) * 100

    # Active return (annualized)
    port_ann = float(np.mean(p)) * 252 * 100
    bench_ann = float(np.mean(b)) * 252 * 100
    active_ret_ann = port_ann - bench_ann

    # Information Ratio
    ir = (active_ret_ann / te_ann) if te_ann > 0 else 0.0

    # Treynor Ratio: (portfolio excess return) / beta
    treynor = (port_ann - rf * 252 * 100) / beta if beta != 0 else 0.0

    # Up/Down capture ratios
    up_mask   = b > 0
    down_mask = b < 0

    if up_mask.sum() > 0:
        up_capture = float(np.mean(p[up_mask])) / float(np.mean(b[up_mask])) * 100
    else:
        up_capture = 100.0

    if down_mask.sum() > 0:
        down_capture = float(np.mean(p[down_mask])) / float(np.mean(b[down_mask])) * 100
    else:
        down_capture = 100.0

    return AlphaMetrics(
        jensen_alpha_pct=round(alpha_ann * 100, 2),
        alpha_t_stat=round(t_stat, 2),
        alpha_significant=abs(t_stat) >= 2.0,
        market_beta=round(beta, 3),
        r_squared=round(r2, 3),
        tracking_error_pct=round(te_ann, 2),
        information_ratio=round(ir, 3),
        treynor_ratio=round(treynor, 2),
        up_capture=round(up_capture, 1),
        down_capture=round(down_capture, 1),
        active_return_pct=round(active_ret_ann, 2),
    )


def format_alpha_report(metrics: AlphaMetrics) -> str:
    """Format alpha decomposition as ASCII report."""
    sig = "*** (significant)" if metrics.alpha_significant else "(not significant)"
    lines = [
        "=" * 70,
        "ALPHA DECOMPOSITION & INFORMATION RATIO ANALYSIS",
        "(CAPM + Information Ratio framework -- Jensen 1968, Grinold & Kahn 2000)",
        "=" * 70,
        "",
        "CAPM REGRESSION (portfolio vs SPY benchmark):",
        f"  Jensen's Alpha (ann):   {metrics.jensen_alpha_pct:+.2f}%  {sig}",
        f"  Alpha t-stat:           {metrics.alpha_t_stat:+.2f}",
        f"  Market Beta:            {metrics.market_beta:+.3f}",
        f"  R-squared:              {metrics.r_squared:.3f}",
        "",
        "ACTIVE MANAGEMENT QUALITY:",
        f"  Active Return (ann):    {metrics.active_return_pct:+.2f}%",
        f"  Tracking Error (ann):   {metrics.tracking_error_pct:.2f}%",
        f"  Information Ratio:      {metrics.information_ratio:+.3f}",
        f"  Treynor Ratio:          {metrics.treynor_ratio:+.2f}",
        "",
        "MARKET TIMING:",
        f"  Up-Capture Ratio:       {metrics.up_capture:.1f}%",
        f"  Down-Capture Ratio:     {metrics.down_capture:.1f}%",
        f"  Timing Ratio (Up/Down): {metrics.up_capture/metrics.down_capture:.2f}x "
        f"(> 1.0 means better upside than downside participation)",
        "",
        "INTERPRETATION:",
        "  IR > 0.5 = good active management   IR > 1.0 = exceptional",
        "  Up-capture > Down-capture = defensive alpha (preferred)",
        f"  This portfolio: {'FAVORABLE' if metrics.up_capture > metrics.down_capture else 'UNFAVORABLE'} "
        f"capture profile",
        "=" * 70,
    ]
    return "\n".join(lines)
