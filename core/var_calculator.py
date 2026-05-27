"""
Portfolio Value-at-Risk (VaR) and Conditional Value-at-Risk (CVaR).

Two estimation methods:
1. Historical simulation: use actual return distribution (non-parametric)
2. Parametric (Gaussian): assume normal returns for quick estimates

Both methods are computed at 95% and 99% confidence levels.

VaR(alpha) = minimum loss that will not be exceeded with probability alpha
CVaR(alpha) = expected loss given that loss exceeds VaR(alpha)
           = also called Expected Shortfall (ES)

CVaR is preferred for regulatory and risk management purposes because
it is a coherent risk measure (Artzner et al. 1999) and captures
tail risk that VaR misses.

Academic basis: Artzner et al. (1999) "Coherent Measures of Risk"
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _daily_returns(equity_curve: pd.Series) -> np.ndarray:
    """Compute daily returns from equity curve."""
    return equity_curve.pct_change().dropna().values


def historical_var(
    returns: np.ndarray,
    confidence: float = 0.95,
    horizon_days: int = 1,
    portfolio_value: float = 100_000.0,
) -> float:
    """
    Historical simulation VaR.

    Parameters
    ----------
    returns         : Array of daily portfolio returns
    confidence      : Confidence level (0.95 or 0.99)
    horizon_days    : Holding period in days (scaled by sqrt rule)
    portfolio_value : Current portfolio value in dollars

    Returns
    -------
    float : Dollar VaR (positive = potential loss amount)
    """
    if len(returns) < 30:
        logger.warning("Insufficient history for VaR (%d obs)", len(returns))
        return 0.0

    # Scale to horizon using square-root-of-time (assumes i.i.d.)
    # For correlated returns, a proper term-structure is needed
    sorted_rets = np.sort(returns)
    cutoff_idx  = int(len(sorted_rets) * (1 - confidence))
    var_return  = float(sorted_rets[cutoff_idx])  # negative number

    # Scale to horizon
    var_horizon = var_return * np.sqrt(horizon_days)
    return abs(float(var_horizon * portfolio_value))


def historical_cvar(
    returns: np.ndarray,
    confidence: float = 0.95,
    horizon_days: int = 1,
    portfolio_value: float = 100_000.0,
) -> float:
    """
    Historical simulation Conditional VaR (Expected Shortfall).

    CVaR is the expected loss in the worst (1-confidence)% of scenarios.
    """
    if len(returns) < 30:
        return 0.0

    sorted_rets = np.sort(returns)
    cutoff_idx  = int(len(sorted_rets) * (1 - confidence))
    tail_returns = sorted_rets[:cutoff_idx]  # worst returns

    if len(tail_returns) == 0:
        return 0.0

    mean_tail   = float(tail_returns.mean())
    cvar_horizon = mean_tail * np.sqrt(horizon_days)
    return abs(float(cvar_horizon * portfolio_value))


def parametric_var(
    returns: np.ndarray,
    confidence: float = 0.95,
    horizon_days: int = 1,
    portfolio_value: float = 100_000.0,
) -> float:
    """
    Parametric (Gaussian) VaR.

    Assumes normally distributed returns. Fast but underestimates
    tail risk for fat-tailed financial returns.
    """
    from scipy import stats
    if len(returns) < 10:
        return 0.0

    mu  = float(np.mean(returns))
    sig = float(np.std(returns, ddof=1))

    # Z-score for given confidence level
    z = stats.norm.ppf(1 - confidence)

    daily_var_return = mu + z * sig  # negative number
    horizon_var      = daily_var_return * np.sqrt(horizon_days)
    return abs(float(horizon_var * portfolio_value))


def compute_var_report(
    equity_curve: pd.Series,
    portfolio_value: float = 100_000.0,
) -> Dict[str, float]:
    """
    Full VaR/CVaR report for a portfolio equity curve.

    Returns
    -------
    dict : {metric_label: dollar_amount}
    """
    returns = _daily_returns(equity_curve)

    if len(returns) < 30:
        return {}

    result = {}

    for conf, conf_label in [(0.95, "95"), (0.99, "99")]:
        for horizon, h_label in [(1, "1d"), (10, "10d")]:
            result[f"hist_var_{conf_label}_{h_label}"] = historical_var(
                returns, conf, horizon, portfolio_value)
            result[f"hist_cvar_{conf_label}_{h_label}"] = historical_cvar(
                returns, conf, horizon, portfolio_value)
            result[f"param_var_{conf_label}_{h_label}"] = parametric_var(
                returns, conf, horizon, portfolio_value)

    # Add percentage-based metrics
    pv = max(portfolio_value, 1.0)
    for k in list(result.keys()):
        result[f"{k}_pct"] = round(result[k] / pv * 100, 3)

    # Annualized vol from returns
    result["ann_vol_pct"] = round(float(np.std(returns, ddof=1) * np.sqrt(252) * 100), 2)

    return result


def format_var_report(var_data: Dict[str, float], portfolio_value: float) -> str:
    """Format VaR data as an ASCII table."""
    lines = [
        "=" * 65,
        "VALUE-AT-RISK SUMMARY",
        f"Portfolio Value: ${portfolio_value:,.0f}  |  Method: Historical Simulation",
        "=" * 65,
        f"{'Metric':<30} {'1-Day':>12} {'10-Day':>12}",
        "-" * 58,
    ]

    for conf in ("95", "99"):
        lines.append(f"--- {conf}% Confidence Level ---")
        lines.append(
            f"{'  Hist VaR':<30} "
            f"${var_data.get(f'hist_var_{conf}_1d', 0):>10,.0f} "
            f"${var_data.get(f'hist_var_{conf}_10d', 0):>10,.0f}"
        )
        lines.append(
            f"{'  Hist CVaR (Exp Shortfall)':<30} "
            f"${var_data.get(f'hist_cvar_{conf}_1d', 0):>10,.0f} "
            f"${var_data.get(f'hist_cvar_{conf}_10d', 0):>10,.0f}"
        )
        lines.append(
            f"{'  Parametric VaR (Gaussian)':<30} "
            f"${var_data.get(f'param_var_{conf}_1d', 0):>10,.0f} "
            f"${var_data.get(f'param_var_{conf}_10d', 0):>10,.0f}"
        )

    ann_vol = var_data.get("ann_vol_pct", 0)
    lines += [
        "",
        f"Annualized Volatility: {ann_vol:.2f}%",
        "Note: CVaR > VaR by construction (average of tail losses vs threshold)",
        "Note: Parametric VaR underestimates tail risk for fat-tailed returns",
        "=" * 65,
    ]
    return "\n".join(lines)
