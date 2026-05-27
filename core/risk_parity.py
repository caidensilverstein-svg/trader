"""
Risk parity weight computation.

Risk parity allocates capital so each asset contributes equally to
total portfolio risk (volatility). Unlike equal-weight, it automatically
reduces allocation to volatile assets.

Academic basis: Qian (2005) "Risk Parity Portfolios", Maillard et al. (2010)

This is used to:
1. Compare risk parity weights vs our fixed ETF target weights
2. Identify when an ETF has become disproportionately risky
3. Provide an alternative rebalancing target if config changes

We use the simplified "inverse volatility" approximation rather than
full covariance matrix optimization, which requires >60 days of history
and is unstable with small portfolios.
"""

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def inverse_vol_weights(
    returns: Dict[str, pd.Series],
    lookback: int = 63,   # 63 trading days = 1 quarter
    min_weight: float = 0.02,
) -> Dict[str, float]:
    """
    Compute inverse-volatility weights for a set of return series.

    Weight_i = (1 / vol_i) / sum(1 / vol_j)

    Assets with lower realized volatility get higher weights.
    This is the closed-form approximation to risk parity that is stable
    with small N and short history.

    Parameters
    ----------
    returns  : {ticker: pd.Series of daily returns}
    lookback : Window for realized volatility estimate (trading days)
    min_weight : Floor weight for any single asset

    Returns
    -------
    dict : {ticker: fraction_of_total} summing to 1.0
    """
    vols = {}
    for ticker, rets in returns.items():
        r = rets.dropna()
        if len(r) < lookback // 2:
            logger.warning("Insufficient data for %s (%d days), using max vol", ticker, len(r))
            vols[ticker] = float("inf")  # will get minimum weight
        else:
            vol = float(r.tail(lookback).std() * (252 ** 0.5))
            vols[ticker] = max(vol, 1e-6)  # avoid division by zero

    inv_vols = {t: (1.0 / v) if v < float("inf") else 0.0 for t, v in vols.items()}
    total_inv = sum(inv_vols.values())

    if total_inv == 0:
        n = len(returns)
        return {t: 1.0 / n for t in returns}

    raw_weights = {t: iv / total_inv for t, iv in inv_vols.items()}

    # Apply minimum weight floor and re-normalize
    final = {}
    for t, w in raw_weights.items():
        final[t] = max(w, min_weight)

    total = sum(final.values())
    return {t: round(w / total, 4) for t, w in final.items()}


def risk_contribution(
    weights: Dict[str, float],
    returns: Dict[str, pd.Series],
    lookback: int = 63,
) -> Dict[str, float]:
    """
    Compute actual risk contribution of each asset in a portfolio.

    Risk contribution_i = w_i * (Cov * w)_i / portfolio_variance

    Returns fractional risk contribution (should sum to 1.0).
    """
    tickers = sorted(weights.keys())
    w_vec   = np.array([weights[t] for t in tickers])

    # Build return matrix
    ret_matrix = pd.DataFrame({
        t: returns[t].dropna() for t in tickers
        if t in returns
    }).dropna()

    if len(ret_matrix) < lookback // 2 or len(ret_matrix.columns) < 2:
        # Fallback: approximate from individual vols
        vols = {}
        for t in tickers:
            r = returns.get(t, pd.Series()).dropna()
            vols[t] = float(r.std()) if len(r) > 1 else 1e-6
        total_risk = sum(weights[t] * vols[t] for t in tickers)
        return {t: (weights[t] * vols[t]) / total_risk for t in tickers}

    cov = ret_matrix.cov() * 252
    cov_arr = cov.values

    # Only use tickers that are in the return matrix
    available = list(ret_matrix.columns)
    w_avail   = np.array([weights.get(t, 0) for t in available])

    port_var   = w_avail @ cov_arr @ w_avail
    marginal_c = cov_arr @ w_avail
    contrib    = w_avail * marginal_c / port_var if port_var > 0 else w_avail

    return {t: round(float(c), 4) for t, c in zip(available, contrib)}


def compare_to_target(
    rp_weights: Dict[str, float],
    target_weights: Dict[str, float],
    scale_to_etf: float = 0.75,  # ETF sleeve is 75% of portfolio
) -> Dict[str, dict]:
    """
    Compare risk parity weights to fixed target weights.

    Shows where the current targets over/under-allocate relative to
    what risk parity would suggest.

    Parameters
    ----------
    rp_weights     : {ticker: fraction} from inverse_vol_weights (sum to 1.0)
    target_weights : {ticker: fraction_of_total_capital} from config
    scale_to_etf   : Scale factor (RP weights sum to 1, ETF sleeve uses 0.75 of capital)
    """
    comparison = {}
    for ticker in target_weights:
        target = target_weights.get(ticker, 0)
        rp     = rp_weights.get(ticker, 0) * scale_to_etf
        diff   = target - rp
        comparison[ticker] = {
            "target_weight": round(target * 100, 1),
            "rp_weight":     round(rp * 100, 1),
            "difference":    round(diff * 100, 1),
            "note":          "overweight vs RP" if diff > 0.01 else ("underweight vs RP" if diff < -0.01 else "in line"),
        }
    return comparison


def format_rp_report(comparison: Dict[str, dict]) -> str:
    """ASCII-safe risk parity comparison report."""
    lines = [
        "=" * 60,
        "RISK PARITY vs TARGET WEIGHT COMPARISON",
        "(RP = Inverse Vol, scaled to 75% sleeve allocation)",
        "=" * 60,
        "",
        f"{'Ticker':<8} {'Target':>8} {'RP Wt':>8} {'Diff':>8}  Note",
        "-" * 55,
    ]
    for ticker, d in sorted(comparison.items()):
        lines.append(
            f"{ticker:<8} {d['target_weight']:>7.1f}%  {d['rp_weight']:>7.1f}%  "
            f"{d['difference']:>+7.1f}%  {d['note']}"
        )
    lines += ["", "=" * 60]
    return "\n".join(lines)
