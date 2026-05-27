"""
Tail risk decomposition (Component CVaR / Expected Shortfall).

Decomposes portfolio CVaR into per-asset contributions.
A position with a large CVaR contribution is the primary driver of
tail losses and should be reduced or hedged if tail risk is elevated.

Methodology:
  Component CVaR = weight_i * marginal_CVaR_i
  Marginal CVaR  = average portfolio loss conditional on loss exceeding VaR,
                   dotted with the asset's return in those scenarios.

Academic basis:
  Rockafellar & Uryasev (2002) "Conditional Value-at-Risk for General Loss Distributions"
  Garlappi & Skoulakis (2009) "Portfolio Choice with Skewness Preference and Aversion"
  McNeil, Frey & Embrechts (2005) "Quantitative Risk Management"
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class AssetTailRisk:
    """Tail risk contribution for a single asset."""
    ticker:             str
    weight:             float       # portfolio weight
    standalone_var95:   float       # VaR at 95%, as fraction of position value
    standalone_cvar95:  float       # CVaR (ES) at 95%
    component_cvar95:   float       # contribution to portfolio CVaR (diversified)
    marginal_cvar95:    float       # marginal contribution per unit weight
    pct_of_total_cvar:  float       # share of total portfolio CVaR
    tail_beta:          float       # correlation to portfolio in tail scenarios
    worst_day:          float       # single worst daily return in history


@dataclass
class TailRiskSummary:
    """Portfolio-level tail risk summary."""
    portfolio_var95:        float   # portfolio 1-day VaR at 95%
    portfolio_cvar95:       float   # portfolio 1-day CVaR at 95%
    portfolio_var99:        float   # portfolio 1-day VaR at 99%
    portfolio_cvar99:       float   # portfolio 1-day CVaR at 99%
    diversification_benefit: float  # sum(standalone) - portfolio (diversification reduces CVaR)
    dominant_risk_asset:    str     # ticker with highest CVaR contribution
    dominant_risk_pct:      float   # that asset's % of total CVaR
    skewness:               float   # portfolio return skewness
    kurtosis:               float   # portfolio return excess kurtosis (fat tails > 0)


def compute_portfolio_tail_risk(
    prices: pd.DataFrame,
    weights: Dict[str, float],
    confidence: float = 0.95,
    window: Optional[int] = None,
) -> Tuple[List[AssetTailRisk], TailRiskSummary]:
    """
    Decompose portfolio tail risk into per-asset CVaR contributions.

    Parameters
    ----------
    prices     : DataFrame with columns = tickers, index = dates
    weights    : {ticker: portfolio_weight} (should sum to ~1)
    confidence : VaR/CVaR confidence level (0.95 or 0.99)
    window     : optional: limit to last N days

    Returns
    -------
    (asset_risks, summary)
    """
    if prices.empty or not weights:
        return [], TailRiskSummary(0, 0, 0, 0, 0, "", 0, 0, 0)

    # Align weights to available tickers
    available = [t for t in weights if t in prices.columns]
    if not available:
        return [], TailRiskSummary(0, 0, 0, 0, 0, "", 0, 0, 0)

    w = np.array([weights.get(t, 0.0) for t in available])
    w = w / w.sum() if w.sum() > 0 else w  # normalize

    returns = prices[available].pct_change().dropna()
    if window:
        returns = returns.tail(window)

    if len(returns) < 30:
        return [], TailRiskSummary(0, 0, 0, 0, 0, "", 0, 0, 0)

    R = returns.values  # shape: (n_days, n_assets)
    port_returns = R @ w  # portfolio daily returns

    # Portfolio VaR / CVaR
    alpha95 = 1 - confidence
    alpha99 = 0.01

    var95 = float(-np.percentile(port_returns, alpha95 * 100))
    var99 = float(-np.percentile(port_returns, alpha99 * 100))

    tail_mask95 = port_returns <= -var95
    tail_mask99 = port_returns <= -var99

    cvar95 = float(-np.mean(port_returns[tail_mask95])) if tail_mask95.sum() > 0 else var95
    cvar99 = float(-np.mean(port_returns[tail_mask99])) if tail_mask99.sum() > 0 else var99

    # Portfolio stats
    port_skew = float(pd.Series(port_returns).skew())
    port_kurt = float(pd.Series(port_returns).kurtosis())  # excess kurtosis

    # Per-asset stats
    asset_risks: List[AssetTailRisk] = []
    total_component_cvar = 0.0

    for idx, ticker in enumerate(available):
        r_i = R[:, idx]
        w_i = float(w[idx])

        # Standalone VaR/CVaR
        sa_var95  = float(-np.percentile(r_i, alpha95 * 100))
        tail_i    = r_i[r_i <= -sa_var95]
        sa_cvar95 = float(-np.mean(tail_i)) if len(tail_i) > 0 else sa_var95

        # Marginal CVaR: asset's average return when portfolio is in tail
        if tail_mask95.sum() > 0:
            marginal = float(np.mean(r_i[tail_mask95]))
        else:
            marginal = 0.0

        component_cvar = w_i * (-marginal)  # contribution to portfolio CVaR
        total_component_cvar += component_cvar

        # Tail beta: correlation of asset to portfolio in tail scenarios
        if tail_mask95.sum() > 5:
            tail_beta = float(np.corrcoef(r_i[tail_mask95], port_returns[tail_mask95])[0, 1])
        else:
            tail_beta = 0.0

        worst_day = float(r_i.min())

        asset_risks.append(AssetTailRisk(
            ticker=ticker,
            weight=round(w_i, 4),
            standalone_var95=round(sa_var95, 4),
            standalone_cvar95=round(sa_cvar95, 4),
            component_cvar95=round(component_cvar, 4),
            marginal_cvar95=round(-marginal, 4) if w_i > 0 else 0.0,
            pct_of_total_cvar=0.0,  # fill after total is known
            tail_beta=round(tail_beta, 3),
            worst_day=round(worst_day, 4),
        ))

    # Fill pct_of_total
    for ar in asset_risks:
        ar.pct_of_total_cvar = round(
            ar.component_cvar95 / total_component_cvar * 100
            if abs(total_component_cvar) > 1e-10 else 0.0,
            1
        )

    # Diversification benefit
    standalone_sum = sum(ar.standalone_cvar95 * ar.weight for ar in asset_risks)
    div_benefit = standalone_sum - cvar95

    # Sort by CVaR contribution descending
    asset_risks.sort(key=lambda x: x.pct_of_total_cvar, reverse=True)
    dominant = asset_risks[0] if asset_risks else None

    summary = TailRiskSummary(
        portfolio_var95=round(var95, 4),
        portfolio_cvar95=round(cvar95, 4),
        portfolio_var99=round(var99, 4),
        portfolio_cvar99=round(cvar99, 4),
        diversification_benefit=round(div_benefit, 4),
        dominant_risk_asset=dominant.ticker if dominant else "",
        dominant_risk_pct=dominant.pct_of_total_cvar if dominant else 0.0,
        skewness=round(port_skew, 3),
        kurtosis=round(port_kurt, 3),
    )

    return asset_risks, summary


def format_tail_risk_report(
    asset_risks: List[AssetTailRisk],
    summary: TailRiskSummary,
) -> str:
    """Format tail risk decomposition as ASCII table."""
    if not asset_risks:
        return "Tail risk decomposition unavailable."

    lines = [
        "=" * 80,
        "TAIL RISK DECOMPOSITION  (Rockafellar-Uryasev 2002, McNeil et al. 2005)",
        "(Component CVaR -- which asset drives portfolio tail losses?)",
        "=" * 80,
        f"Portfolio 1-Day VaR (95%):   {summary.portfolio_var95*100:.3f}%",
        f"Portfolio 1-Day CVaR (95%):  {summary.portfolio_cvar95*100:.3f}%  (Expected Shortfall)",
        f"Portfolio 1-Day VaR (99%):   {summary.portfolio_var99*100:.3f}%",
        f"Portfolio 1-Day CVaR (99%):  {summary.portfolio_cvar99*100:.3f}%",
        f"Diversification Benefit:     {summary.diversification_benefit*100:.3f}%  "
        f"(reduction vs sum of standalone CVaRs)",
        f"Return Skewness:             {summary.skewness:.3f}  "
        f"({'negative = left tail risk' if summary.skewness < 0 else 'positive = right tail'})",
        f"Excess Kurtosis:             {summary.kurtosis:.3f}  "
        f"({'fat tails = > 0' if summary.kurtosis > 0 else 'thin tails'})",
        "",
        f"{'Asset':<8} {'Weight':>7} {'SA-CVaR':>9} {'CompCVaR':>10} {'%Total':>7} {'TailBeta':>9} {'WorstDay':>9}",
        "-" * 65,
    ]

    for ar in asset_risks:
        bar = "#" * int(ar.pct_of_total_cvar / 5)
        lines.append(
            f"{ar.ticker:<8} "
            f"{ar.weight*100:>6.1f}% "
            f"{ar.standalone_cvar95*100:>8.3f}% "
            f"{ar.component_cvar95*100:>9.4f}% "
            f"{ar.pct_of_total_cvar:>6.1f}% "
            f"{ar.tail_beta:>9.3f} "
            f"{ar.worst_day*100:>8.2f}%"
        )

    lines += [
        "",
        f"Dominant tail risk: {summary.dominant_risk_asset} ({summary.dominant_risk_pct:.1f}% of portfolio CVaR)",
        "",
        "SA-CVaR = Standalone Expected Shortfall (ignores diversification)",
        "CompCVaR = Component CVaR (actual contribution to portfolio tail risk)",
        "TailBeta = Correlation to portfolio during tail events (1.0 = perfectly aligned)",
        "=" * 80,
    ]

    return "\n".join(lines)
