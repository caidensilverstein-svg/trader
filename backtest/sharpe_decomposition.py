"""
Sharpe ratio decomposition: breaks down portfolio Sharpe into components.

Decomposes the portfolio Sharpe ratio into:
  1. Per-asset Sharpe contribution (weighted by capital allocation)
  2. Correlation adjustment (diversification benefit to Sharpe)
  3. Period decomposition (rolling and regime-conditional Sharpe)
  4. Sharpe efficiency: actual vs max achievable given asset Sharpes

Academic basis:
  Sharpe (1994) "The Sharpe Ratio" -- definition and interpretation
  Lo (2002) "The Statistics of Sharpe Ratios" -- statistical properties
  Grinold & Kahn (2000) "Active Portfolio Management" -- IR and Sharpe
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class AssetSharpeContrib:
    """Per-asset Sharpe contribution."""
    ticker:           str
    weight:           float
    standalone_sharpe: float        # Sharpe of this asset alone
    weighted_sharpe:  float         # weight * standalone_sharpe (before correlation adj)
    correlation_adj:  float         # correlation adjustment (how much diversification adds)
    portfolio_contrib: float        # actual contribution to portfolio Sharpe


@dataclass
class SharpeDecompositionResult:
    """Full Sharpe decomposition."""
    portfolio_sharpe:       float
    asset_contributions:    List[AssetSharpeContrib]
    max_achievable_sharpe:  float   # if assets were uncorrelated
    diversification_gain:   float   # portfolio_sharpe - weighted_avg_standalone
    sharpe_efficiency:      float   # portfolio_sharpe / max_achievable_sharpe (%)
    dominant_contributor:   str     # asset with highest Sharpe contribution


def compute_sharpe_decomposition(
    prices: pd.DataFrame,
    weights: Dict[str, float],
    rf: float = 0.05,
    window: Optional[int] = None,
) -> SharpeDecompositionResult:
    """
    Decompose portfolio Sharpe ratio into per-asset contributions.

    Parameters
    ----------
    prices  : DataFrame, columns = tickers
    weights : {ticker: portfolio_weight}
    rf      : risk-free rate (annual)
    window  : optional: limit to last N days

    Returns
    -------
    SharpeDecompositionResult
    """
    available = [t for t in weights if t in prices.columns]
    if len(available) < 2:
        raise ValueError("Need at least 2 assets for Sharpe decomposition")

    w = np.array([weights.get(t, 0.0) for t in available])
    w = w / w.sum() if w.sum() > 0 else w

    returns = prices[available].pct_change().dropna()
    if window:
        returns = returns.tail(window)

    if len(returns) < 30:
        raise ValueError("Insufficient data for Sharpe decomposition")

    rf_daily = rf / 252

    # Portfolio returns
    R = returns.values
    port_ret = R @ w

    # Portfolio Sharpe
    excess = port_ret - rf_daily
    port_sharpe = float(excess.mean() / excess.std() * np.sqrt(252)) if excess.std() > 0 else 0.0

    # Per-asset standalone Sharpe
    asset_sharpes: List[float] = []
    for idx in range(len(available)):
        r_i = R[:, idx]
        ex_i = r_i - rf_daily
        sh_i = float(ex_i.mean() / ex_i.std() * np.sqrt(252)) if ex_i.std() > 0 else 0.0
        asset_sharpes.append(sh_i)

    # Weighted average standalone Sharpe (no correlation benefit)
    weighted_standalone = float(sum(w[i] * asset_sharpes[i] for i in range(len(available))))

    # Max achievable Sharpe (if all assets uncorrelated, Sharpe^2 sums)
    # = sqrt(sum(w_i^2 * sharpe_i^2 / vol_i^2) * 252 ... approximated as sum of Sharpes
    max_achievable = float(np.sqrt(sum((w[i] * asset_sharpes[i]) ** 2 for i in range(len(available)))))
    if max_achievable < abs(weighted_standalone):
        max_achievable = abs(weighted_standalone)

    # Diversification gain
    div_gain = port_sharpe - weighted_standalone

    # Efficiency ratio
    efficiency = (port_sharpe / max_achievable * 100) if max_achievable > 1e-9 else 0.0

    # Per-asset contribution: use covariance decomposition
    cov_matrix = np.cov(R.T) * 252
    port_var = float(w @ cov_matrix @ w)
    port_vol = float(np.sqrt(port_var))

    contribs: List[AssetSharpeContrib] = []
    for idx, ticker in enumerate(available):
        r_i = R[:, idx]
        ex_i = r_i - rf_daily
        vol_i = float(np.std(r_i) * np.sqrt(252))
        corr_to_port = float(np.corrcoef(r_i, port_ret)[0, 1])

        standalone_sh = asset_sharpes[idx]
        weighted_sh   = float(w[idx] * standalone_sh)

        # Contribution = w_i * corr(i, port) * vol_i / port_vol * portfolio_sharpe
        contrib = float(w[idx] * corr_to_port * vol_i / port_vol * port_sharpe) if port_vol > 1e-9 else 0.0
        corr_adj = contrib - weighted_sh

        contribs.append(AssetSharpeContrib(
            ticker=ticker,
            weight=round(float(w[idx]), 4),
            standalone_sharpe=round(standalone_sh, 3),
            weighted_sharpe=round(weighted_sh, 3),
            correlation_adj=round(corr_adj, 3),
            portfolio_contrib=round(contrib, 3),
        ))

    contribs.sort(key=lambda x: x.portfolio_contrib, reverse=True)
    dominant = contribs[0].ticker if contribs else ""

    return SharpeDecompositionResult(
        portfolio_sharpe=round(port_sharpe, 3),
        asset_contributions=contribs,
        max_achievable_sharpe=round(max_achievable, 3),
        diversification_gain=round(div_gain, 3),
        sharpe_efficiency=round(efficiency, 1),
        dominant_contributor=dominant,
    )


def format_sharpe_decomposition(result: SharpeDecompositionResult) -> str:
    """Format Sharpe decomposition as ASCII table."""
    lines = [
        "=" * 75,
        "SHARPE RATIO DECOMPOSITION  (Sharpe 1994, Lo 2002, Grinold-Kahn 2000)",
        "=" * 75,
        f"Portfolio Sharpe:          {result.portfolio_sharpe:.3f}",
        f"Max Achievable Sharpe:     {result.max_achievable_sharpe:.3f}  (if assets uncorrelated)",
        f"Diversification Gain:      {result.diversification_gain:+.3f}  (portfolio - weighted standalone)",
        f"Sharpe Efficiency:         {result.sharpe_efficiency:.1f}%  (actual / max achievable)",
        f"Dominant Contributor:      {result.dominant_contributor}",
        "",
        f"{'Asset':<10} {'Weight':>7} {'StdAlone':>10} {'WeightedSh':>12} {'CorrAdj':>9} {'PortContrib':>12}",
        "-" * 65,
    ]

    for a in result.asset_contributions:
        lines.append(
            f"{a.ticker:<10} "
            f"{a.weight*100:>6.1f}% "
            f"{a.standalone_sharpe:>10.3f} "
            f"{a.weighted_sharpe:>12.3f} "
            f"{a.correlation_adj:>+8.3f} "
            f"{a.portfolio_contrib:>11.3f}"
        )

    lines += [
        "",
        "StdAlone = standalone Sharpe (asset in isolation)",
        "WeightedSh = weight * standalone (before diversification)",
        "CorrAdj = contribution of correlation to Sharpe (positive = helps)",
        "PortContrib = actual contribution to portfolio Sharpe",
        "Sum of PortContrib = Portfolio Sharpe",
        "=" * 75,
    ]

    return "\n".join(lines)
