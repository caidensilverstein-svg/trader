"""
Risk budget allocation across portfolio sleeves.

A risk budget explicitly allocates how much of total portfolio risk (measured
by volatility or CVaR) each sleeve is permitted to contribute. This ensures
that no single sleeve dominates the portfolio's risk profile.

Method: Euler decomposition (also called marginal risk contribution).
  Risk contribution_i = w_i * (Sigma @ w)_i / (w.T @ Sigma @ w)^0.5

Where the risk contributions sum to total portfolio volatility.

Academic basis:
  Maillard, Roncalli & Teiletche (2010) "On the Properties of Equally-Weighted
    Risk Contributions Portfolios" -- risk parity theory
  Roncalli (2013) "Introduction to Risk Parity and Budgeting"
  Qian (2005) "Risk Parity Portfolios: Efficient Portfolios Through True Diversification"
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# Risk budget targets (% of total portfolio risk allocated to each sleeve)
# These reflect our intended risk allocation philosophy:
#   - ETF sleeve bears majority of risk (factors provide return)
#   - Managed futures provide diversification / crisis hedge
#   - Options / PEAD / M&A contribute less risk
DEFAULT_RISK_BUDGET: Dict[str, float] = {
    "ETF_AVUV":     0.25,   # 25% of total risk
    "ETF_AVDV":     0.25,
    "ETF_QMOM":     0.15,
    "ETF_MF":       0.10,   # managed futures (DBMF + CTA combined)
    "PEAD":         0.10,
    "MA_ARB":       0.05,
    "IRON_CONDOR":  0.05,
    "CASH_BUFFER":  0.05,
}


@dataclass
class SliceRiskBudget:
    """Risk contribution for one portfolio slice."""
    sleeve:             str
    target_allocation:  float   # % of portfolio capital
    target_risk_budget: float   # % of total risk we want this sleeve to bear
    actual_risk_contrib: float  # % of total risk actually contributed
    risk_deviation:     float   # actual - target (positive = overshoot)
    volatility:         float   # standalone annualized volatility (if available)
    in_budget:          bool    # True if actual_risk_contrib is within tolerance of target


@dataclass
class RiskBudgetSummary:
    """Portfolio risk budget summary."""
    total_portfolio_vol: float       # annualized portfolio volatility
    concentration_index: float       # max single-sleeve risk contribution (%)
    risk_parity_distance: float      # distance from equal risk contribution (lower = more parity)
    budget_violations:   int         # number of sleeves deviating >5% from target
    effective_n:         float       # effective number of independent bets (Qian 2005)


def compute_euler_risk_contributions(
    weights: np.ndarray,
    cov_matrix: np.ndarray,
) -> np.ndarray:
    """
    Euler decomposition: risk contribution of each asset to portfolio vol.

    Returns array of risk contributions (as fractions of total portfolio vol).
    Each element = w_i * (Cov @ w)_i / portfolio_vol
    """
    port_vol = float(np.sqrt(weights @ cov_matrix @ weights))
    if port_vol < 1e-10:
        return np.zeros(len(weights))

    marginal_contrib = cov_matrix @ weights
    risk_contrib = weights * marginal_contrib / port_vol
    return risk_contrib


def compute_risk_budget(
    prices: pd.DataFrame,
    capital_weights: Dict[str, float],
    risk_budget_targets: Optional[Dict[str, float]] = None,
    tolerance: float = 0.05,
) -> Tuple[List[SliceRiskBudget], RiskBudgetSummary]:
    """
    Compute risk contributions and compare to targets.

    Parameters
    ----------
    prices          : DataFrame, columns = tickers (subset of capital_weights keys)
    capital_weights : {ticker: capital_weight_fraction} summing to 1
    risk_budget_targets : {ticker: target_risk_fraction} summing to 1 (optional)
    tolerance       : allowable deviation from risk budget target (as fraction)

    Returns
    -------
    (sleeve_budgets, summary)
    """
    if prices.empty or not capital_weights:
        return [], RiskBudgetSummary(0, 0, 0, 0, 0)

    available = [t for t in capital_weights if t in prices.columns]
    if len(available) < 2:
        return [], RiskBudgetSummary(0, 0, 0, 0, 0)

    w = np.array([capital_weights.get(t, 0.0) for t in available])
    total_w = w.sum()
    if total_w > 0:
        w = w / total_w

    returns = prices[available].pct_change().dropna()
    cov = returns.cov().values * 252  # annualized

    port_vol = float(np.sqrt(w @ cov @ w))
    euler_rc = compute_euler_risk_contributions(w, cov)

    # Total portfolio vol as fraction base
    total_rc = euler_rc.sum()
    pct_rc = euler_rc / total_rc if total_rc > 1e-10 else np.zeros(len(w))

    budgets: List[SliceRiskBudget] = []
    budget_violations = 0

    for idx, ticker in enumerate(available):
        cap_weight = float(w[idx])
        actual_rc  = float(pct_rc[idx])
        target_rc  = (risk_budget_targets or {}).get(ticker, 1 / len(available))
        deviation  = actual_rc - target_rc
        in_budget  = abs(deviation) <= tolerance

        if not in_budget:
            budget_violations += 1

        # Standalone vol from diagonal of cov matrix
        standalone_vol = float(np.sqrt(cov[idx, idx]))

        budgets.append(SliceRiskBudget(
            sleeve=ticker,
            target_allocation=round(cap_weight * 100, 1),
            target_risk_budget=round(target_rc * 100, 1),
            actual_risk_contrib=round(actual_rc * 100, 1),
            risk_deviation=round(deviation * 100, 1),
            volatility=round(standalone_vol * 100, 2),
            in_budget=in_budget,
        ))

    # Sort by actual_risk_contrib descending
    budgets.sort(key=lambda x: x.actual_risk_contrib, reverse=True)

    # Concentration index
    concentration = max(b.actual_risk_contrib for b in budgets) if budgets else 0

    # Risk parity distance: distance from equal contributions
    eq_target = 100 / len(budgets) if budgets else 0
    rp_dist = float(np.sqrt(np.mean([(b.actual_risk_contrib - eq_target) ** 2 for b in budgets])))

    # Effective number of bets (Qian 2005)
    # EN = 1 / sum(rc_i^2) where rc_i are normalized risk contributions
    rc_normalized = pct_rc / pct_rc.sum() if pct_rc.sum() > 0 else pct_rc
    effective_n = float(1 / np.sum(rc_normalized ** 2)) if np.sum(rc_normalized ** 2) > 0 else 0

    summary = RiskBudgetSummary(
        total_portfolio_vol=round(port_vol * 100, 2),
        concentration_index=round(concentration, 1),
        risk_parity_distance=round(rp_dist, 2),
        budget_violations=budget_violations,
        effective_n=round(effective_n, 2),
    )

    return budgets, summary


def format_risk_budget_report(
    budgets: List[SliceRiskBudget],
    summary: RiskBudgetSummary,
) -> str:
    """Format risk budget allocation as ASCII table."""
    if not budgets:
        return "Risk budget analysis unavailable."

    lines = [
        "=" * 80,
        "RISK BUDGET ALLOCATION  (Maillard-Roncalli-Teiletche 2010, Qian 2005)",
        "(How much of total portfolio risk does each sleeve contribute?)",
        "=" * 80,
        f"Total Portfolio Volatility: {summary.total_portfolio_vol:.2f}% annualized",
        f"Effective Number of Bets:   {summary.effective_n:.2f}  (Qian 2005, max = {len(budgets)})",
        f"Concentration Index:        {summary.concentration_index:.1f}%  (max single-sleeve risk)",
        f"Risk Parity Distance:       {summary.risk_parity_distance:.2f}%  (lower = more balanced)",
        f"Budget Violations:          {summary.budget_violations}  sleeves exceed +/-5% target",
        "",
        f"{'Sleeve':<12} {'Cap%':>6} {'Target%':>8} {'Actual%':>8} {'Dev%':>7} {'Vol':>7} {'OK?':>5}",
        "-" * 58,
    ]

    for b in budgets:
        ok = "OK" if b.in_budget else "OVER" if b.risk_deviation > 0 else "UNDR"
        lines.append(
            f"{b.sleeve:<12} "
            f"{b.target_allocation:>5.1f}% "
            f"{b.target_risk_budget:>7.1f}% "
            f"{b.actual_risk_contrib:>7.1f}% "
            f"{b.risk_deviation:>+6.1f}% "
            f"{b.volatility:>6.2f}% "
            f"{ok:>5}"
        )

    lines += [
        "",
        "Cap% = capital allocation | Target% = risk budget target",
        "Actual% = Euler risk contribution | Dev% = actual - target",
        "Vol = standalone annualized volatility",
        "",
        "Risk parity portfolio would have equal Actual% for all sleeves.",
        "OVER = contributes more risk than budget; UNDR = contributes less.",
        "=" * 80,
    ]

    return "\n".join(lines)
