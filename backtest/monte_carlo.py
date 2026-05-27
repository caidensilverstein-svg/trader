"""
Monte Carlo simulation for portfolio forward projection.

Simulates future portfolio paths by resampling historical daily returns
with replacement. Produces a distribution of outcomes at key horizons:
1 year, 3 years, 5 years.

This is used for:
1. Investor communication: "what's the realistic range of outcomes?"
2. Drawdown risk assessment at longer horizons
3. Probability of achieving specific return targets

Methodology: Block bootstrap resampling (5-day blocks) preserves short-term
serial correlation while generating independent long-run paths.

Academic basis: Efron & Tibshirani (1993) "An Introduction to the Bootstrap"
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _block_resample_returns(
    returns: np.ndarray,
    n_days: int,
    block_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Resample returns using fixed-length block bootstrap."""
    n_orig = len(returns)
    result = []
    while len(result) < n_days:
        start = rng.integers(0, n_orig)
        block = [returns[(start + i) % n_orig] for i in range(block_size)]
        result.extend(block)
    return np.array(result[:n_days])


def run_monte_carlo(
    equity_curve: pd.Series,
    n_simulations: int = 1000,
    horizons: Tuple[int, ...] = (252, 756, 1260),  # 1y, 3y, 5y in trading days
    initial_value: float = 100_000.0,
    block_size: int = 5,
    seed: int = 42,
) -> Dict[str, Dict[str, float]]:
    """
    Run Monte Carlo simulation from historical equity curve.

    Parameters
    ----------
    equity_curve   : Historical daily portfolio values
    n_simulations  : Number of paths to simulate (1000 = good balance)
    horizons       : Simulation endpoints in trading days (252=1yr, 1260=5yr)
    initial_value  : Starting portfolio value
    block_size     : Bootstrap block size (5 days preserves weekly structure)
    seed           : Random seed for reproducibility

    Returns
    -------
    dict : {horizon_label: {percentile_label: final_portfolio_value}}
    """
    returns = equity_curve.pct_change().dropna().values
    if len(returns) < 50:
        logger.warning("Insufficient history for Monte Carlo (%d obs)", len(returns))
        return {}

    rng     = np.random.default_rng(seed)
    results = {}

    for n_days in horizons:
        years   = n_days / 252
        label   = f"{years:.0f}yr" if years == int(years) else f"{years:.1f}yr"
        finals  = []

        for _ in range(n_simulations):
            sim_rets = _block_resample_returns(returns, n_days, block_size, rng)
            final    = float(initial_value * np.prod(1 + sim_rets))
            finals.append(final)

        finals = np.array(finals)

        results[label] = {
            "p05": round(float(np.percentile(finals, 5)), 0),
            "p25": round(float(np.percentile(finals, 25)), 0),
            "p50": round(float(np.percentile(finals, 50)), 0),
            "p75": round(float(np.percentile(finals, 75)), 0),
            "p95": round(float(np.percentile(finals, 95)), 0),
            "mean": round(float(np.mean(finals)), 0),
            "prob_loss": round(float(np.mean(finals < initial_value)) * 100, 1),
            "prob_2x":   round(float(np.mean(finals > initial_value * 2)) * 100, 1),
            "n_sims":    n_simulations,
        }

        logger.info(
            "MC %s: p05=$%,.0f  median=$%,.0f  p95=$%,.0f  P(loss)=%.1f%%",
            label, results[label]["p05"], results[label]["p50"],
            results[label]["p95"], results[label]["prob_loss"],
        )

    return results


def format_mc_report(
    mc_results: Dict[str, Dict[str, float]],
    initial_value: float = 100_000.0,
) -> str:
    """Format Monte Carlo results as ASCII table."""
    lines = [
        "=" * 75,
        f"MONTE CARLO PROJECTION ({mc_results[list(mc_results.keys())[0]].get('n_sims', 0):,} simulations)",
        f"Starting Value: ${initial_value:,.0f}  |  Block Bootstrap (5-day blocks)",
        "=" * 75,
        f"{'Horizon':<8} {'5th %ile':>10} {'25th':>10} {'Median':>10} {'75th':>10} {'95th':>10} {'P(Loss)':>8}",
        "-" * 70,
    ]
    for label, d in mc_results.items():
        lines.append(
            f"{label:<8} ${d['p05']:>8,.0f} ${d['p25']:>8,.0f} ${d['p50']:>8,.0f} "
            f"${d['p75']:>8,.0f} ${d['p95']:>8,.0f} {d['prob_loss']:>7.1f}%"
        )
    lines += [
        "",
        f"P(2x) by 5yr: {mc_results.get('5yr', {}).get('prob_2x', 0):.1f}%",
        "Note: Past performance not indicative of future results.",
        "=" * 75,
    ]
    return "\n".join(lines)
