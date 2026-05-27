"""
Markov Regime Transition Analysis.

Computes historical transition probabilities between the 5 market regimes
and estimates expected dwell time in each regime.

This is used to:
1. Assess how likely the current BULL regime is to persist
2. Inform PEAD and M&A position sizing based on regime stability
3. Report transition probabilities in the PDF

Methodology: Empirical first-order Markov chain estimated from the
backtest regime sequence. Not a hidden Markov model -- regimes are
directly observed from market indicators.

Academic basis: Hamilton (1989) "A New Approach to the Economic Analysis
of Nonstationary Time Series and the Business Cycle"
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

REGIMES = ["BULL", "MILD_BULL", "SIDEWAYS", "BEAR", "BEAR_CRISIS"]


def compute_transition_matrix(
    regime_sequence: List[str],
) -> Dict[str, Dict[str, float]]:
    """
    Estimate first-order Markov transition probabilities.

    P(next_regime = j | current_regime = i) is estimated by counting
    observed transitions and normalizing.

    Parameters
    ----------
    regime_sequence : Ordered list of regime labels (monthly frequency)

    Returns
    -------
    dict : {from_regime: {to_regime: probability}}
            Rows sum to 1.0 (or 0 if regime never observed)
    """
    if len(regime_sequence) < 2:
        return {r: {r2: 0.0 for r2 in REGIMES} for r in REGIMES}

    # Count transitions
    counts: Dict[str, Dict[str, int]] = {r: {r2: 0 for r2 in REGIMES} for r in REGIMES}
    for i in range(len(regime_sequence) - 1):
        fr = regime_sequence[i]
        to = regime_sequence[i + 1]
        if fr in counts and to in counts[fr]:
            counts[fr][to] += 1

    # Normalize to probabilities
    probs: Dict[str, Dict[str, float]] = {}
    for regime, row in counts.items():
        total = sum(row.values())
        if total > 0:
            probs[regime] = {r2: round(v / total, 3) for r2, v in row.items()}
        else:
            probs[regime] = {r2: 0.0 for r2 in REGIMES}

    return probs


def expected_dwell_time(
    transition_matrix: Dict[str, Dict[str, float]],
) -> Dict[str, float]:
    """
    Compute expected dwell time in each regime.

    For a Markov chain, E[dwell] = 1 / (1 - P_ii)
    where P_ii is the self-transition probability.

    Returns periods (months if transition matrix is monthly).
    """
    result = {}
    for regime in REGIMES:
        p_stay = transition_matrix.get(regime, {}).get(regime, 0)
        if p_stay < 1.0:
            result[regime] = round(1.0 / (1.0 - p_stay), 1)
        else:
            result[regime] = float("inf")
    return result


def stationary_distribution(
    transition_matrix: Dict[str, Dict[str, float]],
) -> Dict[str, float]:
    """
    Compute the long-run stationary distribution of the Markov chain.

    The stationary distribution pi satisfies: pi = pi @ P

    Solved via eigenvector decomposition of the transition matrix.

    Returns
    -------
    dict : {regime: long_run_probability}
    """
    # Build matrix in REGIMES order
    P = np.zeros((len(REGIMES), len(REGIMES)))
    for i, fr in enumerate(REGIMES):
        for j, to in enumerate(REGIMES):
            P[i, j] = transition_matrix.get(fr, {}).get(to, 0.0)

    # Ensure row sums are 1 (add self-transition for unobserved regimes)
    for i in range(len(REGIMES)):
        if P[i].sum() == 0:
            P[i, i] = 1.0

    # Find left eigenvector for eigenvalue 1
    eigvals, eigvecs = np.linalg.eig(P.T)
    idx = np.argmin(np.abs(eigvals - 1.0))
    pi  = np.real(eigvecs[:, idx])
    pi  = pi / pi.sum()  # normalize

    return {r: round(float(pi[i]), 4) for i, r in enumerate(REGIMES)}


def regime_persistence_score(
    current_regime: str,
    transition_matrix: Dict[str, Dict[str, float]],
) -> float:
    """
    Return the probability that the current regime persists next period.

    Higher = more stable/predictable market environment.
    """
    return transition_matrix.get(current_regime, {}).get(current_regime, 0.0)


def format_transition_report(
    regime_sequence: List[str],
    current_regime: str,
) -> str:
    """Format Markov transition analysis as ASCII report."""
    tm    = compute_transition_matrix(regime_sequence)
    dwell = expected_dwell_time(tm)
    stat  = stationary_distribution(tm)
    persist = regime_persistence_score(current_regime, tm)

    lines = [
        "=" * 72,
        "REGIME TRANSITION ANALYSIS (Empirical Markov Chain)",
        "=" * 72,
        "",
        "Transition Matrix (rows=from, cols=to):".ljust(72),
        "         " + "  ".join(f"{r[:5]:>7}" for r in REGIMES),
        "-" * 72,
    ]

    for fr in REGIMES:
        row = " ".join(f"{tm.get(fr, {}).get(to, 0):>7.3f}" for to in REGIMES)
        dw  = dwell.get(fr, 0)
        dw_str = f"{dw:.1f}" if dw != float("inf") else "inf"
        lines.append(f"{fr:<9}  {row}   dwell={dw_str}mo")

    lines += [
        "",
        "Stationary Distribution (long-run regime probabilities):",
        " ".join(f"  {r[:5]}={v:.1%}" for r, v in stat.items()),
        "",
        f"Current Regime: {current_regime}",
        f"Persistence Probability: {persist:.1%} (chance regime persists next month)",
        "",
        "=" * 72,
    ]
    return "\n".join(lines)
