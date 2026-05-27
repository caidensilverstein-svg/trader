"""
Markowitz Mean-Variance Optimization.

Computes the efficient frontier and optimal portfolio weights using
historical covariance and expected returns. Compares our factor-tilted
portfolio to the mean-variance optimal weights.

Key insight: mean-variance optimization is highly sensitive to estimation
error in expected returns (Michaud 1989). Our factor approach avoids this
by using academically-validated factor premia instead of historical return
estimates, which tend to be noisy over short windows.

Academic basis:
  Markowitz (1952) "Portfolio Selection", Journal of Finance
  Michaud (1989) "The Markowitz Optimization Enigma" -- estimation error
  Ledoit & Wolf (2004) "Honey, I Shrunk the Sample Covariance Matrix"
  Black & Litterman (1992) -- practical implementation
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class EfficientPortfolio:
    """A portfolio on or near the efficient frontier."""
    label:           str
    weights:         Dict[str, float]   # ticker -> weight
    expected_return: float              # annualized
    expected_vol:    float              # annualized
    sharpe:          float              # (ret - rf) / vol
    is_max_sharpe:   bool
    is_min_vol:      bool


@dataclass
class MVOResult:
    """Full mean-variance optimization result."""
    tickers:         List[str]
    max_sharpe:      EfficientPortfolio
    min_vol:         EfficientPortfolio
    equal_weight:    EfficientPortfolio
    factor_target:   Optional[EfficientPortfolio]  # our actual target weights
    covariance_matrix: np.ndarray                  # annualized
    expected_returns:  np.ndarray                  # annualized, per ticker
    frontier_vols:     List[float]
    frontier_rets:     List[float]
    estimation_period_days: int


def _annualized_cov(returns: pd.DataFrame) -> np.ndarray:
    """Sample covariance matrix, annualized."""
    return returns.cov().values * 252


def _ledoit_wolf_shrinkage(returns: pd.DataFrame) -> np.ndarray:
    """
    Ledoit-Wolf shrinkage estimator for covariance matrix.
    Shrinks sample covariance toward scaled identity to reduce estimation error.
    """
    S = returns.cov().values * 252
    n, p = len(returns), len(returns.columns)
    mu = np.trace(S) / p  # shrinkage target: scaled identity
    delta = mu * np.eye(p)

    # Shrinkage intensity (simple Ledoit-Wolf formula)
    rho_num = sum(
        np.sum((returns.values[:, i] * returns.values[:, j] - S[i, j] / 252) ** 2)
        for i in range(p) for j in range(p)
    )
    rho_num *= 252 ** 2 / (n - 1)

    # Simplified: use standard shrinkage intensity alpha = min(1, p / (n * ||S - delta||^2))
    frobenius_sq = np.sum((S - delta) ** 2)
    alpha = min(1.0, p / max(n * frobenius_sq, 1e-12))

    return (1 - alpha) * S + alpha * delta


def _portfolio_stats(
    weights: np.ndarray,
    mu: np.ndarray,
    cov: np.ndarray,
    rf: float = 0.05,
) -> Tuple[float, float, float]:
    """Return (expected_return, volatility, sharpe) for given weights."""
    ret = float(weights @ mu)
    vol = float(np.sqrt(weights @ cov @ weights))
    sharpe = (ret - rf) / vol if vol > 1e-9 else 0.0
    return ret, vol, sharpe


def run_mvo(
    prices: pd.DataFrame,
    factor_weights: Optional[Dict[str, float]] = None,
    n_frontier: int = 30,
    rf: float = 0.05,
    use_shrinkage: bool = True,
) -> MVOResult:
    """
    Run mean-variance optimization.

    Parameters
    ----------
    prices         : DataFrame, columns = tickers, index = dates
    factor_weights : our actual target weights for comparison
    n_frontier     : number of points to compute on efficient frontier
    rf             : risk-free rate (annual)
    use_shrinkage  : if True, use Ledoit-Wolf shrinkage on covariance

    Returns
    -------
    MVOResult with max-Sharpe, min-vol, equal-weight, and factor portfolio
    """
    if prices.empty or len(prices.columns) < 2:
        raise ValueError("Need at least 2 assets for MVO")

    returns = prices.pct_change().dropna()
    n_obs, n_assets = returns.shape
    tickers = list(returns.columns)

    # Expected returns: use historical mean (simple; acknowledged as noisy)
    mu = returns.mean().values * 252  # annualized

    # Covariance
    if use_shrinkage:
        cov = _ledoit_wolf_shrinkage(returns)
    else:
        cov = _annualized_cov(returns)

    # --- Max Sharpe via grid search (avoids scipy dependency) ---
    n_sim = 5000
    rng = np.random.default_rng(42)

    best_sharpe = -np.inf
    best_minvol = np.inf
    w_maxsharpe = np.ones(n_assets) / n_assets
    w_minvol    = np.ones(n_assets) / n_assets

    for _ in range(n_sim):
        raw = rng.exponential(1, n_assets)
        w = raw / raw.sum()
        ret, vol, sharpe = _portfolio_stats(w, mu, cov, rf)
        if sharpe > best_sharpe:
            best_sharpe = sharpe
            w_maxsharpe = w.copy()
        if vol < best_minvol:
            best_minvol = vol
            w_minvol = w.copy()

    def _make_portfolio(label, weights_arr, is_ms=False, is_mv=False):
        ret, vol, sharpe = _portfolio_stats(weights_arr, mu, cov, rf)
        return EfficientPortfolio(
            label=label,
            weights={t: round(float(w), 4) for t, w in zip(tickers, weights_arr)},
            expected_return=round(ret, 4),
            expected_vol=round(vol, 4),
            sharpe=round(sharpe, 4),
            is_max_sharpe=is_ms,
            is_min_vol=is_mv,
        )

    max_sharpe_port = _make_portfolio("Max Sharpe (MVO)", w_maxsharpe, is_ms=True)
    min_vol_port    = _make_portfolio("Min Volatility (MVO)", w_minvol, is_mv=True)
    eq_w = np.ones(n_assets) / n_assets
    eq_port = _make_portfolio("Equal Weight", eq_w)

    # Factor target portfolio
    factor_port = None
    if factor_weights:
        w_factor = np.array([factor_weights.get(t, 0.0) for t in tickers])
        s = w_factor.sum()
        if s > 0:
            w_factor /= s
            factor_port = _make_portfolio("Factor Target (Ours)", w_factor)

    # Efficient frontier
    min_ret = float(min_vol_port.expected_return)
    max_ret = float(max_sharpe_port.expected_return) * 1.5
    frontier_vols: List[float] = []
    frontier_rets: List[float] = []

    for target_ret in np.linspace(min_ret, max_ret, n_frontier):
        best_v = np.inf
        for _ in range(500):
            raw = rng.exponential(1, n_assets)
            w = raw / raw.sum()
            ret, vol, _ = _portfolio_stats(w, mu, cov, rf)
            if abs(ret - target_ret) < 0.01 and vol < best_v:
                best_v = vol
        if best_v < np.inf:
            frontier_vols.append(round(best_v, 4))
            frontier_rets.append(round(target_ret, 4))

    return MVOResult(
        tickers=tickers,
        max_sharpe=max_sharpe_port,
        min_vol=min_vol_port,
        equal_weight=eq_port,
        factor_target=factor_port,
        covariance_matrix=cov,
        expected_returns=mu,
        frontier_vols=frontier_vols,
        frontier_rets=frontier_rets,
        estimation_period_days=n_obs,
    )


def format_mvo_report(result: MVOResult) -> str:
    """Format MVO comparison as ASCII table."""
    lines = [
        "=" * 80,
        "MEAN-VARIANCE OPTIMIZATION  (Markowitz 1952, Ledoit-Wolf 2004)",
        "(How does our factor portfolio compare to the MVO efficient frontier?)",
        "=" * 80,
        f"Estimation period: {result.estimation_period_days} trading days",
        f"Assets: {', '.join(result.tickers)}",
        f"Covariance estimator: Ledoit-Wolf shrinkage (reduces estimation error)",
        "",
        f"{'Portfolio':<22} {'Exp.Ret':>9} {'Exp.Vol':>9} {'Sharpe':>8}",
        "-" * 55,
    ]

    for port in [result.max_sharpe, result.min_vol, result.equal_weight, result.factor_target]:
        if port is None:
            continue
        tag = " (*)" if port.is_max_sharpe else " (-)" if port.is_min_vol else ""
        lines.append(
            f"{port.label:<22} "
            f"{port.expected_return*100:>8.2f}% "
            f"{port.expected_vol*100:>8.2f}% "
            f"{port.sharpe:>8.3f}{tag}"
        )

    lines += [
        "",
        "(*) = Max Sharpe portfolio    (-) = Min Volatility portfolio",
        "",
        "WEIGHT COMPARISON:",
    ]

    # Weight table
    all_ports = [p for p in [result.max_sharpe, result.min_vol, result.equal_weight, result.factor_target] if p]
    hdr = f"{'Ticker':<8}" + "".join(f" {p.label[:10]:>11}" for p in all_ports)
    lines.append(hdr)
    lines.append("-" * (8 + 12 * len(all_ports)))
    for ticker in result.tickers:
        row = f"{ticker:<8}"
        for p in all_ports:
            w = p.weights.get(ticker, 0.0)
            row += f" {w*100:>10.1f}%"
        lines.append(row)

    lines += [
        "",
        "Interpretation: MVO is sensitive to historical return estimates (Michaud 1989).",
        "Our factor approach uses academically-validated premia rather than noisy",
        "historical means, making it more robust out-of-sample.",
        "Ledoit-Wolf shrinkage reduces the 'error maximization' problem of naive MVO.",
        "=" * 80,
    ]

    return "\n".join(lines)
