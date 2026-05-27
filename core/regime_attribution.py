"""
Regime performance attribution.

For each regime (BULL, MILD_BULL, SIDEWAYS, BEAR, BEAR_CRISIS), computes:
  - Fraction of total time spent in regime
  - Fraction of total return generated in regime
  - Regime-conditional Sharpe ratio
  - Regime-conditional max drawdown
  - Average duration of regime episodes

This answers the key question: "Does this strategy need bull markets to work,
or does it generate alpha across all regimes?"

Academic basis: Hamilton (1989) "A New Approach to the Economic Analysis
of Nonstationary Time Series and the Business Cycle"
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class RegimeStats:
    regime:            str
    n_days:            int
    pct_time:          float   # % of total history
    cumulative_return: float   # % total return in this regime
    ann_return:        float   # annualized return in regime
    sharpe:            float   # annualized Sharpe (no RF)
    max_drawdown:      float   # max drawdown within regime episodes
    n_episodes:        int     # how many times we entered this regime
    avg_duration:      float   # average episode length in days


def compute_regime_attribution(
    equity_curve: pd.Series,
    regime_series: pd.Series,
    regimes: Optional[List[str]] = None,
) -> List[RegimeStats]:
    """
    Compute per-regime performance attribution.

    Parameters
    ----------
    equity_curve  : Daily portfolio values indexed by date
    regime_series : Daily regime labels (same index), e.g. "BULL", "BEAR"
    regimes       : List of regime labels to include (None = auto-detect)

    Returns
    -------
    List of RegimeStats, sorted by cumulative return descending
    """
    if len(equity_curve) < 20 or len(regime_series) < 20:
        return []

    eq = equity_curve.copy()
    rs = regime_series.copy()
    eq.index = pd.to_datetime(eq.index)
    rs.index = pd.to_datetime(rs.index)

    # Align
    combined = pd.DataFrame({"equity": eq, "regime": rs}).dropna()
    if len(combined) < 20:
        return []

    combined["daily_ret"] = combined["equity"].pct_change()
    total_days = len(combined)

    if regimes is None:
        regimes = sorted(combined["regime"].unique())

    results = []
    for regime in regimes:
        mask   = combined["regime"] == regime
        subset = combined[mask]

        if len(subset) < 2:
            continue

        rets = subset["daily_ret"].dropna()
        if len(rets) < 2:
            continue

        # Cumulative return in this regime
        cum_ret = float((1 + rets).prod() - 1) * 100

        # Annualised return
        n_days = len(subset)
        ann_ret = ((1 + cum_ret / 100) ** (252 / n_days) - 1) * 100

        # Annualised Sharpe
        vol = float(rets.std()) * np.sqrt(252)
        ann_mu = float(rets.mean()) * 252
        sharpe = (ann_mu / vol) if vol > 0 else 0.0

        # Max drawdown within regime (across all regime episodes)
        eq_regime = subset["equity"]
        cummax     = eq_regime.cummax()
        dd         = float((eq_regime / cummax - 1.0).min()) * 100

        # Count regime episodes (consecutive runs)
        episodes = 0
        durations = []
        in_episode = False
        run_len = 0
        for flag in mask.values:
            if flag:
                if not in_episode:
                    in_episode = True
                    run_len = 1
                    episodes += 1
                else:
                    run_len += 1
            else:
                if in_episode:
                    durations.append(run_len)
                    in_episode = False
                    run_len = 0
        if in_episode:
            durations.append(run_len)

        avg_dur = float(np.mean(durations)) if durations else float(n_days)

        results.append(RegimeStats(
            regime=regime,
            n_days=n_days,
            pct_time=round(n_days / total_days * 100, 1),
            cumulative_return=round(cum_ret, 2),
            ann_return=round(ann_ret, 2),
            sharpe=round(sharpe, 2),
            max_drawdown=round(dd, 2),
            n_episodes=episodes,
            avg_duration=round(avg_dur, 1),
        ))

    return sorted(results, key=lambda r: r.cumulative_return, reverse=True)


def regime_attribution_summary(stats: List[RegimeStats]) -> Dict[str, float]:
    """Aggregate summary across all regimes."""
    if not stats:
        return {}

    bull_regimes = [s for s in stats if "BULL" in s.regime]
    bear_regimes = [s for s in stats if "BEAR" in s.regime]

    bull_return = sum(s.cumulative_return for s in bull_regimes)
    bear_return = sum(s.cumulative_return for s in bear_regimes)

    best  = max(stats, key=lambda s: s.ann_return)
    worst = min(stats, key=lambda s: s.ann_return)

    return {
        "n_regimes":       len(stats),
        "bull_total_return": round(bull_return, 2),
        "bear_total_return": round(bear_return, 2),
        "best_regime":     best.regime,
        "best_ann_ret":    best.ann_return,
        "worst_regime":    worst.regime,
        "worst_ann_ret":   worst.ann_return,
        "positive_regimes": sum(1 for s in stats if s.ann_return > 0),
    }


def format_regime_attribution(stats: List[RegimeStats]) -> str:
    """Format regime attribution as ASCII table."""
    if not stats:
        return "No regime attribution data available."

    summary = regime_attribution_summary(stats)

    lines = [
        "=" * 85,
        "REGIME PERFORMANCE ATTRIBUTION",
        "(Return decomposition by market regime)",
        "=" * 85,
        f"{'Regime':<14} {'Days':>6} {'%Time':>6} {'CumRet':>8} {'AnnRet':>8} "
        f"{'Sharpe':>7} {'MaxDD':>7} {'Episodes':>9} {'AvgDur':>7}",
        "-" * 78,
    ]
    for s in stats:
        sign = "+" if s.cumulative_return >= 0 else ""
        lines.append(
            f"{s.regime:<14} {s.n_days:>6} {s.pct_time:>5.1f}% "
            f"{sign}{s.cumulative_return:>6.1f}% {s.ann_return:>+7.1f}% "
            f"{s.sharpe:>+7.2f} {s.max_drawdown:>6.1f}% {s.n_episodes:>9} "
            f"{s.avg_duration:>6.1f}d"
        )
    lines += [
        "",
        f"Best regime: {summary.get('best_regime', 'N/A')} "
        f"({summary.get('best_ann_ret', 0):+.1f}% ann)",
        f"Worst regime: {summary.get('worst_regime', 'N/A')} "
        f"({summary.get('worst_ann_ret', 0):+.1f}% ann)",
        f"Positive regimes: {summary.get('positive_regimes', 0)}/{summary.get('n_regimes', 0)}",
        "=" * 85,
    ]
    return "\n".join(lines)
