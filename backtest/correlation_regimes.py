"""
Correlation regime analysis.

Computes rolling pairwise correlations between ETF holdings and identifies
how correlations shift during stress (BEAR/BEAR_CRISIS) vs. calm (BULL) regimes.

Key insight: diversification often fails exactly when needed most -- correlations
spike toward 1.0 during crises, reducing the benefit of multi-asset portfolios.

Academic basis:
  Longin & Solnik (2001) "Extreme Correlation of International Equity Markets"
  Ang & Bekaert (2002) "International Asset Allocation with Regime Shifts"
  Kritzman et al. (2010) "Skulls, Financial Turbulence, and Risk Management"
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class PairCorrelation:
    """Pairwise correlation stats for two assets."""
    asset_a:          str
    asset_b:          str
    full_corr:        float   # over entire period
    calm_corr:        float   # during BULL/MILD_BULL regimes
    stress_corr:      float   # during BEAR/BEAR_CRISIS regimes
    corr_breakdown:   float   # stress_corr - calm_corr (positive = correlation rises in stress)
    rolling_min:      float   # minimum rolling 63-day correlation
    rolling_max:      float   # maximum rolling 63-day correlation
    rolling_mean:     float   # average rolling correlation


@dataclass
class CorrelationRegimeSummary:
    """Summary stats for the full correlation matrix."""
    avg_full_corr:    float
    avg_calm_corr:    float
    avg_stress_corr:  float
    avg_breakdown:    float   # avg corr increase in stress
    worst_pair:       str     # pair with largest corr breakdown
    best_pair:        str     # pair most uncorrelated in stress
    diversification_calm:   float   # 1 - avg_calm_corr (higher = more diverse)
    diversification_stress: float   # 1 - avg_stress_corr


def compute_rolling_correlations(
    prices: pd.DataFrame,
    window: int = 63,
) -> Dict[Tuple[str, str], pd.Series]:
    """
    Compute rolling pairwise correlations for all asset pairs.

    Parameters
    ----------
    prices : DataFrame with columns = tickers, index = dates
    window : rolling window in trading days (63 = ~1 quarter)

    Returns
    -------
    Dict mapping (asset_a, asset_b) -> rolling correlation Series
    """
    returns = prices.pct_change().dropna()
    tickers = list(returns.columns)
    result: Dict[Tuple[str, str], pd.Series] = {}

    for i in range(len(tickers)):
        for j in range(i + 1, len(tickers)):
            a, b = tickers[i], tickers[j]
            corr = returns[a].rolling(window).corr(returns[b])
            result[(a, b)] = corr

    return result


def compute_regime_correlations(
    prices: pd.DataFrame,
    regime_series: Optional[pd.Series] = None,
    rolling_window: int = 63,
) -> Tuple[List[PairCorrelation], CorrelationRegimeSummary]:
    """
    Compute pairwise correlations split by market regime.

    Parameters
    ----------
    prices         : DataFrame, columns = tickers
    regime_series  : Series with values like 'BULL', 'BEAR', 'BEAR_CRISIS', etc.
                     If None, skips regime breakdown.
    rolling_window : window for rolling stats

    Returns
    -------
    (pair_correlations, summary)
    """
    if prices.empty or len(prices.columns) < 2:
        return [], CorrelationRegimeSummary(0, 0, 0, 0, "", "", 0, 0)

    returns = prices.pct_change().dropna()
    tickers = list(returns.columns)

    CALM_REGIMES   = {"BULL", "MILD_BULL", "SIDEWAYS"}
    STRESS_REGIMES = {"BEAR", "BEAR_CRISIS"}

    # Align regime series to returns index
    if regime_series is not None:
        reg = regime_series.reindex(returns.index).ffill().bfill()
        calm_mask   = reg.isin(CALM_REGIMES)
        stress_mask = reg.isin(STRESS_REGIMES)
    else:
        calm_mask   = pd.Series(True, index=returns.index)
        stress_mask = pd.Series(False, index=returns.index)

    rolling_corrs = compute_rolling_correlations(prices, window=rolling_window)

    pairs: List[PairCorrelation] = []

    for i in range(len(tickers)):
        for j in range(i + 1, len(tickers)):
            a, b = tickers[i], tickers[j]
            r_a = returns[a]
            r_b = returns[b]

            full_corr = float(r_a.corr(r_b)) if len(r_a) > 10 else 0.0

            calm_corr = 0.0
            stress_corr = 0.0

            if calm_mask.sum() > 20:
                calm_corr = float(r_a[calm_mask].corr(r_b[calm_mask]))
            if stress_mask.sum() > 20:
                stress_corr = float(r_a[stress_mask].corr(r_b[stress_mask]))

            roll_series = rolling_corrs.get((a, b), pd.Series(dtype=float)).dropna()
            roll_min  = float(roll_series.min()) if len(roll_series) > 0 else float('nan')
            roll_max  = float(roll_series.max()) if len(roll_series) > 0 else float('nan')
            roll_mean = float(roll_series.mean()) if len(roll_series) > 0 else float('nan')

            pairs.append(PairCorrelation(
                asset_a=a,
                asset_b=b,
                full_corr=round(full_corr, 3),
                calm_corr=round(calm_corr, 3),
                stress_corr=round(stress_corr, 3),
                corr_breakdown=round(stress_corr - calm_corr, 3),
                rolling_min=round(roll_min, 3) if not np.isnan(roll_min) else 0.0,
                rolling_max=round(roll_max, 3) if not np.isnan(roll_max) else 0.0,
                rolling_mean=round(roll_mean, 3) if not np.isnan(roll_mean) else 0.0,
            ))

    if not pairs:
        return [], CorrelationRegimeSummary(0, 0, 0, 0, "", "", 0, 0)

    avg_full   = float(np.mean([p.full_corr   for p in pairs]))
    avg_calm   = float(np.mean([p.calm_corr   for p in pairs]))
    avg_stress = float(np.mean([p.stress_corr for p in pairs]))
    avg_bkd    = float(np.mean([p.corr_breakdown for p in pairs]))

    worst = max(pairs, key=lambda p: p.corr_breakdown)
    best  = min(pairs, key=lambda p: p.stress_corr)

    summary = CorrelationRegimeSummary(
        avg_full_corr=round(avg_full, 3),
        avg_calm_corr=round(avg_calm, 3),
        avg_stress_corr=round(avg_stress, 3),
        avg_breakdown=round(avg_bkd, 3),
        worst_pair=f"{worst.asset_a}/{worst.asset_b}",
        best_pair=f"{best.asset_a}/{best.asset_b}",
        diversification_calm=round(1.0 - avg_calm, 3),
        diversification_stress=round(1.0 - avg_stress, 3),
    )

    return sorted(pairs, key=lambda p: p.full_corr, reverse=True), summary


def format_correlation_regime_report(
    pairs: List[PairCorrelation],
    summary: CorrelationRegimeSummary,
) -> str:
    """Format correlation regime analysis as ASCII table."""
    if not pairs:
        return "Correlation regime analysis unavailable."

    lines = [
        "=" * 80,
        "CORRELATION REGIME ANALYSIS  (Longin-Solnik 2001, Ang-Bekaert 2002)",
        "(Does diversification hold during market stress?)",
        "=" * 80,
        f"{'Pair':<16} {'Full':>7} {'Calm':>7} {'Stress':>8} {'Breakdown':>10} {'Roll Min':>9} {'Roll Max':>9}",
        "-" * 70,
    ]

    for p in pairs:
        breakdown_flag = " (!)" if p.corr_breakdown > 0.15 else ""
        lines.append(
            f"{p.asset_a+'/'+p.asset_b:<16} "
            f"{p.full_corr:>7.3f} "
            f"{p.calm_corr:>7.3f} "
            f"{p.stress_corr:>8.3f} "
            f"{p.corr_breakdown:>+9.3f}{breakdown_flag:<4}"
            f"{p.rolling_min:>9.3f} "
            f"{p.rolling_max:>9.3f}"
        )

    lines += [
        "",
        "SUMMARY:",
        f"  Avg full-period correlation:  {summary.avg_full_corr:.3f}",
        f"  Avg calm-regime correlation:  {summary.avg_calm_corr:.3f}  "
        f"  (diversification = {summary.diversification_calm:.3f})",
        f"  Avg stress-regime correlation:{summary.avg_stress_corr:.3f}  "
        f"  (diversification = {summary.diversification_stress:.3f})",
        f"  Avg correlation breakdown:   {summary.avg_breakdown:+.3f}  "
        f"(positive = correlations rise in crisis)",
        f"  Worst pair (most breakdown):  {summary.worst_pair}",
        f"  Best pair  (lowest stress):   {summary.best_pair}",
        "",
        "Columns: Full=full-period | Calm=BULL/MILD_BULL/SIDEWAYS | Stress=BEAR/BEAR_CRISIS",
        "Breakdown = Stress - Calm (positive means diversification erodes in crisis)",
        "! = correlation spikes >0.15 in stress (diversification breakdown warning)",
        "=" * 80,
    ]

    return "\n".join(lines)
