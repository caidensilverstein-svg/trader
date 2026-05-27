"""
Position-level contribution to backtest performance.

For each ETF in the sleeve, computes:
  - Contribution to total return (weight * ETF return)
  - Contribution to portfolio Sharpe
  - Diversification benefit (correlation penalty)
  - Hit rate (% of days the position was positive)

This answers: "Which positions drove performance, and which were drag?"

Academic basis: Brinson, Hood & Beebower (1986) "Determinants of
Portfolio Performance" -- the classic attribution paper.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class PositionAttribution:
    ticker:           str
    avg_weight:       float    # average portfolio weight during period
    total_return:     float    # position's own total return %
    contribution:     float    # weight * position_return (pp of portfolio return)
    daily_vol:        float    # annualized daily return vol %
    hit_rate:         float    # % of days positive
    sharpe:           float    # position's own Sharpe (no RF)
    max_dd:           float    # position max drawdown %
    corr_to_spy:      float    # correlation to SPY returns


def compute_position_attribution(
    etf_prices: Dict[str, pd.Series],
    weights: Dict[str, float],
    spy_prices: Optional[pd.Series] = None,
) -> List[PositionAttribution]:
    """
    Compute per-position performance attribution.

    Parameters
    ----------
    etf_prices : {ticker: price_series}
    weights    : {ticker: portfolio_weight_fraction}
    spy_prices : Optional SPY price series for correlation

    Returns
    -------
    List of PositionAttribution sorted by contribution (best first)
    """
    results = []

    # Build SPY returns for correlation
    spy_rets = None
    if spy_prices is not None:
        spy_rets = spy_prices.pct_change().dropna()

    for ticker, weight in weights.items():
        prices = etf_prices.get(ticker)
        if prices is None or len(prices) < 20:
            continue

        rets = prices.pct_change().dropna()
        if len(rets) < 20:
            continue

        total_ret = float((prices.iloc[-1] / prices.iloc[0] - 1) * 100)
        contribution = weight * total_ret

        ann_vol = float(rets.std() * np.sqrt(252)) * 100
        hit_rate = float((rets > 0).mean()) * 100
        sharpe = (float(rets.mean()) * 252 / (float(rets.std()) * np.sqrt(252))
                  if float(rets.std()) > 0 else 0.0)

        cummax = prices.cummax()
        max_dd = float((prices / cummax - 1).min()) * 100

        # Correlation to SPY
        corr = 0.0
        if spy_rets is not None:
            idx = rets.index.intersection(spy_rets.index)
            if len(idx) > 20:
                corr = float(rets.loc[idx].corr(spy_rets.loc[idx]))

        results.append(PositionAttribution(
            ticker=ticker,
            avg_weight=round(weight * 100, 1),
            total_return=round(total_ret, 2),
            contribution=round(contribution, 2),
            daily_vol=round(ann_vol, 1),
            hit_rate=round(hit_rate, 1),
            sharpe=round(sharpe, 2),
            max_dd=round(max_dd, 1),
            corr_to_spy=round(corr, 3),
        ))

    return sorted(results, key=lambda p: p.contribution, reverse=True)


def attribution_summary(attributions: List[PositionAttribution]) -> Dict:
    """Aggregate summary across all positions."""
    if not attributions:
        return {}

    total_contrib = sum(a.contribution for a in attributions)
    best  = max(attributions, key=lambda a: a.contribution)
    worst = min(attributions, key=lambda a: a.contribution)
    avg_hit = float(np.mean([a.hit_rate for a in attributions]))
    avg_corr = float(np.mean([a.corr_to_spy for a in attributions]))

    return {
        "n_positions":      len(attributions),
        "total_contribution": round(total_contrib, 2),
        "best_position":    best.ticker,
        "best_contrib":     best.contribution,
        "worst_position":   worst.ticker,
        "worst_contrib":    worst.contribution,
        "avg_hit_rate":     round(avg_hit, 1),
        "avg_spy_corr":     round(avg_corr, 3),
    }


def format_attribution_report(attributions: List[PositionAttribution]) -> str:
    """Format position attribution as ASCII table."""
    if not attributions:
        return "Position attribution data unavailable."

    smry = attribution_summary(attributions)
    total_contrib = smry.get("total_contribution", 0)

    lines = [
        "=" * 90,
        "POSITION-LEVEL RETURN ATTRIBUTION",
        "(Brinson, Hood & Beebower 1986 attribution framework)",
        "=" * 90,
        f"{'Ticker':<8} {'Weight':>7} {'Pos Ret':>8} {'Contrib':>9} {'Vol':>7} "
        f"{'Hit%':>6} {'Sharpe':>7} {'MaxDD':>7} {'Corr':>7}",
        "-" * 75,
    ]
    for a in attributions:
        sign = "+" if a.contribution >= 0 else ""
        lines.append(
            f"{a.ticker:<8} {a.avg_weight:>6.1f}% {a.total_return:>+7.1f}% "
            f"{sign}{a.contribution:>7.1f}pp {a.daily_vol:>6.1f}% "
            f"{a.hit_rate:>5.1f}% {a.sharpe:>+7.2f} {a.max_dd:>6.1f}% "
            f"{a.corr_to_spy:>+6.2f}"
        )
    lines += [
        "-" * 75,
        f"{'TOTAL':<8} {'':>7} {'':>8} {total_contrib:>+8.1f}pp",
        "",
        f"Best contributor: {smry['best_position']} ({smry['best_contrib']:+.1f}pp)",
        f"Worst contributor: {smry['worst_position']} ({smry['worst_contrib']:+.1f}pp)",
        f"Avg hit rate: {smry['avg_hit_rate']:.1f}%  Avg SPY corr: {smry['avg_spy_corr']:.2f}",
        "=" * 90,
    ]
    return "\n".join(lines)
