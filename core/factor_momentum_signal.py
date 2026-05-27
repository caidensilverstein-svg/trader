"""
Factor momentum signal analysis and history tracker.

Computes and stores the 6-month factor momentum scores used for ETF weight
tilting. Scores above threshold get a +10% boost; below get a -20% penalty.

Also computes the Information Coefficient (IC) of the momentum signal --
i.e., how well the momentum score predicts next-period returns.

Academic basis:
  Jegadeesh & Titman (1993) "Returns to Buying Winners and Selling Losers"
  Asness, Moskowitz & Pedersen (2013) "Value and Momentum Everywhere"
  Novy-Marx (2012) "Is Momentum Really Momentum?" -- cross-sectional vs TS
  Hurst, Ooi & Pedersen (2017) "A Century of Evidence on Trend-Following"
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class MomentumSignal:
    """Momentum signal for a single ETF at a point in time."""
    ticker:         str
    date:           str
    score_6m:       float    # 6-month total return (momentum signal)
    score_12m:      float    # 12-month return (for cross-check)
    score_1m:       float    # 1-month return (skip for standard 12-1 momentum)
    composite_score: float   # 0.7 * 6m + 0.3 * 12m_skip1 (skip 1 month reversal)
    signal:         str      # "BOOST", "NEUTRAL", "PENALTY"
    weight_adj:     float    # +0.10, 0.0, or -0.20 multiplier applied


@dataclass
class ICAnalysis:
    """Information Coefficient analysis for momentum signal."""
    n_observations:  int
    ic_mean:         float    # mean correlation between signal and next-period return
    ic_std:          float    # standard deviation of IC
    ic_ir:           float    # IC Information Ratio = ic_mean / ic_std
    pct_positive_ic: float    # % of periods with positive IC
    ic_t_stat:       float    # t-statistic (>1.96 = statistically significant)


BOOST_THRESHOLD  = 0.05    # 6-month return > 5% => BOOST (+10%)
PENALTY_THRESHOLD = -0.02  # 6-month return < -2% => PENALTY (-20%)


def compute_momentum_signals(
    prices: pd.DataFrame,
    lookback_months: int = 6,
    skip_months: int = 1,
) -> List[MomentumSignal]:
    """
    Compute momentum signals for all ETFs at each rebalance date.

    Follows standard "12-1" momentum convention: compute 12-month return
    but skip the most recent month (to avoid short-term reversal).
    For 6-month lookback: use 6-month return, skip 1 month.

    Parameters
    ----------
    prices         : DataFrame, columns = tickers, index = dates
    lookback_months: months for momentum calculation
    skip_months    : months to skip (reversal avoidance)

    Returns
    -------
    List of MomentumSignal for each (ticker, date) combination
    """
    if prices.empty:
        return []

    tickers = list(prices.columns)
    returns = prices.pct_change().dropna()

    lookback_days = lookback_months * 21  # approx trading days
    skip_days     = skip_months * 21
    window_12m    = 252
    window_1m     = 21

    signals: List[MomentumSignal] = []

    # Compute monthly signals at monthly intervals
    monthly_dates = pd.date_range(returns.index[max(lookback_days + skip_days, 252)],
                                   returns.index[-1], freq="ME")

    for date in monthly_dates:
        if date not in returns.index:
            # Find nearest date
            idx_pos = returns.index.searchsorted(date, side="right")
            if idx_pos <= 0 or idx_pos >= len(returns.index):
                continue
            date = returns.index[idx_pos - 1]

        date_pos = returns.index.get_loc(date)
        if date_pos < lookback_days + skip_days:
            continue

        for ticker in tickers:
            if ticker not in returns.columns:
                continue

            price_s = prices[ticker]
            p_now   = float(price_s.iloc[date_pos])

            # 6-month return (skip 1 month)
            p_6m_ago = float(price_s.iloc[max(0, date_pos - lookback_days - skip_days)])
            p_skip   = float(price_s.iloc[max(0, date_pos - skip_days)])
            score_6m = (p_skip / p_6m_ago) - 1.0 if p_6m_ago > 0 else 0.0

            # 12-month return (skip 1 month)
            p_12m_ago = float(price_s.iloc[max(0, date_pos - window_12m - skip_days)])
            score_12m = (p_skip / p_12m_ago) - 1.0 if p_12m_ago > 0 else 0.0

            # 1-month return (the "reversal" signal)
            p_1m_ago = float(price_s.iloc[max(0, date_pos - window_1m)])
            score_1m = (p_now / p_1m_ago) - 1.0 if p_1m_ago > 0 else 0.0

            # Composite: 70% 6m + 30% 12m (skip 1m reversal)
            composite = 0.70 * score_6m + 0.30 * score_12m

            if composite >= BOOST_THRESHOLD:
                signal     = "BOOST"
                weight_adj = +0.10
            elif composite <= PENALTY_THRESHOLD:
                signal     = "PENALTY"
                weight_adj = -0.20
            else:
                signal     = "NEUTRAL"
                weight_adj = 0.0

            signals.append(MomentumSignal(
                ticker=ticker,
                date=str(date.date()),
                score_6m=round(score_6m, 4),
                score_12m=round(score_12m, 4),
                score_1m=round(score_1m, 4),
                composite_score=round(composite, 4),
                signal=signal,
                weight_adj=weight_adj,
            ))

    return signals


def compute_momentum_ic(
    prices: pd.DataFrame,
    forward_window: int = 21,
    lookback_months: int = 6,
) -> ICAnalysis:
    """
    Compute Information Coefficient of momentum signal.

    IC = correlation between momentum score at t and forward return at t+window.
    Positive IC = signal has predictive power.

    Parameters
    ----------
    prices         : DataFrame, columns = tickers
    forward_window : prediction horizon (days)
    lookback_months: momentum lookback

    Returns
    -------
    ICAnalysis
    """
    signals = compute_momentum_signals(prices, lookback_months=lookback_months)
    if not signals:
        return ICAnalysis(0, 0, 0, 0, 0, 0)

    # Build DataFrame
    df = pd.DataFrame([{
        "ticker": s.ticker,
        "date": pd.to_datetime(s.date),
        "score": s.composite_score,
    } for s in signals])

    returns = prices.pct_change()

    ics: List[float] = []
    unique_dates = sorted(df["date"].unique())

    for date in unique_dates:
        day_signals = df[df["date"] == date]
        # Forward return for each ticker
        fwd_rets = []
        scores = []
        for _, row in day_signals.iterrows():
            ticker = row["ticker"]
            if ticker not in prices.columns:
                continue
            date_pos = prices.index.searchsorted(date, side="right") - 1
            if date_pos + forward_window >= len(prices):
                continue
            fwd_price_start = float(prices[ticker].iloc[date_pos])
            fwd_price_end   = float(prices[ticker].iloc[date_pos + forward_window])
            if fwd_price_start > 0:
                fwd_ret = fwd_price_end / fwd_price_start - 1
                fwd_rets.append(fwd_ret)
                scores.append(float(row["score"]))

        if len(scores) >= 2:
            corr = float(np.corrcoef(scores, fwd_rets)[0, 1])
            if not np.isnan(corr):
                ics.append(corr)

    if not ics:
        return ICAnalysis(0, 0, 0, 0, 0, 0)

    ic_arr = np.array(ics)
    ic_mean = float(ic_arr.mean())
    ic_std  = float(ic_arr.std()) if len(ic_arr) > 1 else 0.0
    ic_ir   = ic_mean / ic_std if ic_std > 1e-9 else 0.0
    pct_pos = float((ic_arr > 0).mean() * 100)
    t_stat  = ic_mean / (ic_std / np.sqrt(len(ic_arr))) if ic_std > 0 and len(ic_arr) > 1 else 0.0

    return ICAnalysis(
        n_observations=len(ics),
        ic_mean=round(ic_mean, 4),
        ic_std=round(ic_std, 4),
        ic_ir=round(ic_ir, 3),
        pct_positive_ic=round(pct_pos, 1),
        ic_t_stat=round(t_stat, 3),
    )


def format_momentum_signal_report(
    signals: List[MomentumSignal],
    ic: Optional[ICAnalysis] = None,
    n_recent: int = 10,
) -> str:
    """Format momentum signals as ASCII report."""
    if not signals:
        return "Momentum signal data unavailable."

    # Group by ticker, get latest signal for each
    latest: Dict[str, MomentumSignal] = {}
    for s in signals:
        if s.ticker not in latest or s.date > latest[s.ticker].date:
            latest[s.ticker] = s

    lines = [
        "=" * 75,
        "FACTOR MOMENTUM SIGNALS  (Jegadeesh-Titman 1993, Asness et al. 2013)",
        "6-month momentum, skip 1 month (standard 12-1 convention)",
        "=" * 75,
        "CURRENT SIGNALS (latest rebalance date):",
        f"{'Ticker':<10} {'6M Ret':>9} {'12M Ret':>9} {'Composite':>10} {'Signal':>8} {'Adj':>7}",
        "-" * 58,
    ]

    for ticker, s in sorted(latest.items(), key=lambda x: x[1].composite_score, reverse=True):
        lines.append(
            f"{ticker:<10} "
            f"{s.score_6m*100:>+8.2f}% "
            f"{s.score_12m*100:>+8.2f}% "
            f"{s.composite_score*100:>+9.2f}% "
            f"{s.signal:>8} "
            f"{s.weight_adj*100:>+6.0f}%"
        )

    if ic and ic.n_observations > 0:
        lines += [
            "",
            "INFORMATION COEFFICIENT (Signal Predictive Power):",
            f"  Observations:       {ic.n_observations}",
            f"  Mean IC:            {ic.ic_mean:.4f}  (positive = signal adds value)",
            f"  IC Std Dev:         {ic.ic_std:.4f}",
            f"  IC Information Ratio: {ic.ic_ir:.3f}  (>0.3 = good signal quality)",
            f"  % Positive IC:      {ic.pct_positive_ic:.1f}%",
            f"  IC t-statistic:     {ic.ic_t_stat:.3f}  (>1.96 = statistically significant)",
        ]

    lines += [
        "",
        "BOOST = 6m composite return > 5% (ETF momentum positive, weight +10%)",
        "PENALTY = 6m composite return < -2% (ETF momentum negative, weight -20%)",
        "NEUTRAL = between thresholds (no weight adjustment)",
        "=" * 75,
    ]

    return "\n".join(lines)
