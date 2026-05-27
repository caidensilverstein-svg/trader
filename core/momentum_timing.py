"""
Time-series (absolute) momentum signal for market timing.

Academic basis: Antonacci (2012) "Risk Premia Harvesting Through Dual Momentum"
Moskowitz et al. (2012) "Time Series Momentum" (JFE)

Time-series momentum: if an asset's own trailing return is positive,
expect it to continue. If negative, expect it to continue falling.

This complements the regime detector by adding a second independent signal:
  - Regime detector: cross-sectional (SPY vs MA, VIX levels)
  - Momentum timing: time-series (SPY trailing return vs T-bills)

When both signals agree (positive momentum + BULL regime), confidence is high.
When they disagree, take the more conservative view.
"""

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Risk-free rate proxy (annualized, as daily)
RISK_FREE_DAILY = 0.05 / 252  # 5% annual T-bill rate


def spy_time_series_momentum(
    spy_prices: pd.Series,
    lookbacks: tuple = (1, 3, 6, 12),   # months
) -> Dict[str, float]:
    """
    Compute SPY time-series momentum over multiple lookback windows.

    Returns excess return (vs risk-free rate) for each window.
    Positive excess return = momentum is positive = stay in equities.

    Parameters
    ----------
    spy_prices : Daily SPY closing prices
    lookbacks  : Tuple of lookback periods in months (approx 21 trading days/month)

    Returns
    -------
    dict : {f'mom_{n}m': excess_return} for each lookback, plus composite signal
    """
    result = {}
    trading_days_per_month = 21

    for n_months in lookbacks:
        lookback_td = n_months * trading_days_per_month
        if len(spy_prices) < lookback_td + 1:
            result[f"mom_{n_months}m"] = None
            continue

        price_now  = float(spy_prices.iloc[-1])
        price_then = float(spy_prices.iloc[-lookback_td - 1])
        raw_return = (price_now / price_then) - 1.0

        # Excess return vs T-bill
        rf_return = RISK_FREE_DAILY * lookback_td
        excess    = raw_return - rf_return
        result[f"mom_{n_months}m"] = round(excess * 100, 2)  # in percentage points

    # Composite signal: weighted average of available lookbacks
    # Weights: 1m=20%, 3m=30%, 6m=30%, 12m=20%
    weights = {1: 0.20, 3: 0.30, 6: 0.30, 12: 0.20}
    composite = 0.0
    total_w   = 0.0
    for n_months, w in weights.items():
        val = result.get(f"mom_{n_months}m")
        if val is not None:
            composite += w * val
            total_w   += w

    result["composite"] = round(composite / total_w, 2) if total_w > 0 else 0.0
    result["signal"]    = "positive" if result["composite"] > 0 else "negative"

    return result


def etf_momentum_scores(etf_prices: Dict[str, pd.Series]) -> Dict[str, Dict]:
    """
    Compute 6-month time-series momentum for each ETF.

    Used to identify which factors are in a favorable regime.
    A factor with negative 6-month absolute momentum may be entering a downtrend.

    Returns dict of {ticker: {mom_6m, signal}}
    """
    scores = {}
    for ticker, prices in etf_prices.items():
        lookback_td = 126  # ~6 months
        if len(prices) < lookback_td + 1:
            scores[ticker] = {"mom_6m": None, "signal": "insufficient_data"}
            continue

        price_now  = float(prices.iloc[-1])
        price_then = float(prices.iloc[-lookback_td - 1])
        raw_return = (price_now / price_then) - 1.0
        rf_return  = RISK_FREE_DAILY * lookback_td
        excess     = (raw_return - rf_return) * 100

        scores[ticker] = {
            "mom_6m": round(excess, 2),
            "signal": "positive" if excess > 0 else "negative",
        }

    return scores


def combined_regime_signal(spy_regime: str, spy_mom_composite: float) -> str:
    """
    Combine the regime detector signal with the momentum signal.

    Priority:
      - If both are bearish -> DEFENSIVE (most conservative)
      - If regime is bearish but momentum positive -> CAUTIOUS
      - If regime is bullish but momentum negative -> CAUTIOUS
      - If both are bullish -> AGGRESSIVE

    Returns
    -------
    str : 'DEFENSIVE' / 'CAUTIOUS' / 'NEUTRAL' / 'AGGRESSIVE'
    """
    bear_regimes = {"BEAR", "BEAR_CRISIS"}
    bull_regimes = {"BULL", "MILD_BULL"}

    regime_bearish = spy_regime in bear_regimes
    regime_bullish = spy_regime in bull_regimes
    mom_positive   = spy_mom_composite > 0

    if regime_bearish and not mom_positive:
        return "DEFENSIVE"
    elif regime_bearish and mom_positive:
        return "CAUTIOUS"
    elif regime_bullish and mom_positive:
        return "AGGRESSIVE"
    elif regime_bullish and not mom_positive:
        return "CAUTIOUS"
    else:
        return "NEUTRAL"


def format_momentum_report(spy_mom: Dict, etf_scores: Dict, signal: str) -> str:
    """ASCII-safe momentum timing report."""
    lines = [
        "=" * 60,
        "MOMENTUM TIMING ANALYSIS",
        "=" * 60,
        "",
        "SPY TIME-SERIES MOMENTUM (excess vs 5% T-bill):",
        f"  1-month  : {spy_mom.get('mom_1m', 'N/A'):>+8}%",
        f"  3-month  : {spy_mom.get('mom_3m', 'N/A'):>+8}%",
        f"  6-month  : {spy_mom.get('mom_6m', 'N/A'):>+8}%",
        f"  12-month : {spy_mom.get('mom_12m', 'N/A'):>+8}%",
        f"  Composite: {spy_mom.get('composite', 0):>+8.2f}% --> {spy_mom.get('signal', 'N/A').upper()}",
        "",
        "ETF 6-MONTH MOMENTUM SCORES:",
    ]
    for ticker, s in etf_scores.items():
        mom = s.get("mom_6m")
        sig = s.get("signal", "N/A")
        mom_str = f"{mom:>+.2f}%" if mom is not None else "N/A"
        lines.append(f"  {ticker:<6} {mom_str:>8}  {sig.upper()}")

    lines += [
        "",
        f"COMBINED SIGNAL: {signal}",
        "  AGGRESSIVE: Full target weights",
        "  NEUTRAL:    Standard regime weights",
        "  CAUTIOUS:   Reduce 20% from regime target",
        "  DEFENSIVE:  Reduce 40% from regime target",
        "=" * 60,
    ]
    return "\n".join(lines)
