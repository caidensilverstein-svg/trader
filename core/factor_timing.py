"""
Factor Timing: Adjusts ETF sleeve weights based on individual ETF momentum.

Standard practice in factor investing: reduce allocation to factors with
negative recent momentum (momentum crash protection) and tilt toward
factors showing positive price momentum.

Strategy:
  - Compute 6-month momentum for each ETF
  - ETFs with negative 6-month return get a 20% weight reduction
  - ETFs with positive 6-month return maintain or get a 10% increase
  - Re-normalize so all weights still sum to target ETF sleeve total
  - NEVER reduces an ETF weight below 50% of its target

Why 6-month?
  - 1-month momentum is too noisy (reversal effect)
  - 12-month momentum has too much lag for factor timing
  - 6-month is the empirical sweet spot (Asness et al., 2013)

Academic basis: Asness, Moskowitz, Pedersen (2013) "Value and Momentum Everywhere"
               Gupta, Kelly (2019) "Factor Momentum Everywhere"
"""

import logging
from typing import Dict, Optional, Tuple

import pandas as pd
import numpy as np

import config
from core import data as mdata

logger = logging.getLogger(__name__)

# Factor timing parameters
MOMENTUM_LOOKBACK_DAYS = 126   # 6 months
NEGATIVE_MOM_PENALTY   = 0.80  # 20% reduction for negative momentum
POSITIVE_MOM_BOOST     = 1.10  # 10% increase for positive momentum
MIN_WEIGHT_FRACTION    = 0.50  # Never below 50% of target weight


def compute_etf_momentum(
    prices: Dict[str, pd.Series],
    lookback: int = MOMENTUM_LOOKBACK_DAYS,
) -> Dict[str, float]:
    """
    Compute 6-month (126-day) price momentum for each ETF.

    Momentum = (current price / price 126 days ago) - 1

    Parameters
    ----------
    prices  : {ticker: price_series}
    lookback: Days for lookback window

    Returns
    -------
    dict : {ticker: momentum_return_fraction}
    """
    momentum = {}
    for ticker, price_series in prices.items():
        prices_arr = price_series.dropna()
        if len(prices_arr) < lookback + 5:
            logger.warning("Insufficient data for %s momentum (%d days)", ticker, len(prices_arr))
            momentum[ticker] = 0.0
            continue

        current = float(prices_arr.iloc[-1])
        past    = float(prices_arr.iloc[-lookback])
        if past <= 0:
            momentum[ticker] = 0.0
        else:
            momentum[ticker] = round((current / past) - 1.0, 4)

    logger.info("ETF 6-month momentum: %s", momentum)
    return momentum


def compute_timing_multipliers(
    momentum: Dict[str, float],
    penalty: float = NEGATIVE_MOM_PENALTY,
    boost: float   = POSITIVE_MOM_BOOST,
) -> Dict[str, float]:
    """
    Convert momentum signals to weight multipliers.

    Positive momentum -> boost multiplier
    Negative momentum -> penalty multiplier

    Parameters
    ----------
    momentum : {ticker: return_fraction}

    Returns
    -------
    dict : {ticker: weight_multiplier}
    """
    multipliers = {}
    for ticker, mom in momentum.items():
        if mom < 0:
            multipliers[ticker] = penalty
        elif mom > 0:
            multipliers[ticker] = boost
        else:
            multipliers[ticker] = 1.0
    return multipliers


def apply_factor_timing(
    base_weights: Dict[str, float],
    momentum: Dict[str, float],
    penalty: float = NEGATIVE_MOM_PENALTY,
    boost: float   = POSITIVE_MOM_BOOST,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Apply factor timing adjustments to base ETF weights.

    Process:
    1. Apply momentum multipliers
    2. Clip to minimum weight (50% of target)
    3. Re-normalize so total equals original sum

    Parameters
    ----------
    base_weights : {ticker: target_fraction} (e.g., from compute_effective_weights)
    momentum     : {ticker: 6m_return} from compute_etf_momentum

    Returns
    -------
    (timed_weights, multipliers) tuple
    """
    multipliers = compute_timing_multipliers(momentum, penalty, boost)

    # Apply multipliers
    raw_adjusted = {}
    for ticker, w in base_weights.items():
        mult  = multipliers.get(ticker, 1.0)
        min_w = w * MIN_WEIGHT_FRACTION
        raw_adjusted[ticker] = max(w * mult, min_w)

    # Re-normalize to preserve total sleeve allocation
    original_total = sum(base_weights.values())
    adjusted_total = sum(raw_adjusted.values())

    if adjusted_total > 0 and original_total > 0:
        scale = original_total / adjusted_total
        timed = {t: round(w * scale, 4) for t, w in raw_adjusted.items()}
    else:
        timed = dict(base_weights)

    logger.info(
        "Factor timing applied: adjustments=%s, total %.2f%% -> %.2f%%",
        {t: f"{m:.2f}x" for t, m in multipliers.items()},
        original_total * 100, sum(timed.values()) * 100,
    )

    return timed, multipliers


def factor_timing_summary(
    base_weights: Dict[str, float],
    timed_weights: Dict[str, float],
    momentum: Dict[str, float],
    multipliers: Dict[str, float],
) -> str:
    """Format factor timing comparison as ASCII table."""
    lines = [
        "=" * 72,
        "FACTOR TIMING ADJUSTMENTS",
        "(6-month momentum-based weight tilt, Asness et al. 2013)",
        "=" * 72,
        f"{'Ticker':<8} {'Base Wt':>8} {'6M Mom':>8} {'Multiplier':>11} {'Timed Wt':>9} {'Change':>8}",
        "-" * 58,
    ]
    for ticker in sorted(base_weights.keys()):
        bw   = base_weights.get(ticker, 0)
        tw   = timed_weights.get(ticker, 0)
        mom  = momentum.get(ticker, 0)
        mult = multipliers.get(ticker, 1.0)
        diff = tw - bw
        lines.append(
            f"{ticker:<8} {bw*100:>7.1f}% {mom*100:>+7.1f}% {mult:>10.2f}x "
            f"{tw*100:>8.1f}% {diff*100:>+7.1f}%"
        )
    total_base  = sum(base_weights.values())
    total_timed = sum(timed_weights.values())
    lines += [
        "-" * 58,
        f"{'TOTAL':<8} {total_base*100:>7.1f}% {'':>8} {'':>11} {total_timed*100:>8.1f}%",
        "",
        "=" * 72,
    ]
    return "\n".join(lines)
