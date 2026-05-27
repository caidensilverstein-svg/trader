"""
Kelly Criterion position sizing.

Computes full and fractional Kelly weights for each strategy layer.
Academic basis: Kelly (1956), Thorp (2008), MacLean-Ziemba-Blazenko (1992).

The Kelly fraction maximizes long-run geometric growth rate.
We use HALF-Kelly (50%) by default for robustness — reduces bet size
and variance while still improving on equal-weight allocation.

This is used for:
  - PEAD position sizing ($2,000-$5,000 range, Kelly decides within range)
  - M&A spread sizing ($2,500 base, Kelly adjusts by spread quality)
  - Iron condor sizing (already handled by VIX-based multiplier)
"""

import logging
import math
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)


def kelly_fraction(
    win_rate: float,
    win_size: float,
    loss_size: float,
) -> float:
    """
    Compute Kelly fraction for a binary bet.

    Kelly formula: f* = (p * b - q) / b
    where p = win probability, q = 1-p, b = win/loss ratio.

    Parameters
    ----------
    win_rate  : Historical win rate [0, 1]
    win_size  : Average win as fraction of position (e.g. 0.08 = 8% gain)
    loss_size : Average loss as fraction of position (e.g. 0.04 = 4% loss, positive)

    Returns
    -------
    float : Optimal Kelly fraction of capital to bet [0, 1]
            Returns 0 if the bet has negative expected value.
    """
    if win_rate <= 0 or win_rate >= 1 or win_size <= 0 or loss_size <= 0:
        return 0.0

    b = win_size / loss_size   # win/loss ratio
    p = win_rate
    q = 1.0 - win_rate

    kelly = (p * b - q) / b
    return max(0.0, kelly)


def half_kelly_pead(
    win_rate: float = 0.55,
    avg_win_pct: float = 0.08,
    avg_loss_pct: float = 0.07,
    capital: float = 100_000,
    min_notional: float = 2_000,
    max_notional: float = 5_000,
) -> float:
    """
    Compute PEAD position size using half-Kelly.

    Default parameters based on PEAD academic literature:
    - Win rate: 55% (consistent with Kaczmarek & Zaremba 2025 small-cap PEAD)
    - Avg win: 8% (drift continuation)
    - Avg loss: 7% (stop loss)

    Returns notional in dollars, clamped to [min_notional, max_notional].
    """
    f_star = kelly_fraction(win_rate, avg_win_pct, avg_loss_pct)
    half_f = f_star * 0.5  # half-Kelly for robustness

    notional = half_f * capital
    result = max(min_notional, min(max_notional, notional))

    logger.debug(
        "Kelly PEAD: win_rate=%.2f avg_win=%.1f%% avg_loss=%.1f%% "
        "f*=%.3f half_f=%.3f notional=$%.0f",
        win_rate, avg_win_pct * 100, avg_loss_pct * 100,
        f_star, half_f, result,
    )
    return result


def half_kelly_ma(
    spread_pct: float,
    deal_success_rate: float = 0.90,
    breakeven_days: int = 90,
    capital: float = 100_000,
    min_notional: float = 1_500,
    max_notional: float = 3_500,
) -> float:
    """
    Compute M&A arbitrage position size using half-Kelly.

    M&A Kelly parameters:
    - Deal success rate: 90% (historical cash deal completion rate)
    - Win = spread (e.g. 2% spread captured)
    - Loss = stop-loss level (10% stop)

    Parameters
    ----------
    spread_pct        : Current spread as fraction (e.g. 0.02 = 2%)
    deal_success_rate : P(deal closes) estimate
    """
    win_size  = spread_pct          # gain if deal closes
    loss_size = 0.10                # loss if deal falls through (our stop)

    f_star = kelly_fraction(deal_success_rate, win_size, loss_size)
    half_f = f_star * 0.5

    notional = half_f * capital
    result = max(min_notional, min(max_notional, notional))

    logger.debug(
        "Kelly M&A: spread=%.2f%% success_rate=%.2f f*=%.3f notional=$%.0f",
        spread_pct * 100, deal_success_rate, f_star, result,
    )
    return result


def log_growth_rate(
    kelly_f: float,
    win_rate: float,
    win_size: float,
    loss_size: float,
) -> float:
    """
    Compute the expected log growth rate at a given Kelly fraction.

    G(f) = p * log(1 + f*b) + q * log(1 - f)
    where b = win_size / loss_size.

    Returns expected log return per bet (per trade).
    """
    if kelly_f <= 0 or win_rate <= 0:
        return 0.0

    b = win_size / loss_size
    p = win_rate
    q = 1.0 - win_rate

    try:
        g = p * math.log(1 + kelly_f * b) + q * math.log(1 - kelly_f)
        return g
    except ValueError:
        return 0.0


def kelly_summary(
    win_rate: float,
    win_size: float,
    loss_size: float,
    capital: float,
    label: str = "Strategy",
) -> Dict:
    """
    Return a summary dict for reporting and logging.
    """
    f_star  = kelly_fraction(win_rate, win_size, loss_size)
    half_f  = f_star * 0.5
    g_full  = log_growth_rate(f_star, win_rate, win_size, loss_size)
    g_half  = log_growth_rate(half_f, win_rate, win_size, loss_size)
    ev      = win_rate * win_size - (1 - win_rate) * loss_size

    return {
        "label":           label,
        "win_rate":        round(win_rate, 3),
        "win_size":        round(win_size, 3),
        "loss_size":       round(loss_size, 3),
        "expected_value":  round(ev, 4),
        "kelly_fraction":  round(f_star, 4),
        "half_kelly":      round(half_f, 4),
        "notional_full":   round(f_star * capital, 2),
        "notional_half":   round(half_f * capital, 2),
        "log_growth_full": round(g_full, 6),
        "log_growth_half": round(g_half, 6),
    }
