"""
Market regime detection.

Uses SPY vs 200-day MA, 60-day momentum, VIX level, and drawdown from
52-week high to classify the current market into one of five regimes.
Based on 45-wave research showing these four signals are sufficient.
"""

import logging
from typing import Tuple
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

REGIMES = ("BULL", "MILD_BULL", "SIDEWAYS", "BEAR", "BEAR_CRISIS")


def get_regime(
    spy_price: float,
    spy_ma200: float,
    spy_mom60: float,
    vix: float,
    dd_from_peak: float,
) -> str:
    """
    Classify current market regime from pre-fetched indicators.

    Parameters
    ----------
    spy_price   : Current SPY closing price
    spy_ma200   : SPY 200-day simple moving average
    spy_mom60   : SPY 60-calendar-day return (fraction, e.g. 0.05 = +5%)
    vix         : Current VIX level
    dd_from_peak: Drawdown from 52-week high (fraction, e.g. -0.15 = -15%)

    Returns
    -------
    str : One of BULL / MILD_BULL / SIDEWAYS / BEAR / BEAR_CRISIS
    """
    above_200 = spy_price > spy_ma200
    low_vix   = vix < 20.0
    strong_dd = dd_from_peak > -0.10  # within 10% of peak

    # Crisis overrides everything
    if vix > 30.0 or dd_from_peak < -0.20:
        return "BEAR_CRISIS"

    # Bear: price below 200MA and falling
    if not above_200 and spy_mom60 < 0:
        return "BEAR"

    # Bull: all green lights
    if above_200 and spy_mom60 > 0 and low_vix and strong_dd:
        return "BULL"

    # Mild bull: above 200MA but not all green
    if above_200 and low_vix:
        return "MILD_BULL"

    return "SIDEWAYS"


def compute_regime_indicators(
    spy_history: pd.Series,
    vix_history: pd.Series,
) -> Tuple[float, float, float, float, float]:
    """
    Compute regime indicators from price history.

    Parameters
    ----------
    spy_history : Daily SPY close prices (at least 252 trading days)
    vix_history : Daily VIX closes (at least 60 days)

    Returns
    -------
    Tuple of (spy_price, spy_ma200, spy_mom60, vix, dd_from_peak)
    """
    if len(spy_history) < 200:
        raise ValueError(f"Need at least 200 days of SPY data, got {len(spy_history)}")

    spy_price  = float(spy_history.iloc[-1])
    spy_ma200  = float(spy_history.rolling(200).mean().iloc[-1])

    # 60-day momentum (use min of available or 60)
    lookback = min(60, len(spy_history) - 1)
    spy_mom60 = float(spy_history.pct_change(lookback).iloc[-1])

    vix = float(vix_history.iloc[-1])

    peak_window = min(252, len(spy_history))
    peak = float(spy_history.iloc[-peak_window:].max())
    dd_from_peak = (spy_price / peak) - 1.0

    return spy_price, spy_ma200, spy_mom60, vix, dd_from_peak


def regime_from_history(spy_history: pd.Series, vix_history: pd.Series) -> str:
    """
    Full pipeline: compute indicators and classify regime.

    Parameters
    ----------
    spy_history : pd.Series of daily SPY closes
    vix_history : pd.Series of daily VIX closes

    Returns
    -------
    str : Current market regime
    """
    spy_price, spy_ma200, spy_mom60, vix, dd = compute_regime_indicators(
        spy_history, vix_history
    )
    regime = get_regime(spy_price, spy_ma200, spy_mom60, vix, dd)

    logger.info(
        "Regime: %s | SPY %.2f vs MA200 %.2f | Mom60 %+.1f%% | VIX %.1f | DD %.1f%%",
        regime, spy_price, spy_ma200, spy_mom60 * 100, vix, dd * 100,
    )
    return regime


def regime_summary(spy_history: pd.Series, vix_history: pd.Series) -> dict:
    """Return a dictionary of all regime signals for reporting."""
    spy_price, spy_ma200, spy_mom60, vix, dd = compute_regime_indicators(
        spy_history, vix_history
    )
    return {
        "regime":        regime_from_history(spy_history, vix_history),
        "spy_price":     round(spy_price, 2),
        "spy_ma200":     round(spy_ma200, 2),
        "spy_above_200": spy_price > spy_ma200,
        "spy_mom_60d":   round(spy_mom60 * 100, 2),
        "vix":           round(vix, 2),
        "dd_from_peak":  round(dd * 100, 2),
    }
