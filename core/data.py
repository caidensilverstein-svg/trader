"""
Data fetching layer.

Single source of truth for all market data. Wraps yfinance for historical
data and Alpaca for current positions and account state.
Caches results per session to avoid hammering APIs.
"""

import logging
import time
from datetime import datetime, timezone
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
import yfinance as yf

logger = logging.getLogger(__name__)

# Module-level cache: ticker -> (timestamp, DataFrame)
_price_cache: Dict[str, Tuple[float, pd.Series]] = {}
CACHE_TTL = 3600  # 1 hour


def _cached(ticker: str, period: str, force: bool = False) -> pd.Series:
    """Fetch closing prices with simple TTL cache."""
    key = f"{ticker}|{period}"
    now = time.time()
    if not force and key in _price_cache:
        ts, data = _price_cache[key]
        if now - ts < CACHE_TTL:
            return data

    try:
        raw = yf.download(ticker, period=period, auto_adjust=True, progress=False)
        if raw.empty:
            raise ValueError(f"Empty data for {ticker}")
        close = raw["Close"].squeeze()
        _price_cache[key] = (now, close)
        return close
    except Exception as exc:
        logger.error("yfinance download failed for %s: %s", ticker, exc)
        raise


def get_spy_vix(period: str = "2y") -> Tuple[pd.Series, pd.Series]:
    """Return (SPY, VIX) closing price series."""
    spy = _cached("SPY", period)
    vix = _cached("^VIX", period)
    # Align on common dates
    aligned = pd.concat([spy, vix], axis=1, keys=["SPY", "VIX"]).dropna()
    return aligned["SPY"], aligned["VIX"]


def get_etf_prices(tickers: List[str], period: str = "1y") -> pd.DataFrame:
    """Return a DataFrame of closing prices for given tickers."""
    frames = {}
    for t in tickers:
        try:
            frames[t] = _cached(t, period)
        except Exception as exc:
            logger.warning("Could not fetch %s: %s", t, exc)
    if not frames:
        raise RuntimeError("No ETF price data available")
    df = pd.DataFrame(frames).dropna()
    return df


def get_current_vix() -> float:
    """Return the most recent VIX close."""
    try:
        vix = _cached("^VIX", "5d", force=True)
        return float(vix.iloc[-1])
    except Exception:
        logger.warning("VIX fetch failed, returning 20.0 as default")
        return 20.0


def get_current_price(ticker: str) -> float:
    """Return most recent closing price for a ticker."""
    prices = _cached(ticker, "5d", force=True)
    return float(prices.iloc[-1])


def get_earnings_history(ticker: str) -> Optional[pd.DataFrame]:
    """
    Return earnings history for a ticker from yfinance.
    Columns include: EPS Estimate, Reported EPS, Surprise(%)
    """
    try:
        t = yf.Ticker(ticker)
        hist = t.get_earnings_history()
        if hist is None or hist.empty:
            return None
        return hist
    except Exception as exc:
        logger.warning("Earnings fetch failed for %s: %s", ticker, exc)
        return None


def get_price_history(ticker: str, period: str = "1y") -> pd.Series:
    """Return daily close prices for a single ticker."""
    return _cached(ticker, period)


def get_market_cap(ticker: str) -> Optional[float]:
    """Return market cap from yfinance info."""
    try:
        info = yf.Ticker(ticker).info
        mc = info.get("marketCap") or info.get("market_cap")
        return float(mc) if mc else None
    except Exception:
        return None


def get_avg_volume(ticker: str, days: int = 20) -> float:
    """Return average daily dollar volume over last `days` days."""
    try:
        raw = yf.download(ticker, period="3mo", auto_adjust=True, progress=False)
        if raw.empty:
            return 0.0
        dv = (raw["Close"].squeeze() * raw["Volume"].squeeze()).dropna()
        return float(dv.tail(days).mean())
    except Exception:
        return 0.0


def compute_realized_variance(prices: pd.Series, window: int) -> float:
    """
    Compute rolling realized variance over last `window` days.
    Returns annualized variance (daily_var * 252).
    """
    daily_returns = prices.pct_change().dropna()
    if len(daily_returns) < window:
        window = len(daily_returns)
    daily_var = float(daily_returns.tail(window).var())
    return daily_var * 252


def get_etf_momentum(tickers: List[str], period: str = "18mo") -> Dict[str, float]:
    """Return 12-month trailing return for each ticker."""
    result = {}
    for t in tickers:
        try:
            prices = _cached(t, period)
            ret = float(prices.pct_change(252).iloc[-1])
            result[t] = round(ret * 100, 2)
        except Exception:
            result[t] = float("nan")
    return result


def clear_cache():
    """Clear the price cache (call before a new session.)"""
    global _price_cache
    _price_cache = {}
    logger.info("Price cache cleared")
