"""
Post-Earnings Announcement Drift (PEAD) Screener.

Screens for stocks that just beat earnings by a large margin and are
showing drift continuation. Executes long equity positions on Alpaca.

Academic backing: Kaczmarek & Zaremba (2025) — ML-enhanced PEAD for
small-mid cap delivers Sharpe 0.63, +0.4%/month in live-traded universe.

Target universe: $500M - $3B market cap, >$1M daily volume.
Entry: Day 2 after announcement if stock gapped up AND held the gain.
Exit: -7% stop OR 45 trading days.
Position size: $2,000 - $5,000, max 3 simultaneous.
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

import config
from core import data as mdata
from core.utils import load_state, save_state, now_utc, append_trade_log
from execution.alpaca_client import AlpacaClient
from execution.order_manager import OrderManager

logger = logging.getLogger(__name__)

# Earnings screener universe — liquid small-mid caps that report regularly
# In production this would come from an earnings calendar API.
# Using a curated watchlist from yfinance's most actively reporting tickers.
SCREENER_UNIVERSE = [
    # Small-cap technology
    "APPS", "ACLS", "FORM", "KLIC", "RMBS",
    # Small-cap financials
    "CATY", "EWBC", "FFIN", "IBTX", "PRSS",
    # Small-cap healthcare
    "ACAD", "AKRO", "ARDX", "ARWR", "AVXL",
    # Small-cap industrials
    "AQUA", "ASTE", "CMT", "GNRC", "HEES",
    # Small-cap consumer
    "CAKE", "CHUY", "JACK", "NATH", "SHAK",
    # Mid-cap tech
    "ALTR", "BLKB", "CGNX", "COHU", "DIOD",
    # Mid-cap healthcare
    "AGIO", "ALNY", "ARGX", "ARGT", "ARVN",
    # Additional liquid names
    "EXLS", "GFAI", "HALO", "IIVI", "JAMF",
    "KRYS", "LPSN", "MARA", "NABL", "NTLA",
    "OSIS", "PCOR", "QDEL", "RGEN", "RXRX",
    "SAGA", "TBBK", "UDMY", "VRNS", "XPRO",
]


# ---------------------------------------------------------------------------
# SUE calculation
# ---------------------------------------------------------------------------

def calculate_sue_score(ticker: str) -> Optional[float]:
    """
    Standardized Unexpected Earnings (SUE) score.

    SUE = (Actual EPS - Mean of last 8 quarters EPS) / Std(last 8 quarters EPS)

    Returns None if insufficient data.
    """
    try:
        earnings = mdata.get_earnings_history(ticker)
        if earnings is None or len(earnings) < 4:
            return None

        # yfinance earnings history columns vary — try different formats
        for col in ["Reported EPS", "reportedEPS", "Reported", "EPS Actual"]:
            if col in earnings.columns:
                eps_col = col
                break
        else:
            # try to find any numeric column
            num_cols = earnings.select_dtypes(include=[float, int]).columns.tolist()
            if not num_cols:
                return None
            eps_col = num_cols[0]

        eps = earnings[eps_col].dropna().astype(float)
        if len(eps) < 3:
            return None

        actual   = float(eps.iloc[0])
        history  = eps.iloc[1:9]

        if history.std() == 0:
            return None

        sue = (actual - history.mean()) / history.std()
        return float(sue)

    except Exception as exc:
        logger.debug("SUE calc failed for %s: %s", ticker, exc)
        return None


def get_earnings_surprise_pct(ticker: str) -> Optional[float]:
    """
    Return EPS surprise percentage from yfinance earnings history.
    Returns None if unavailable.
    """
    try:
        t = yf.Ticker(ticker)
        hist = t.get_earnings_history()
        if hist is None or hist.empty:
            return None

        for col in ["surprisePercent", "Surprise(%)", "Surprise Percent"]:
            if col in hist.columns:
                val = hist[col].iloc[0]
                return float(val) if pd.notna(val) else None

        # Try to compute from estimate vs actual
        for est_col in ["EPS Estimate", "epsEstimate", "Estimated EPS"]:
            for act_col in ["Reported EPS", "reportedEPS", "Actual EPS"]:
                if est_col in hist.columns and act_col in hist.columns:
                    est = hist[est_col].iloc[0]
                    act = hist[act_col].iloc[0]
                    if pd.notna(est) and pd.notna(act) and est != 0:
                        return float((act - est) / abs(est))

        return None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Gap detection
# ---------------------------------------------------------------------------

def detect_earnings_gap(prices: pd.Series, announcement_date_idx: int = -2) -> Tuple[float, bool]:
    """
    Detect if a stock gapped up on/after earnings announcement.

    announcement_date_idx : -2 means the day before yesterday in prices series

    Returns
    -------
    (gap_pct, held_gain)
    gap_pct  : Percentage gap from prior close to announcement day open/close
    held_gain: True if Day 1 close is above the gap close
    """
    if len(prices) < 4:
        return 0.0, False

    # Day 0 = announcement day, Day 1 = day after
    prior_close = float(prices.iloc[announcement_date_idx - 1])
    day0_close  = float(prices.iloc[announcement_date_idx])
    day1_close  = float(prices.iloc[-1])

    if prior_close <= 0:
        return 0.0, False

    gap_pct   = (day0_close / prior_close) - 1.0
    held_gain = day1_close >= day0_close  # Day 1 doesn't give back the gap

    return float(gap_pct), held_gain


# ---------------------------------------------------------------------------
# Candidate scoring
# ---------------------------------------------------------------------------

def score_candidate(ticker: str) -> Optional[Dict]:
    """
    Full PEAD candidate evaluation.

    Returns score dict or None if candidate doesn't pass filters.
    """
    # Market cap filter
    mcap = mdata.get_market_cap(ticker)
    if mcap is None or mcap < config.PEAD_MCAP_MIN or mcap > config.PEAD_MCAP_MAX:
        return None

    # Volume filter
    avg_vol = mdata.get_avg_volume(ticker, days=20)
    if avg_vol < config.PEAD_VOLUME_MIN:
        return None

    # Earnings surprise
    surprise = get_earnings_surprise_pct(ticker)
    if surprise is None or surprise < config.PEAD_SURPRISE_MIN:
        return None

    # Price history for gap analysis
    try:
        prices = mdata.get_price_history(ticker, "1mo")
        if len(prices) < 5:
            return None
        gap_pct, held_gain = detect_earnings_gap(prices)
    except Exception:
        return None

    if gap_pct < config.PEAD_GAP_MIN or not held_gain:
        return None

    # SUE score
    sue = calculate_sue_score(ticker)

    # Composite score (weighted combination)
    surprise_score = min(surprise / 0.30, 1.0)  # normalized to max at 30%
    gap_score      = min(gap_pct / 0.10, 1.0)   # normalized to max at 10% gap
    sue_norm       = min(abs(sue) / 3.0, 1.0) if sue else 0.0

    composite = 0.40 * surprise_score + 0.35 * gap_score + 0.25 * sue_norm

    current_price = float(prices.iloc[-1])
    stop_price    = current_price * (1 + config.PEAD_STOP_LOSS)

    return {
        "ticker":        ticker,
        "surprise_pct":  round(surprise * 100, 2),
        "gap_pct":       round(gap_pct * 100, 2),
        "held_gain":     held_gain,
        "sue_score":     round(sue, 3) if sue else None,
        "composite":     round(composite, 3),
        "mcap_m":        round(mcap / 1e6, 1),
        "avg_vol_m":     round(avg_vol / 1e6, 2),
        "current_price": round(current_price, 2),
        "stop_price":    round(stop_price, 2),
        "scanned_at":    now_utc(),
    }


def get_pead_candidates(max_candidates: int = 10) -> List[Dict]:
    """
    Scan the universe and return top PEAD candidates sorted by composite score.
    """
    logger.info("Scanning PEAD universe (%d tickers)…", len(SCREENER_UNIVERSE))
    candidates = []

    for ticker in SCREENER_UNIVERSE:
        try:
            result = score_candidate(ticker)
            if result:
                candidates.append(result)
                logger.debug("PEAD candidate: %s  score=%.3f  surprise=%.1f%%",
                             ticker, result["composite"], result["surprise_pct"])
        except Exception as exc:
            logger.debug("PEAD screen error %s: %s", ticker, exc)

    candidates.sort(key=lambda x: x["composite"], reverse=True)
    logger.info("Found %d PEAD candidates", len(candidates))
    return candidates[:max_candidates]


# ---------------------------------------------------------------------------
# Position management
# ---------------------------------------------------------------------------

def open_pead_positions(
    candidates: List[Dict],
    om: OrderManager,
    client: AlpacaClient,
    dry_run: bool = False,
) -> List[Dict]:
    """
    Open PEAD positions for top candidates.

    Respects max simultaneous position limit and per-trade size limits.
    """
    state = load_state(config.PEAD_FILE, {"open_positions": {}, "closed_positions": []})
    open_pos = state.get("open_positions", {})

    # Check how many slots remain
    available_slots = config.PEAD_MAX_POSITIONS - len(open_pos)
    if available_slots <= 0:
        logger.info("PEAD: max positions (%d) reached", config.PEAD_MAX_POSITIONS)
        return []

    # Check circuit breaker
    if om.is_trading_halted():
        logger.warning("PEAD: circuit breaker open, skipping")
        return []

    opened = []
    for candidate in candidates[:available_slots]:
        ticker = candidate["ticker"]
        if ticker in open_pos:
            logger.debug("PEAD: %s already in portfolio", ticker)
            continue

        # Position sizing: use composite score to scale between min and max
        position_size = config.PEAD_POSITION_MIN + (
            (config.PEAD_POSITION_MAX - config.PEAD_POSITION_MIN) * candidate["composite"]
        )
        position_size = round(position_size, 2)

        logger.info(
            "PEAD ENTRY: %s  composite=%.3f  surprise=%.1f%%  size=$%.2f",
            ticker, candidate["composite"], candidate["surprise_pct"], position_size,
        )

        order_id = None
        if not dry_run:
            order_id = om.buy_notional(
                ticker, position_size, "PEAD",
                f"surprise={candidate['surprise_pct']:.1f}% sue={candidate.get('sue_score')}",
                max_loss=config.MAX_LOSS_PEAD,
            )

        if dry_run or order_id:
            open_pos[ticker] = {
                "entry_date":    now_utc()[:10],
                "entry_price":   candidate["current_price"],
                "stop_price":    candidate["stop_price"],
                "position_size": position_size,
                "order_id":      order_id or "dry_run",
                "days_held":     0,
                "composite":     candidate["composite"],
                "surprise_pct":  candidate["surprise_pct"],
            }
            opened.append({"ticker": ticker, **open_pos[ticker]})

    state["open_positions"] = open_pos
    state["last_scan"] = now_utc()
    save_state(config.PEAD_FILE, state)
    return opened


def check_pead_exits(
    om: OrderManager,
    client: AlpacaClient,
    dry_run: bool = False,
) -> List[Dict]:
    """
    Check all open PEAD positions for stop-loss and time exits.
    """
    state = load_state(config.PEAD_FILE, {"open_positions": {}, "closed_positions": []})
    open_pos = state.get("open_positions", {})
    if not open_pos:
        return []

    positions = client.get_positions()
    closed_today = []

    for ticker, pos_info in list(open_pos.items()):
        pos_info["days_held"] = pos_info.get("days_held", 0) + 1
        close_reason = None

        # Get current price
        if ticker in positions:
            current_price = float(positions[ticker]["avg_entry"])  # approximate
            entry_price = float(pos_info["entry_price"])
            pnl_pct = (current_price / entry_price) - 1.0
        else:
            pnl_pct = 0.0
            current_price = float(pos_info["entry_price"])

        # Stop loss
        if pnl_pct <= config.PEAD_STOP_LOSS:
            close_reason = f"Stop loss hit ({pnl_pct*100:.1f}%)"

        # Time exit
        if pos_info["days_held"] >= config.PEAD_HOLD_DAYS:
            close_reason = f"Time exit ({config.PEAD_HOLD_DAYS} days)"

        if close_reason:
            logger.info("PEAD EXIT: %s  reason=%s  pnl=%.1f%%", ticker, close_reason, pnl_pct * 100)
            order_id = None
            if not dry_run:
                order_id = om.close(ticker, "PEAD", close_reason)

            closed_record = {
                "ticker":       ticker,
                "close_reason": close_reason,
                "days_held":    pos_info["days_held"],
                "pnl_pct":      round(pnl_pct * 100, 2),
                "order_id":     order_id or "dry_run",
                "closed_at":    now_utc(),
            }
            closed_today.append(closed_record)
            state.setdefault("closed_positions", []).append(closed_record)
            append_trade_log(config.LOG_FILE, {"action": "PEAD_CLOSE", **closed_record})
            del open_pos[ticker]

    state["open_positions"] = open_pos
    save_state(config.PEAD_FILE, state)
    return closed_today


def get_pead_status() -> dict:
    """Return PEAD strategy summary."""
    state = load_state(config.PEAD_FILE, {"open_positions": {}, "closed_positions": []})
    closed = state.get("closed_positions", [])
    wins   = [c for c in closed if float(c.get("pnl_pct", 0)) > 0]

    return {
        "open_count":   len(state.get("open_positions", {})),
        "closed_count": len(closed),
        "win_rate":     round(len(wins) / len(closed) * 100, 1) if closed else 0,
        "open_positions": list(state.get("open_positions", {}).keys()),
    }
