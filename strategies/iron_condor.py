"""
Iron Condor Signal Generator.

IMPORTANT: Alpaca paper trading does NOT support options execution.
This module generates signals, tracks hypothetical trades, and reports
P&L — but does NOT place orders through Alpaca.

For live execution, move positions to tastytrade or IBKR.

Strategy:
  - SPX monthly iron condors, 30-45 DTE
  - Sell 16-delta call + 16-delta put
  - Buy wings 35 points OTM
  - Exit: 50% profit, 200% loss, or 21 DTE
  - Skip when VIX < 15 or VIX > 35
  - Section 1256 tax treatment (SPX cash-settled, 60/40 long/short gain)
"""

import logging
from datetime import datetime, timezone, timedelta
from typing import Optional, List
import math

import config
from core.utils import load_state, save_state, now_utc, append_trade_log

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# VIX-based sizing
# ---------------------------------------------------------------------------

def condor_size_multiplier(vix: float) -> float:
    """
    Return position size multiplier based on VIX.

    VIX < 15  : 0.0  (skip)
    15-20      : 1.0  (full size)
    20-25      : 0.75 (reduced)
    25-35      : 0.25 (very small)
    > 35       : 0.0  (skip)
    """
    if vix < config.CONDOR_VIX_MIN:
        return 0.0
    elif vix < 20:
        return 1.0
    elif vix < 25:
        return 0.75
    elif vix < config.CONDOR_VIX_MAX:
        return 0.25
    else:
        return 0.0


def should_open_condor(vix: float, regime: str) -> tuple:
    """
    Determine whether to open a new iron condor.

    Returns
    -------
    (bool, str) : (open_it, reason)
    """
    mult = condor_size_multiplier(vix)
    if mult == 0.0:
        return False, f"VIX={vix:.1f} outside tradeable range ({config.CONDOR_VIX_MIN}-{config.CONDOR_VIX_MAX})"
    if regime == "BEAR_CRISIS":
        return False, f"Regime={regime} — skip during crisis"
    regime_mult = config.REGIME_CONDOR_MULT.get(regime, 1.0)
    if regime_mult == 0.0:
        return False, f"Regime={regime} condor mult=0"
    return True, f"VIX={vix:.1f} regime={regime} size_mult={mult:.2f}"


# ---------------------------------------------------------------------------
# Strike estimation (simplified without live options chain)
# ---------------------------------------------------------------------------

def estimate_strikes(spx_price: float, vix: float, dte: int) -> dict:
    """
    Estimate iron condor strikes from SPX price and VIX.

    Uses the relationship: 1-sigma move ≈ SPX * (VIX/100) * sqrt(dte/365)
    16-delta ≈ 1-sigma move from current price.

    Parameters
    ----------
    spx_price : Current SPX index level
    vix       : Current VIX level
    dte       : Days to expiration

    Returns
    -------
    dict with short_put, long_put, short_call, long_call, expected_credit
    """
    sigma = spx_price * (vix / 100) * math.sqrt(dte / 365.0)
    sigma_16 = sigma * 0.87  # ~16-delta is ~0.87 sigma

    # Round to nearest 5 (SPX options strike spacing)
    def r5(x): return round(x / 5) * 5

    short_put  = r5(spx_price - sigma_16)
    long_put   = r5(short_put - config.CONDOR_WING_POINTS)
    short_call = r5(spx_price + sigma_16)
    long_call  = r5(short_call + config.CONDOR_WING_POINTS)

    # Rough credit estimate: ~1/3 of wing width at 16-delta
    credit_per_spread = config.CONDOR_WING_POINTS / 3.0
    total_credit = credit_per_spread  # net credit per condor

    return {
        "spx_price":   round(spx_price, 2),
        "short_put":   int(short_put),
        "long_put":    int(long_put),
        "short_call":  int(short_call),
        "long_call":   int(long_call),
        "wing_width":  config.CONDOR_WING_POINTS,
        "est_credit":  round(total_credit * 100, 2),  # in dollars (x100 multiplier)
        "max_loss":    round((config.CONDOR_WING_POINTS - total_credit) * 100, 2),
        "profit_target": round(total_credit * 100 * config.CONDOR_PROFIT_TARGET, 2),
    }


# ---------------------------------------------------------------------------
# Condor state management
# ---------------------------------------------------------------------------

def open_condor_signal(
    spx_price: float,
    vix: float,
    regime: str,
    dte: int = config.CONDOR_DTE_TARGET,
) -> Optional[dict]:
    """
    Generate a new iron condor signal.

    Returns signal dict or None if conditions not met.
    Signal is logged to state file for tracking.
    """
    open_it, reason = should_open_condor(vix, regime)
    if not open_it:
        logger.info("Condor skipped: %s", reason)
        return None

    state = load_state(config.CONDOR_FILE, {"open_condors": []})
    open_condors = state.get("open_condors", [])

    # Only one condor at a time (simplified for single-account paper trading)
    if open_condors:
        logger.info("Condor already open (%d), skipping new entry", len(open_condors))
        return None

    strikes = estimate_strikes(spx_price, vix, dte)
    mult    = condor_size_multiplier(vix) * config.REGIME_CONDOR_MULT.get(regime, 1.0)

    entry_date = datetime.now(timezone.utc).date().isoformat()
    expiry_date = (datetime.now(timezone.utc) + timedelta(days=dte)).date().isoformat()

    condor = {
        "id":           len(state.get("all_condors", [])) + 1,
        "entry_date":   entry_date,
        "expiry_date":  expiry_date,
        "dte":          dte,
        "spx_at_entry": spx_price,
        "vix_at_entry": vix,
        "regime":       regime,
        "short_put":    strikes["short_put"],
        "long_put":     strikes["long_put"],
        "short_call":   strikes["short_call"],
        "long_call":    strikes["long_call"],
        "est_credit":   strikes["est_credit"],
        "max_loss":     strikes["max_loss"],
        "profit_target": strikes["profit_target"],
        "size_mult":    round(mult, 2),
        "status":       "OPEN",
        "pnl":          0.0,
        "close_reason": None,
    }

    open_condors.append(condor)
    state["open_condors"] = open_condors
    state.setdefault("all_condors", []).append(condor)
    state["last_signal"] = now_utc()
    save_state(config.CONDOR_FILE, state)

    logger.info(
        "CONDOR SIGNAL: SPX=%.0f  Put spread %d/%d  Call spread %d/%d  "
        "Credit $%.2f  DTE=%d",
        spx_price, strikes["long_put"], strikes["short_put"],
        strikes["short_call"], strikes["long_call"],
        strikes["est_credit"], dte,
    )

    append_trade_log(config.LOG_FILE, {"action": "CONDOR_OPEN", **condor})
    return condor


def check_condor_exits(spx_price: float, vix: float) -> List[dict]:
    """
    Check open condors against exit rules.

    Exit rules (in priority order):
    1. DTE <= 21: time exit
    2. Estimated value >= profit_target (50% of credit)
    3. Estimated value >= 2x max_loss (200% loss)

    NOTE: Without a live options chain, P&L is estimated from
    SPX move relative to short strikes (simplified model).

    Returns list of closed condor dicts.
    """
    state = load_state(config.CONDOR_FILE, {"open_condors": []})
    open_condors = state.get("open_condors", [])
    if not open_condors:
        return []

    closed = []
    remaining = []
    today = datetime.now(timezone.utc).date()

    for condor in open_condors:
        expiry = datetime.fromisoformat(condor["expiry_date"]).date()
        dte_remaining = (expiry - today).days
        close_reason = None

        # Check DTE exit
        if dte_remaining <= config.CONDOR_DTE_EXIT:
            close_reason = f"DTE exit ({dte_remaining} days remaining)"

        # Simplified P&L estimate from SPX move
        if close_reason is None:
            entry_spx = float(condor["spx_at_entry"])
            move_pct   = abs(spx_price - entry_spx) / entry_spx
            move_pts   = abs(spx_price - entry_spx)

            # If SPX has moved within short strikes: value decays
            # Rough: if within short strikes, P&L = +50% credit at midpoint of life
            # If outside short strikes: P&L = negative
            short_put  = condor["short_put"]
            short_call = condor["short_call"]
            credit     = float(condor["est_credit"])
            max_loss   = float(condor["max_loss"])

            if spx_price < short_put:
                # Below short put: losing on put spread
                penetration = (short_put - spx_price) / config.CONDOR_WING_POINTS
                est_pnl = -min(penetration * max_loss, max_loss)
            elif spx_price > short_call:
                # Above short call: losing on call spread
                penetration = (spx_price - short_call) / config.CONDOR_WING_POINTS
                est_pnl = -min(penetration * max_loss, max_loss)
            else:
                # Inside the strikes: time decay working in our favor
                days_elapsed = (today - datetime.fromisoformat(condor["entry_date"]).date()).days
                decay_frac = min(days_elapsed / float(condor["dte"]), 1.0)
                est_pnl = credit * decay_frac * 0.5  # simplified theta decay

            condor["pnl"] = round(est_pnl, 2)

            # Check profit target
            if est_pnl >= condor["profit_target"]:
                close_reason = f"Profit target hit (est P&L ${est_pnl:.2f})"

            # Check loss limit (200% of credit = max_loss)
            if est_pnl <= -max_loss:
                close_reason = f"Loss limit hit (est P&L ${est_pnl:.2f})"

        if close_reason:
            condor["status"] = "CLOSED"
            condor["close_reason"] = close_reason
            condor["close_date"] = today.isoformat()
            condor["dte_remaining"] = dte_remaining
            closed.append(condor)
            append_trade_log(config.LOG_FILE, {"action": "CONDOR_CLOSE", **condor})
            logger.info("CONDOR CLOSED: %s  P&L=$%.2f", close_reason, condor.get("pnl", 0))
        else:
            condor["dte_remaining"] = dte_remaining
            remaining.append(condor)

    state["open_condors"] = remaining
    save_state(config.CONDOR_FILE, state)

    return closed


def get_condor_status() -> dict:
    """Return current condor state summary."""
    state = load_state(config.CONDOR_FILE, {"open_condors": [], "all_condors": []})
    all_condors = state.get("all_condors", [])
    open_condors = state.get("open_condors", [])

    closed = [c for c in all_condors if c.get("status") == "CLOSED"]
    wins   = [c for c in closed if float(c.get("pnl", 0)) > 0]
    total_pnl = sum(float(c.get("pnl", 0)) for c in closed)

    return {
        "open_count":  len(open_condors),
        "closed_count": len(closed),
        "win_rate":    round(len(wins) / len(closed) * 100, 1) if closed else 0,
        "total_pnl":   round(total_pnl, 2),
        "open_condors": open_condors,
    }
