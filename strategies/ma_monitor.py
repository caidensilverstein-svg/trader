"""
M&A Arbitrage Monitor.

Tracks announced cash acquisition deals and manages long equity positions
in target companies.

Data source: Manual entry + SEC EDGAR monitoring.
Execution: Alpaca long equity positions in target stock.

Strategy:
  - Cash deals only (no stock-for-stock currency risk)
  - Deal size $500M - $10B
  - Buy within 2-3 days of announcement
  - Exit when stock reaches 95% of deal price OR deal fails/times out
  - Position size: $2,500 per deal, max 2 simultaneous

Note on deal sourcing: This implementation provides a manual deal
entry system + a basic EDGAR SC-TO filing monitor. Real-time deal
flow requires Bloomberg/Refinitiv in production.
"""

import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional

import requests

import config
from core.utils import load_state, save_state, now_utc, append_trade_log
from execution.alpaca_client import AlpacaClient
from execution.order_manager import OrderManager

logger = logging.getLogger(__name__)

EDGAR_BASE = "https://efts.sec.gov/LATEST/search-index?q=%22SC+TO%22&dateRange=custom&startdt={}&enddt={}&hits.hits.total.value=true&hits.hits._source.period_of_report=true"


# ---------------------------------------------------------------------------
# Deal registry
# ---------------------------------------------------------------------------

def add_deal(
    target_ticker: str,
    acquirer: str,
    deal_price: float,
    deal_type: str = "CASH",
    expected_close_days: int = 90,
) -> dict:
    """
    Manually register an announced M&A deal.

    Parameters
    ----------
    target_ticker       : Stock ticker of acquisition target
    acquirer            : Name of acquirer
    deal_price          : Announced cash deal price per share
    deal_type           : 'CASH' (only type we trade)
    expected_close_days : Expected days to close

    Returns
    -------
    dict : Deal record
    """
    if deal_type != "CASH":
        logger.warning("Only CASH deals supported; got %s for %s", deal_type, target_ticker)
        return {}

    state = load_state(config.MA_FILE, {"deals": {}, "closed_deals": []})
    if target_ticker in state.get("deals", {}):
        logger.info("Deal for %s already registered", target_ticker)
        return state["deals"][target_ticker]

    deal = {
        "ticker":              target_ticker,
        "acquirer":            acquirer,
        "deal_price":          deal_price,
        "deal_type":           deal_type,
        "announced":           now_utc()[:10],
        "expected_close_days": expected_close_days,
        "deadline":            (datetime.now(timezone.utc) + timedelta(days=expected_close_days)).date().isoformat(),
        "status":              "ANNOUNCED",
        "position_size":       config.MA_POSITION,
        "entry_price":         None,
        "order_id":            None,
    }

    state.setdefault("deals", {})[target_ticker] = deal
    save_state(config.MA_FILE, state)
    logger.info("M&A deal registered: %s -> %s at $%.2f", target_ticker, acquirer, deal_price)
    return deal


def get_spread(ticker: str) -> Optional[float]:
    """
    Compute current deal spread for a registered deal.
    spread = (deal_price - current_price) / current_price
    Returns None if not found or no current price.
    """
    from core import data as mdata
    state = load_state(config.MA_FILE, {"deals": {}})
    deal = state.get("deals", {}).get(ticker)
    if not deal:
        return None

    try:
        current_price = mdata.get_current_price(ticker)
        spread = (deal["deal_price"] - current_price) / current_price
        return round(spread, 4)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Position management
# ---------------------------------------------------------------------------

def open_ma_positions(
    om: OrderManager,
    client: AlpacaClient,
    dry_run: bool = False,
) -> List[Dict]:
    """
    Open M&A arb positions for deals that meet entry criteria.

    Entry criteria:
    - Deal announced within last 3 days
    - Current spread between 1% and 15%
    - Deal has not been entered yet
    """
    from core import data as mdata

    state = load_state(config.MA_FILE, {"deals": {}, "closed_deals": []})
    deals = state.get("deals", {})
    opened = []

    # Count open positions
    open_count = sum(1 for d in deals.values()
                     if d.get("status") == "ENTERED")
    if open_count >= 2:
        logger.info("M&A: max 2 open positions reached")
        return []

    if om.is_trading_halted():
        return []

    for ticker, deal in deals.items():
        if deal.get("status") != "ANNOUNCED":
            continue

        # Spread check
        try:
            current_price = mdata.get_current_price(ticker)
        except Exception:
            continue

        spread = (deal["deal_price"] - current_price) / current_price
        if spread < config.MA_SPREAD_MIN or spread > config.MA_SPREAD_MAX:
            logger.info("M&A %s: spread %.1f%% outside range, skipping",
                        ticker, spread * 100)
            continue

        logger.info("M&A ENTRY: %s  deal=$%.2f  current=$%.2f  spread=%.1f%%",
                    ticker, deal["deal_price"], current_price, spread * 100)

        order_id = None
        if not dry_run:
            order_id = om.buy_notional(
                ticker, config.MA_POSITION, "MA_ARB",
                f"deal={deal['acquirer']} price={deal['deal_price']} spread={spread:.1%}",
                max_loss=config.MAX_LOSS_MA,
            )

        deal["status"]      = "ENTERED"
        deal["entry_price"] = round(current_price, 2)
        deal["order_id"]    = order_id or "dry_run"
        deal["spread_at_entry"] = round(spread, 4)
        opened.append({"ticker": ticker, **deal})
        open_count += 1

        if open_count >= 2:
            break

    state["deals"] = deals
    save_state(config.MA_FILE, state)
    return opened


def check_ma_exits(
    om: OrderManager,
    client: AlpacaClient,
    dry_run: bool = False,
) -> List[Dict]:
    """
    Check open M&A positions for exit signals.

    Exit rules:
    - Current price >= 95% of deal price → take profit
    - Deal deadline passed → close (deal fell through)
    - Position down >10% → stop loss (deal likely failed)
    """
    from core import data as mdata

    state = load_state(config.MA_FILE, {"deals": {}, "closed_deals": []})
    deals = state.get("deals", {})
    closed_today = []
    today = datetime.now(timezone.utc).date()

    for ticker, deal in list(deals.items()):
        if deal.get("status") != "ENTERED":
            continue

        try:
            current_price = mdata.get_current_price(ticker)
        except Exception:
            continue

        deal_price   = float(deal["deal_price"])
        entry_price  = float(deal.get("entry_price", current_price))
        pnl_pct      = (current_price / entry_price) - 1.0
        close_reason = None

        # Profit target: stock near deal price
        if current_price >= deal_price * config.MA_EXIT_PCT:
            close_reason = f"Price target ({current_price:.2f} >= {deal_price*config.MA_EXIT_PCT:.2f})"

        # Deadline
        deadline = datetime.fromisoformat(deal["deadline"]).date()
        if today > deadline:
            close_reason = f"Deadline passed ({deadline})"

        # Stop loss: deal likely fell through
        if pnl_pct <= -0.10:
            close_reason = f"Stop loss ({pnl_pct*100:.1f}%) — deal may have failed"

        if close_reason:
            logger.info("M&A EXIT: %s  reason=%s  pnl=%.1f%%", ticker, close_reason, pnl_pct * 100)
            order_id = None
            if not dry_run:
                order_id = om.close(ticker, "MA_ARB", close_reason)

            closed_record = {
                "ticker":       ticker,
                "acquirer":     deal.get("acquirer"),
                "close_reason": close_reason,
                "pnl_pct":      round(pnl_pct * 100, 2),
                "order_id":     order_id or "dry_run",
                "closed_at":    now_utc(),
            }
            closed_today.append(closed_record)
            state.setdefault("closed_deals", []).append(closed_record)
            append_trade_log(config.LOG_FILE, {"action": "MA_CLOSE", **closed_record})

            deal["status"] = "CLOSED"
            deal["pnl_pct"] = round(pnl_pct * 100, 2)

    state["deals"] = deals
    save_state(config.MA_FILE, state)
    return closed_today


def get_ma_status() -> dict:
    """Return M&A strategy summary."""
    state = load_state(config.MA_FILE, {"deals": {}, "closed_deals": []})
    deals  = state.get("deals", {})
    closed = state.get("closed_deals", [])
    open_deals = [t for t, d in deals.items() if d.get("status") == "ENTERED"]
    wins = [c for c in closed if float(c.get("pnl_pct", 0)) > 0]

    return {
        "open_count":   len(open_deals),
        "closed_count": len(closed),
        "win_rate":     round(len(wins) / len(closed) * 100, 1) if closed else 0,
        "open_deals":   open_deals,
    }


# ---------------------------------------------------------------------------
# EDGAR filing monitor (basic)
# ---------------------------------------------------------------------------

def scan_edgar_sc_to(lookback_days: int = 3) -> List[Dict]:
    """
    Query SEC EDGAR for recent SC-TO (tender offer) filings.

    Returns list of basic filing info dicts.
    NOTE: This is informational only — human review required before
    entering a position.
    """
    today = datetime.now(timezone.utc).date()
    start = (today - timedelta(days=lookback_days)).isoformat()
    url   = (
        f"https://efts.sec.gov/LATEST/search-index?q=%22SC+TO%22"
        f"&dateRange=custom&startdt={start}&enddt={today.isoformat()}"
    )
    try:
        resp = requests.get(url, timeout=10,
                            headers={"User-Agent": "QuantBot/1.0 research@example.com"})
        if resp.status_code != 200:
            return []
        hits = resp.json().get("hits", {}).get("hits", [])
        results = []
        for h in hits[:10]:
            src = h.get("_source", {})
            results.append({
                "company":   src.get("entity_name", "Unknown"),
                "form":      src.get("form_type", ""),
                "filed":     src.get("file_date", ""),
                "cik":       src.get("entity_id", ""),
            })
        logger.info("EDGAR scan: found %d SC-TO filings in last %d days", len(results), lookback_days)
        return results
    except Exception as exc:
        logger.warning("EDGAR scan failed: %s", exc)
        return []
