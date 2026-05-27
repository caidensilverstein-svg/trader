"""
Main system orchestrator.

Coordinates all four strategies on a schedule:
  - Monday 9:00 AM ET  : Weekly report + ETF rebalance check
  - Mon-Fri 9:35 AM ET : Iron condor signal check
  - Mon-Fri 4:00 PM ET : PEAD exit check (after market close)
  - 1st of month 10 AM : Full rebalance run
  - Daily              : Portfolio drawdown check / circuit breaker

Run modes:
  --once         : Run one full cycle immediately (for testing)
  --daily        : Run the daily checks (9:35 AM routine)
  --weekly       : Run the weekly checks (Monday routine)
  --monitor      : Run continuous background monitoring
"""

import argparse
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Ensure project root is on PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from core import data as mdata
from core.regime import regime_summary
from core.utils import setup_logging, load_state, save_state, now_utc
from execution.alpaca_client import AlpacaClient
from execution.order_manager import OrderManager
from strategies.etf_manager import run_etf_manager
from strategies.iron_condor import (
    open_condor_signal, check_condor_exits, get_condor_status
)
from strategies.pead_screener import (
    get_pead_candidates, open_pead_positions, check_pead_exits, get_pead_status
)
from strategies.ma_monitor import (
    open_ma_positions, check_ma_exits, get_ma_status, scan_edgar_sc_to
)
from reporting.email_reporter import (
    send_weekly_report, send_alert, send_progress_update
)
from core.equity_tracker import record_equity, get_equity_series, days_tracked

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def build_clients() -> tuple:
    """Construct and verify Alpaca client + order manager."""
    client = AlpacaClient(paper=config.PAPER_TRADING)
    if not client.verify_connection():
        raise RuntimeError("Cannot connect to Alpaca — check credentials")
    om = OrderManager(client)
    return client, om


# ---------------------------------------------------------------------------
# Daily routine (9:35 AM ET weekdays)
# ---------------------------------------------------------------------------

def run_daily(client: AlpacaClient, om: OrderManager, dry_run: bool = False) -> dict:
    """
    Full daily routine:
    1. Portfolio drawdown / circuit breaker check
    2. Iron condor signal + exit check
    3. PEAD exit check
    4. M&A exit check
    5. Optional: new PEAD entries if slots available
    """
    logger.info("=== DAILY ROUTINE === %s", now_utc())

    # 1. Circuit breaker
    dd = om.compute_portfolio_drawdown()
    cb_level = om.check_circuit_breaker(dd)
    if cb_level == "halt":
        send_alert("CIRCUIT BREAKER HALT", f"Portfolio drawdown {dd*100:.1f}% >= -20%")

    # 2. Market data
    spy_hist, vix_hist = mdata.get_spy_vix("2y")
    reg = regime_summary(spy_hist, vix_hist)
    vix    = reg["vix"]
    regime = reg["regime"]

    # 3. SPX price (approximate from SPY * 10 for strike estimation)
    spx_price = mdata.get_current_price("SPY") * 10

    # 4. Iron condor: check exits, then consider new entry
    condor_exits = check_condor_exits(spx_price, vix)
    condor_entry = None
    if cb_level not in ("halt", "reduce"):
        condor_entry = open_condor_signal(spx_price, vix, regime)

    # 5. PEAD exits
    pead_exits = check_pead_exits(om, client, dry_run=dry_run)

    # 6. PEAD entries (if slots available)
    pead_entries = []
    if cb_level not in ("halt", "reduce"):
        pead_candidates = get_pead_candidates(max_candidates=5)
        pead_entries    = open_pead_positions(pead_candidates, om, client, dry_run=dry_run)

    # 7. M&A exits and entries
    ma_exits   = check_ma_exits(om, client, dry_run=dry_run)
    ma_entries = []
    if cb_level not in ("halt", "reduce"):
        ma_entries = open_ma_positions(om, client, dry_run=dry_run)

    # Record equity snapshot for performance tracking
    acct_snap = client.get_account()
    record_equity(acct_snap["equity"], acct_snap["cash"], acct_snap["portfolio_value"])

    summary = {
        "ts":            now_utc(),
        "regime":        regime,
        "vix":           vix,
        "circuit":       cb_level,
        "drawdown":      round(dd * 100, 2),
        "equity":        round(acct_snap["equity"], 2),
        "condor_exits":  len(condor_exits),
        "condor_entry":  condor_entry is not None,
        "pead_exits":    len(pead_exits),
        "pead_entries":  len(pead_entries),
        "ma_exits":      len(ma_exits),
        "ma_entries":    len(ma_entries),
    }

    logger.info("Daily routine complete: %s", summary)
    return summary


# ---------------------------------------------------------------------------
# Weekly routine (Monday 9:00 AM ET)
# ---------------------------------------------------------------------------

def run_weekly(client: AlpacaClient, om: OrderManager, dry_run: bool = False) -> dict:
    """
    Full weekly routine:
    1. ETF rebalance check (with B-SC + regime)
    2. EDGAR scan for new M&A deals
    3. Weekly email report
    """
    logger.info("=== WEEKLY ROUTINE === %s", now_utc())

    # 1. Market data
    spy_hist, vix_hist = mdata.get_spy_vix("2y")
    reg = regime_summary(spy_hist, vix_hist)

    # 2. ETF rebalance
    etf_result = run_etf_manager(client, om, dry_run=dry_run)

    # 3. EDGAR scan
    edgar_filings = scan_edgar_sc_to(lookback_days=7)
    if edgar_filings:
        logger.info("EDGAR found %d SC-TO filings this week", len(edgar_filings))

    # 4. Gather status for report
    acct         = client.get_account()
    condor_status = get_condor_status()
    pead_status   = get_pead_status()
    ma_status     = get_ma_status()

    # Equity snapshot for today
    record_equity(acct["equity"], acct["cash"], acct["portfolio_value"])
    equity_series = get_equity_series()

    # Count recent trades
    from core.utils import read_trade_log
    from reporting.performance_tracker import compute_portfolio_metrics
    all_trades  = read_trade_log(config.LOG_FILE)
    week_trades = len([t for t in all_trades if t.get("ts", "")[:10] >= now_utc()[:10]])

    # Compute metrics if we have enough history
    perf_metrics = compute_portfolio_metrics(equity_series) if len(equity_series) >= 2 else {}

    # 5. Send weekly report
    send_weekly_report(
        regime_data=reg,
        account_data=acct,
        etf_status=etf_result,
        condor_status=condor_status,
        pead_status=pead_status,
        ma_status=ma_status,
        trade_count=week_trades,
        perf_metrics=perf_metrics,
    )

    summary = {
        "ts":             now_utc(),
        "regime":         reg["regime"],
        "vix":            reg["vix"],
        "bsc_scalar":     etf_result.get("bsc_scalar"),
        "rebal_actions":  etf_result.get("actions", {}),
        "edgar_filings":  len(edgar_filings),
        "email_sent":     True,
    }

    logger.info("Weekly routine complete: %s", summary)
    return summary


# ---------------------------------------------------------------------------
# One-shot: run everything once (for initial deployment + testing)
# ---------------------------------------------------------------------------

def run_once(dry_run: bool = False) -> dict:
    """Run a complete cycle of all strategies once."""
    logger.info("=== ONE-SHOT RUN (dry_run=%s) ===", dry_run)

    client, om = build_clients()

    # Clear stale cache
    mdata.clear_cache()

    weekly = run_weekly(client, om, dry_run=dry_run)
    daily  = run_daily(client, om, dry_run=dry_run)

    return {"weekly": weekly, "daily": daily}


# ---------------------------------------------------------------------------
# Continuous monitoring loop
# ---------------------------------------------------------------------------

def run_monitor(interval_seconds: int = 300, dry_run: bool = False):
    """
    Long-running monitoring loop.
    Checks circuit breaker every `interval_seconds`.
    Runs daily/weekly routines on schedule.
    """
    logger.info("Starting monitor loop (interval=%ds dry_run=%s)", interval_seconds, dry_run)
    client, om = build_clients()

    last_daily  = ""
    last_weekly = ""

    while True:
        try:
            now = datetime.now(timezone.utc)
            today = now.strftime("%Y-%m-%d")
            weekday = now.weekday()  # 0=Monday

            # Circuit breaker check (every interval)
            dd = om.compute_portfolio_drawdown()
            cb = om.check_circuit_breaker(dd)
            if cb == "halt":
                send_alert("CIRCUIT BREAKER", f"Portfolio drawdown {dd*100:.1f}%")

            # Daily: run once per day at/after market open (14:30 UTC = 9:30 AM ET)
            if weekday < 5 and today != last_daily and now.hour >= 14:
                mdata.clear_cache()
                run_daily(client, om, dry_run=dry_run)
                last_daily = today

            # Weekly: Monday
            monday_key = now.strftime("%Y-%W")
            if weekday == 0 and monday_key != last_weekly and now.hour >= 13:
                mdata.clear_cache()
                run_weekly(client, om, dry_run=dry_run)
                last_weekly = monday_key

            logger.debug("Monitor tick: DD=%.1f%% circuit=%s", dd * 100, cb)

        except KeyboardInterrupt:
            logger.info("Monitor stopped by user")
            break
        except Exception as exc:
            logger.error("Monitor loop error: %s", exc, exc_info=True)
            send_alert("SYSTEM ERROR", str(exc))

        time.sleep(interval_seconds)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Quant Portfolio System")
    parser.add_argument("--once",    action="store_true", help="Run one full cycle")
    parser.add_argument("--daily",   action="store_true", help="Run daily routine only")
    parser.add_argument("--weekly",  action="store_true", help="Run weekly routine only")
    parser.add_argument("--monitor", action="store_true", help="Run continuous monitor")
    parser.add_argument("--dry-run", action="store_true", help="Log trades, do not execute")
    parser.add_argument("--log",     default="INFO", help="Log level (DEBUG/INFO/WARNING)")
    args = parser.parse_args()

    setup_logging(args.log, "state/system.log")
    Path("state").mkdir(exist_ok=True)

    dry = args.dry_run

    if args.once:
        result = run_once(dry_run=dry)
        logger.info("One-shot result: %s", result)

    elif args.daily:
        client, om = build_clients()
        mdata.clear_cache()
        result = run_daily(client, om, dry_run=dry)
        logger.info("Daily result: %s", result)

    elif args.weekly:
        client, om = build_clients()
        mdata.clear_cache()
        result = run_weekly(client, om, dry_run=dry)
        logger.info("Weekly result: %s", result)

    elif args.monitor:
        run_monitor(dry_run=dry)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
