"""
Initial ETF deployment script.

Run once to buy all 5 ETFs at target weights on a fresh Alpaca account.
Orders are market orders — they queue if market is closed and execute at open.

Usage: python3 scripts/deploy_etfs.py [--dry-run]
"""

import sys
import argparse
import logging
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from core import data as mdata
from core.regime import regime_summary
from core.utils import setup_logging, now_utc
from execution.alpaca_client import AlpacaClient
from execution.order_manager import OrderManager
from strategies.etf_manager import run_etf_manager


def main():
    parser = argparse.ArgumentParser(description="Deploy ETF positions")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    setup_logging("INFO", "state/deploy.log")
    logger = logging.getLogger(__name__)
    Path("state").mkdir(exist_ok=True)

    client = AlpacaClient(paper=config.PAPER_TRADING)
    if not client.verify_connection():
        logger.error("Cannot connect to Alpaca")
        sys.exit(1)

    acct = client.get_account()
    logger.info("Account: equity=$%.2f  cash=$%.2f", acct["equity"], acct["cash"])

    positions = client.get_positions()
    if positions:
        logger.info("Existing positions: %s", list(positions.keys()))
    else:
        logger.info("No existing positions -- fresh deployment")

    mdata.clear_cache()
    result = run_etf_manager(client, OrderManager(client), dry_run=args.dry_run)

    logger.info("Deployment result:")
    logger.info("  Regime:     %s", result["regime"])
    logger.info("  B-SC:       %.2fx (QMOM at %.1f%%)", result["bsc_scalar"], result["eff_qmom_wt"])
    logger.info("  VIX:        %.1f", result["vix"])
    logger.info("  Actions:    %s", result["actions"])

    if args.dry_run:
        logger.info("DRY RUN -- no orders submitted")
    else:
        logger.info("Orders submitted -- will execute at next market open if market is closed")


if __name__ == "__main__":
    main()
