"""
Backtest runner script.

Usage:
    python3 scripts/run_backtest.py [--start YYYY-MM-DD] [--end YYYY-MM-DD]
                                    [--freq M|W|Q] [--costs 0.001]

Example:
    python3 scripts/run_backtest.py --start 2022-03-01 --freq M
"""

import argparse
import logging
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.utils import setup_logging
from backtest.engine import run_backtest, format_backtest_report


def main():
    parser = argparse.ArgumentParser(description="Run ETF sleeve backtest")
    parser.add_argument("--start", default="2022-03-01")
    parser.add_argument("--end",   default=None)
    parser.add_argument("--freq",  default="M", choices=["M", "W", "Q"])
    parser.add_argument("--capital", type=float, default=100_000)
    args = parser.parse_args()

    setup_logging("INFO")
    logger = logging.getLogger(__name__)

    logger.info("Starting backtest %s to %s, freq=%s", args.start, args.end or "today", args.freq)

    result = run_backtest(
        start=args.start,
        end=args.end,
        initial_capital=args.capital,
        rebalance_freq=args.freq,
    )

    print(format_backtest_report(result))

    if "equity_curve" in result:
        eq = result["equity_curve"]
        final = eq.iloc[-1]
        gain  = final - args.capital
        print(f"\nFinal equity: ${final:,.2f}  (gain/loss: ${gain:+,.2f})")
        print(f"Rebalances executed: {result['summary']['rebalance_count']}")

        # Show recent equity
        print("\nLast 5 equity readings:")
        for date, val in list(result["equity_curve"].items())[-5:]:
            print(f"  {date}  ${val:>12,.2f}")


if __name__ == "__main__":
    main()
