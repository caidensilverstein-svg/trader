"""
Portfolio status dashboard.

Prints a complete system status snapshot to the console:
  - Account summary and positions
  - Regime and market conditions
  - Strategy status (condor, PEAD, M&A)
  - Factor attribution (YTD)
  - Performance metrics (if equity history available)

Usage:
    python3 scripts/portfolio_status.py
"""

import sys
from datetime import datetime, timezone
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from execution.alpaca_client import AlpacaClient
from core import data as mdata
from core.regime import regime_summary
from core.equity_tracker import get_equity_series, days_tracked
from core.utils import load_state
from reporting.performance_tracker import compute_portfolio_metrics
from backtest.factor_attribution import compute_attribution, format_attribution_report
from strategies.iron_condor import get_condor_status
from strategies.pead_screener import get_pead_status
from strategies.ma_monitor import get_ma_status


def print_section(title: str):
    print()
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)


def main():
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    print(f"\nQUANT PORTFOLIO STATUS -- {now}")
    print("=" * 60)

    # Account
    try:
        client = AlpacaClient(paper=True)
        acct = client.get_account()
        positions = client.get_positions()
    except Exception as e:
        print(f"ERROR connecting to Alpaca: {e}")
        return

    print_section("ACCOUNT SUMMARY")
    print(f"  Portfolio Value  : ${acct.get('portfolio_value', 0):>12,.2f}")
    print(f"  Equity           : ${acct.get('equity', 0):>12,.2f}")
    print(f"  Cash             : ${acct.get('cash', 0):>12,.2f}")
    print(f"  Return vs $100k  : {((acct.get('equity', 100000) / 100000) - 1) * 100:>+.2f}%")

    # Positions
    if positions:
        print_section("OPEN POSITIONS")
        for ticker, pos in sorted(positions.items()):
            mv  = pos.get("market_value", 0)
            pnl = pos.get("unrealized_pl", 0)
            pct_pnl = (pnl / (mv - pnl) * 100) if mv != pnl else 0
            print(f"  {ticker:<6}  qty={pos.get('qty', ''):<8}  "
                  f"mv=${mv:>10,.2f}  pnl=${pnl:>+8,.2f} ({pct_pnl:+.1f}%)")
    else:
        print_section("OPEN POSITIONS")
        print("  (none -- orders may be pending for next market open)")

    # Market regime
    print_section("MARKET CONDITIONS")
    try:
        spy_hist, vix_hist = mdata.get_spy_vix("2y")
        reg = regime_summary(spy_hist, vix_hist)
        print(f"  Regime       : {reg.get('regime', 'N/A')}")
        print(f"  SPY Price    : ${reg.get('spy_price', 0):.2f}")
        print(f"  200-Day MA   : ${reg.get('spy_ma200', 0):.2f}")
        print(f"  60d Momentum : {reg.get('spy_mom_60d', 0):+.2f}%")
        print(f"  VIX          : {reg.get('vix', 0):.2f}")
        print(f"  DD from Peak : {reg.get('dd_from_peak', 0):+.2f}%")
    except Exception as e:
        print(f"  ERROR: {e}")

    # Strategy status
    print_section("IRON CONDOR (Paper Signal)")
    cs = get_condor_status()
    print(f"  Open signals : {cs.get('open_count', 0)}")
    print(f"  Closed       : {cs.get('closed_count', 0)}")
    for c in cs.get("open_condors", []):
        print(f"  SPX {c.get('spx_at_entry', 0):.0f}  Put {c.get('short_put', 0)}/{c.get('long_put', 0)}"
              f"  Call {c.get('short_call', 0)}/{c.get('long_call', 0)}"
              f"  {c.get('dte_remaining', 0)}DTE")

    print_section("PEAD POSITIONS")
    ps = get_pead_status()
    print(f"  Open : {ps.get('open_count', 0)} ({', '.join(ps.get('open_positions', [])) or 'none'})")
    print(f"  Closed: {ps.get('closed_count', 0)}")

    print_section("M&A ARBITRAGE")
    ms = get_ma_status()
    print(f"  Open : {ms.get('open_count', 0)} ({', '.join(ms.get('open_deals', [])) or 'none'})")
    print(f"  Closed: {ms.get('closed_count', 0)}")

    # Performance metrics
    equity_series = get_equity_series()
    if len(equity_series) >= 2:
        print_section("PORTFOLIO METRICS")
        m = compute_portfolio_metrics(equity_series)
        if "error" not in m:
            print(f"  Total Return  : {m['total_return']:>+.2f}%")
            print(f"  Ann. Return   : {m['ann_return']:>+.2f}%")
            print(f"  Volatility    : {m['ann_vol']:.2f}%")
            print(f"  Sharpe        : {m['sharpe']:.3f}")
            print(f"  Max Drawdown  : {m['max_dd']:.2f}%")
            print(f"  Calmar        : {m['calmar']:.3f}")
            print(f"  Days Tracked  : {m['n_days']}")
    else:
        print_section("PORTFOLIO METRICS")
        print(f"  {days_tracked()} day(s) of history (need >= 2 for metrics)")

    # YTD factor attribution
    print_section("FACTOR ATTRIBUTION (YTD 2025)")
    try:
        ytd_result = compute_attribution("2025-01-01")
        for t, d in sorted(ytd_result.get("per_ticker", {}).items(),
                           key=lambda x: abs(x[1]["contribution"]), reverse=True):
            print(f"  {t:<6} {d['factor']:<25} wt={d['weight']:>4.1f}%  "
                  f"ret={d['total_return']:>+6.1f}%  contrib={d['contribution']:>+5.2f}pp")
        total = ytd_result.get("total_contribution", 0)
        spy   = ytd_result.get("spy_return", 0) or 0
        print(f"  {'TOTAL':<32} {'':>5}   {'':>6}   contrib={total:>+5.2f}pp")
        print(f"  SPY YTD: {spy:+.2f}%   Edge: {total - spy:+.2f}pp")
    except Exception as e:
        print(f"  ERROR: {e}")

    # ETF pending orders
    print_section("ETF ORDERS (pending at market open)")
    try:
        from alpaca.trading.client import TradingClient
        from alpaca.trading.requests import GetOrdersRequest
        from alpaca.trading.enums import QueryOrderStatus
        tc = TradingClient(config.ALPACA_KEY, config.ALPACA_SECRET, paper=True)
        orders = tc.get_orders(GetOrdersRequest(status=QueryOrderStatus.OPEN))
        if orders:
            for o in orders:
                n = float(o.notional) if o.notional else 0
                print(f"  {o.symbol:<6}  {o.side.value:<4}  notional=${n:>10,.2f}  {o.status.value}")
        else:
            print("  (no pending orders)")
    except Exception as e:
        print(f"  ERROR: {e}")

    print()
    print("=" * 60)
    print("  Cron: daily 13:35 UTC / weekly Monday 13:00 UTC")
    print("  Repo: github.com/caidensilverstein-svg/trader")
    print("=" * 60)


if __name__ == "__main__":
    main()
