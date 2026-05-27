"""
Performance tracking and reporting.

Computes portfolio metrics from the Alpaca account + trade log:
  - Total return vs benchmark (SPY)
  - Sharpe ratio (rolling)
  - Maximum drawdown
  - Win rates per strategy
  - P&L attribution

All output is plain ASCII to avoid email/PDF rendering issues.
"""

import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import config
from core.utils import read_trade_log, load_state, now_utc

logger = logging.getLogger(__name__)


def compute_portfolio_metrics(
    equity_history: List[float],
    dates: Optional[List[str]] = None,
    risk_free: float = 0.05,
) -> Dict:
    """
    Compute standard portfolio metrics from a list of daily equity values.

    Parameters
    ----------
    equity_history : List of daily portfolio values (ascending date order)
    dates          : Optional list of date strings for context
    risk_free      : Annual risk-free rate (default 5% = current T-bill approx)

    Returns
    -------
    dict of metrics
    """
    if len(equity_history) < 2:
        return {"error": "Insufficient data (need >= 2 data points)"}

    vals = np.array(equity_history, dtype=float)
    returns = np.diff(vals) / vals[:-1]

    # Returns-based metrics
    ann_factor = 252
    total_return = (vals[-1] / vals[0]) - 1.0
    n_days = len(returns)
    ann_return = ((1 + total_return) ** (ann_factor / max(n_days, 1))) - 1

    # Sharpe
    excess = returns - (risk_free / ann_factor)
    sharpe = float(excess.mean() / excess.std() * (ann_factor ** 0.5)) if excess.std() > 0 else 0.0

    # Max drawdown
    peak = vals[0]
    max_dd = 0.0
    for v in vals:
        if v > peak:
            peak = v
        dd = (v / peak) - 1.0
        if dd < max_dd:
            max_dd = dd

    # Volatility
    ann_vol = float(returns.std() * (ann_factor ** 0.5))

    # Calmar ratio
    calmar = float(ann_return / abs(max_dd)) if max_dd != 0 else float("inf")

    return {
        "start_value":   round(float(vals[0]), 2),
        "end_value":     round(float(vals[-1]), 2),
        "total_return":  round(total_return * 100, 2),
        "ann_return":    round(ann_return * 100, 2),
        "ann_vol":       round(ann_vol * 100, 2),
        "sharpe":        round(sharpe, 3),
        "max_dd":        round(max_dd * 100, 2),
        "calmar":        round(calmar, 3),
        "n_days":        n_days,
    }


def strategy_pnl_summary(trade_log_path: str = config.LOG_FILE) -> Dict:
    """
    Summarize P&L by strategy from the trade log.

    Returns dict with per-strategy win rates and trade counts.
    """
    records = read_trade_log(trade_log_path)
    if not records:
        return {}

    by_strategy: Dict[str, dict] = {}
    for r in records:
        strategy = r.get("strategy", "UNKNOWN")
        if strategy not in by_strategy:
            by_strategy[strategy] = {"trades": 0, "buys": 0, "sells": 0}
        by_strategy[strategy]["trades"] += 1
        action = r.get("action", "")
        if action in ("BUY",):
            by_strategy[strategy]["buys"] += 1
        elif action in ("SELL", "CLOSE"):
            by_strategy[strategy]["sells"] += 1

    return by_strategy


def format_performance_report(
    account_data: dict,
    equity_history: Optional[List[float]] = None,
) -> str:
    """
    Build a text performance report for email or logging.
    All characters are ASCII-safe.
    """
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    equity = account_data.get("equity", 0)
    pv     = account_data.get("portfolio_value", equity)

    lines = [
        "=" * 72,
        "PERFORMANCE REPORT",
        f"Generated: {now}",
        "=" * 72,
        "",
        "ACCOUNT STATE",
        "-" * 40,
        f"Portfolio Value  : ${pv:>12,.2f}",
        f"Cash             : ${account_data.get('cash', 0):>12,.2f}",
        f"Buying Power     : ${account_data.get('buying_power', 0):>12,.2f}",
        f"Return vs $100k  : {((equity/100000)-1)*100:>+.2f}%",
        "",
    ]

    if equity_history and len(equity_history) >= 2:
        metrics = compute_portfolio_metrics(equity_history)
        lines += [
            "PORTFOLIO METRICS (since inception)",
            "-" * 40,
            f"Total Return     : {metrics['total_return']:>+.2f}%",
            f"Annualized       : {metrics['ann_return']:>+.2f}%",
            f"Volatility (ann) : {metrics['ann_vol']:.2f}%",
            f"Sharpe Ratio     : {metrics['sharpe']:.3f}",
            f"Max Drawdown     : {metrics['max_dd']:.2f}%",
            f"Calmar Ratio     : {metrics['calmar']:.3f}",
            f"Trading Days     : {metrics['n_days']}",
            "",
        ]

    # Trade log summary
    trades = read_trade_log(config.LOG_FILE)
    if trades:
        by_strat = strategy_pnl_summary()
        lines += [
            "TRADE LOG SUMMARY",
            "-" * 40,
        ]
        for strat, data in by_strat.items():
            lines.append(f"  {strat:<20} {data['trades']:>4} trades  "
                         f"({data['buys']} buys / {data['sells']} closes)")
        lines.append(f"  TOTAL             : {len(trades)} records")
        lines.append("")

    lines += [
        "=" * 72,
        "System: 4-layer factor + options + PEAD + M&A",
        "Target: $500+/month  |  Current state: ACTIVE",
        "=" * 72,
    ]

    return "\n".join(lines)
