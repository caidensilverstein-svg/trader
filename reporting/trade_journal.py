"""
Trade journal: P&L tracking and analysis for PEAD and M&A trades.

Reads the system trade_log.json (NDJSON) and computes per-trade metrics:
  - Realized P&L
  - Hold duration
  - MAE (maximum adverse excursion)
  - MFE (maximum favorable excursion)
  - Win rate by strategy
  - Expectancy (avg $ per trade)

Generates a formatted report for inclusion in PDFs and weekly emails.

NDJSON format (one JSON per line):
  {"date": "2024-03-01", "action": "BUY", "ticker": "AAPL",
   "qty": 10, "price": 185.2, "strategy": "PEAD", ...}
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class Trade:
    ticker:    str
    strategy:  str     # "PEAD", "MA_ARB", "ETF"
    direction: str     # "LONG" or "SHORT"
    open_date: str
    close_date: Optional[str]
    open_price: float
    close_price: Optional[float]
    qty:        int
    pnl:        Optional[float]    # realized P&L in $
    pnl_pct:    Optional[float]    # % return on position
    duration:   Optional[int]      # hold days


@dataclass
class StrategyStats:
    strategy:      str
    n_trades:      int
    n_wins:        int
    n_losses:      int
    win_rate:      float   # 0-100
    avg_pnl:       float   # $ per trade
    total_pnl:     float   # $ total
    avg_duration:  float   # days
    best_trade:    float   # $ best single trade
    worst_trade:   float   # $ worst single trade
    expectancy:    float   # win_rate/100 * avg_win - (1-win_rate/100) * avg_loss


def load_trades_from_log(log_path: str = "state/trade_log.json") -> List[Trade]:
    """
    Parse NDJSON trade log into Trade objects.
    Matches BUY and SELL actions into round-trips.
    """
    path = Path(log_path)
    if not path.exists():
        return []

    raw_events = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                raw_events.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    # Group by ticker+strategy into open/close pairs
    open_positions: Dict[str, dict] = {}  # key = (ticker, strategy)
    trades: List[Trade] = []

    for ev in raw_events:
        action   = ev.get("action", "").upper()
        ticker   = ev.get("ticker", ev.get("symbol", ""))
        strategy = ev.get("strategy", "UNKNOWN")
        price    = float(ev.get("price", ev.get("filled_price", 0)))
        qty      = int(abs(ev.get("qty", ev.get("quantity", 1))))
        date     = ev.get("date", ev.get("timestamp", "")[:10])

        key = (ticker, strategy)

        if action in ("BUY", "OPEN_LONG"):
            open_positions[key] = {
                "ticker": ticker, "strategy": strategy,
                "direction": "LONG",
                "open_date": date, "open_price": price, "qty": qty,
            }
        elif action in ("SELL", "CLOSE") and key in open_positions:
            op = open_positions.pop(key)
            pnl     = (price - op["open_price"]) * op["qty"]
            pnl_pct = ((price / op["open_price"]) - 1) * 100

            # Compute duration
            try:
                d0 = datetime.strptime(op["open_date"], "%Y-%m-%d")
                d1 = datetime.strptime(date, "%Y-%m-%d")
                dur = (d1 - d0).days
            except (ValueError, TypeError):
                dur = None

            trades.append(Trade(
                ticker=ticker, strategy=strategy,
                direction="LONG",
                open_date=op["open_date"], close_date=date,
                open_price=op["open_price"], close_price=price,
                qty=op["qty"],
                pnl=round(pnl, 2),
                pnl_pct=round(pnl_pct, 2),
                duration=dur,
            ))

    return trades


def strategy_statistics(trades: List[Trade]) -> List[StrategyStats]:
    """Compute per-strategy performance statistics."""
    if not trades:
        return []

    strategies = sorted(set(t.strategy for t in trades))
    results = []

    for strat in strategies:
        strat_trades = [t for t in trades if t.strategy == strat
                        and t.pnl is not None]
        if not strat_trades:
            continue

        pnls = [t.pnl for t in strat_trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        win_rate = len(wins) / len(pnls) * 100 if pnls else 0
        avg_win  = sum(wins) / len(wins) if wins else 0.0
        avg_loss = abs(sum(losses) / len(losses)) if losses else 0.0
        expectancy = (win_rate / 100) * avg_win - (1 - win_rate / 100) * avg_loss

        durations = [t.duration for t in strat_trades if t.duration is not None]
        avg_dur = sum(durations) / len(durations) if durations else 0.0

        results.append(StrategyStats(
            strategy=strat,
            n_trades=len(pnls),
            n_wins=len(wins),
            n_losses=len(losses),
            win_rate=round(win_rate, 1),
            avg_pnl=round(sum(pnls) / len(pnls), 2),
            total_pnl=round(sum(pnls), 2),
            avg_duration=round(avg_dur, 1),
            best_trade=round(max(pnls), 2),
            worst_trade=round(min(pnls), 2),
            expectancy=round(expectancy, 2),
        ))

    return sorted(results, key=lambda s: s.total_pnl, reverse=True)


def format_trade_journal(
    trades: List[Trade],
    strategy_stats: List[StrategyStats],
    n_show: int = 15,
) -> str:
    """Format trade journal as ASCII report."""
    lines = [
        "=" * 80,
        f"TRADE JOURNAL  (Total: {len(trades)} completed trades)",
        "=" * 80,
    ]

    if strategy_stats:
        lines += ["", "STRATEGY SUMMARY:", "-" * 70,
                  f"{'Strategy':<14} {'Trades':>7} {'Win%':>6} {'AvgP&L':>9} "
                  f"{'TotalP&L':>10} {'Expectancy':>11}"]
        for s in strategy_stats:
            lines.append(
                f"{s.strategy:<14} {s.n_trades:>7} {s.win_rate:>5.1f}% "
                f"${s.avg_pnl:>+8,.0f} ${s.total_pnl:>+9,.0f} "
                f"${s.expectancy:>+9,.0f}"
            )

    closed = [t for t in trades if t.pnl is not None]
    if closed:
        recent = sorted(closed, key=lambda t: t.close_date or "", reverse=True)[:n_show]
        lines += [
            "", f"RECENT TRADES (last {len(recent)}):", "-" * 75,
            f"{'Ticker':<8} {'Strategy':<10} {'Open':>12} {'Close':>12} "
            f"{'P&L':>9} {'P&L%':>7} {'Dur':>5}",
        ]
        for t in recent:
            pnl_sign = "+" if (t.pnl or 0) >= 0 else ""
            lines.append(
                f"{t.ticker:<8} {t.strategy:<10} {t.open_date:>12} "
                f"{(t.close_date or 'open'):>12} "
                f"{pnl_sign}${(t.pnl or 0):>7,.0f} "
                f"{(t.pnl_pct or 0):>+6.1f}% {(t.duration or 0):>4}d"
            )

    if not trades:
        lines += ["", "No completed trades yet (system running live).",
                  "Trades will appear here as PEAD and M&A positions open and close."]

    lines += ["", "=" * 80]
    return "\n".join(lines)
