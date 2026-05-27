"""
Equity history tracker.

Appends daily equity snapshots to a JSON list so that performance metrics
(Sharpe, drawdown, etc.) can be computed over real observed data rather than
a single point in time.

The file is append-only during a run; never shrunk unless manually reset.
"""

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import config

logger = logging.getLogger(__name__)

_HIST_FILE = Path(config.STATE_DIR) / "equity_history.json"


def record_equity(equity: float, cash: float, portfolio_value: float) -> None:
    """
    Append current equity to the history file.

    Only records once per calendar day (UTC) — subsequent calls on the same
    day overwrite that day's entry so we don't double-count intraday calls.
    """
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    history = _load()

    # Update or append today's snapshot
    if history and history[-1]["date"] == today:
        history[-1].update({"equity": equity, "cash": cash, "pv": portfolio_value})
        logger.debug("Updated today's equity snapshot: $%.2f", equity)
    else:
        history.append({
            "date":  today,
            "ts":    datetime.now(timezone.utc).isoformat(),
            "equity": equity,
            "cash":   cash,
            "pv":     portfolio_value,
        })
        logger.info("Recorded equity snapshot %s: $%.2f", today, equity)

    _save(history)


def get_equity_series() -> List[float]:
    """Return ordered list of daily equity values (oldest first)."""
    return [r["equity"] for r in _load()]


def get_equity_history() -> List[Dict]:
    """Return full history records."""
    return _load()


def days_tracked() -> int:
    return len(_load())


def _load() -> List[Dict]:
    try:
        if _HIST_FILE.exists():
            with open(_HIST_FILE) as f:
                return json.load(f)
    except Exception as exc:
        logger.error("Failed to load equity history: %s", exc)
    return []


def _save(history: List[Dict]) -> None:
    tmp = str(_HIST_FILE) + ".tmp"
    try:
        with open(tmp, "w") as f:
            json.dump(history, f, indent=2)
        os.replace(tmp, str(_HIST_FILE))
    except Exception as exc:
        logger.error("Failed to save equity history: %s", exc)
