"""
Shared utility functions: logging setup, state persistence, math helpers.
"""

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


def setup_logging(level: str = "INFO", logfile: Optional[str] = None) -> None:
    """Configure root logger with console (and optional file) handler."""
    fmt = "%(asctime)s %(levelname)-8s %(name)-25s %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    handlers: list = [logging.StreamHandler()]
    if logfile:
        Path(logfile).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(logfile))
    logging.basicConfig(level=getattr(logging, level.upper()), format=fmt,
                        datefmt=datefmt, handlers=handlers, force=True)


def load_state(filepath: str, default: Dict) -> Dict:
    """Load JSON state file; return default if missing or corrupt."""
    try:
        with open(filepath) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return dict(default)


def save_state(filepath: str, state: Dict) -> None:
    """Persist state to JSON file atomically (write to .tmp then rename)."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    tmp = filepath + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2, default=str)
    os.replace(tmp, filepath)


def now_utc() -> str:
    """Return current UTC time as ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def pct(x: float) -> str:
    """Format a fraction as a percentage string, e.g. 0.142 -> '+14.2%'."""
    return f"{x * 100:+.2f}%"


def dollars(x: float) -> str:
    """Format as dollar amount, e.g. 1234.56 -> '$1,234.56'."""
    return f"${x:,.2f}"


def clip(value: float, lo: float, hi: float) -> float:
    """Clamp value to [lo, hi]."""
    return max(lo, min(hi, value))


def safe_divide(num: float, denom: float, default: float = 0.0) -> float:
    """Return num/denom; return default if denom is zero."""
    if denom == 0:
        return default
    return num / denom


def annualized_return(total_return: float, years: float) -> float:
    """Convert a total return fraction over `years` to annualized."""
    if years <= 0:
        return 0.0
    return (1 + total_return) ** (1.0 / years) - 1


def sharpe_ratio(returns: list, risk_free: float = 0.05) -> float:
    """
    Simple Sharpe ratio from a list of annual return fractions.
    Uses excess return / std.
    """
    import numpy as np
    arr = np.array(returns, dtype=float)
    excess = arr - (risk_free / 252)  # assumes daily returns
    if excess.std() == 0:
        return 0.0
    return float(excess.mean() / excess.std() * (252 ** 0.5))


def max_drawdown(prices: list) -> float:
    """
    Compute maximum drawdown from a list of prices.
    Returns a negative fraction, e.g. -0.148.
    """
    import numpy as np
    arr = np.array(prices, dtype=float)
    peak = arr[0]
    max_dd = 0.0
    for p in arr:
        if p > peak:
            peak = p
        dd = (p / peak) - 1.0
        if dd < max_dd:
            max_dd = dd
    return max_dd


def append_trade_log(filepath: str, record: Dict[str, Any]) -> None:
    """Append a trade record to the JSON-lines trade log."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    record["ts"] = now_utc()
    with open(filepath, "a") as f:
        f.write(json.dumps(record, default=str) + "\n")


def read_trade_log(filepath: str) -> list:
    """Read all records from the JSON-lines trade log."""
    if not Path(filepath).exists():
        return []
    records = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records
