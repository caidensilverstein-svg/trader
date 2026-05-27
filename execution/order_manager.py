"""
Order manager: translate strategy signals into Alpaca orders.

Handles:
  - Notional sizing (convert target weight to dollar amount)
  - Pre-trade risk checks (max loss per trade, circuit breaker)
  - Order retry logic (2 retries, 5s delay)
  - Trade log persistence
"""

import logging
import time
from typing import Dict, List, Optional, Tuple

import config
from core.utils import append_trade_log, load_state, save_state, now_utc
from execution.alpaca_client import AlpacaClient

logger = logging.getLogger(__name__)

MAX_RETRIES = 2
RETRY_DELAY = 5  # seconds


def _retry_order(fn, *args, **kwargs) -> Optional[str]:
    """Call fn(*args) up to MAX_RETRIES times before giving up."""
    for attempt in range(1 + MAX_RETRIES):
        result = fn(*args, **kwargs)
        if result is not None:
            return result
        if attempt < MAX_RETRIES:
            logger.warning("Order attempt %d failed, retrying in %ds…", attempt + 1, RETRY_DELAY)
            time.sleep(RETRY_DELAY)
    return None


class OrderManager:
    """
    High-level order management: risk checks + execution + logging.
    """

    def __init__(self, client: AlpacaClient):
        self.client = client
        self._circuit_open = False  # True when halting all trades

    # ------------------------------------------------------------------
    # Circuit breaker
    # ------------------------------------------------------------------

    def check_circuit_breaker(self, portfolio_dd: float) -> str:
        """
        Check portfolio drawdown against circuit breaker levels.

        Parameters
        ----------
        portfolio_dd : Fraction drawdown from peak (negative, e.g. -0.18)

        Returns
        -------
        str : 'ok', 'review', 'reduce', 'halt'
        """
        if portfolio_dd <= config.CIRCUIT_HALT_DD:
            self._circuit_open = True
            logger.critical("CIRCUIT BREAKER: HALT — portfolio DD %.1f%%", portfolio_dd * 100)
            return "halt"
        elif portfolio_dd <= config.CIRCUIT_REDUCE_DD:
            logger.warning("CIRCUIT BREAKER: REDUCE — portfolio DD %.1f%%", portfolio_dd * 100)
            return "reduce"
        elif portfolio_dd <= config.CIRCUIT_REVIEW_DD:
            logger.info("CIRCUIT BREAKER: REVIEW — portfolio DD %.1f%%", portfolio_dd * 100)
            return "review"
        self._circuit_open = False
        return "ok"

    def is_trading_halted(self) -> bool:
        return self._circuit_open

    # ------------------------------------------------------------------
    # Core order methods
    # ------------------------------------------------------------------

    def buy_notional(
        self,
        ticker: str,
        notional: float,
        strategy: str,
        reason: str = "",
        max_loss: float = config.MAX_LOSS_PER_TRADE * config.TOTAL_CAPITAL,
    ) -> Optional[str]:
        """
        Buy `notional` dollars of `ticker`.
        Applies risk check: notional * 0.15 (assumed 15% max loss) <= max_loss.
        Logs the trade.
        """
        if self._circuit_open:
            logger.warning("Circuit open — skipping BUY %s", ticker)
            return None

        # Sanity: do not risk more than max_loss
        implied_max_loss = notional * 0.15
        if implied_max_loss > max_loss and max_loss > 0:
            notional = max_loss / 0.15
            logger.warning("Notional capped at $%.2f (max_loss $%.2f)", notional, max_loss)

        if notional < 10:
            logger.info("Notional $%.2f too small, skipping %s", notional, ticker)
            return None

        order_id = _retry_order(self.client.market_buy, ticker, notional)
        record = {
            "action": "BUY", "ticker": ticker, "notional": round(notional, 2),
            "strategy": strategy, "reason": reason, "order_id": order_id,
        }
        append_trade_log(config.LOG_FILE, record)
        return order_id

    def sell_qty(
        self,
        ticker: str,
        qty: float,
        strategy: str,
        reason: str = "",
    ) -> Optional[str]:
        """Sell `qty` shares of `ticker`."""
        if qty <= 0:
            return None
        order_id = _retry_order(self.client.market_sell, ticker, qty)
        record = {
            "action": "SELL", "ticker": ticker, "qty": round(qty, 4),
            "strategy": strategy, "reason": reason, "order_id": order_id,
        }
        append_trade_log(config.LOG_FILE, record)
        return order_id

    def close(self, ticker: str, strategy: str, reason: str = "") -> Optional[str]:
        """Close entire position."""
        order_id = _retry_order(self.client.close_position, ticker)
        record = {
            "action": "CLOSE", "ticker": ticker,
            "strategy": strategy, "reason": reason, "order_id": order_id,
        }
        append_trade_log(config.LOG_FILE, record)
        return order_id

    # ------------------------------------------------------------------
    # Portfolio drawdown helper
    # ------------------------------------------------------------------

    def compute_portfolio_drawdown(self) -> float:
        """
        Compute current drawdown from peak using Alpaca account history.
        Returns fraction (negative number).
        Simple approach: compare current equity to HIGH_WATER_MARK in state.
        """
        state = load_state("state/portfolio_state.json", {"hwm": 0.0})
        acct = self.client.get_account()
        equity = acct["equity"]

        hwm = max(float(state.get("hwm", 0.0)), equity)
        state["hwm"] = hwm
        save_state("state/portfolio_state.json", state)

        if hwm == 0:
            return 0.0
        return (equity / hwm) - 1.0
