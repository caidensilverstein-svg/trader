"""
Alpaca API wrapper.

Thin, tested wrapper around alpaca-py. All order placement, account
queries, and position reads go through this module.

NOTE: Alpaca paper trading does NOT support options. Iron condor signals
      are generated and logged but not executed here.
"""

import logging
import time
from typing import Dict, List, Optional, Tuple

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (
    MarketOrderRequest,
    LimitOrderRequest,
    GetOrdersRequest,
    ClosePositionRequest,
)
from alpaca.trading.enums import OrderSide, TimeInForce, QueryOrderStatus
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest

import config

logger = logging.getLogger(__name__)


class AlpacaClient:
    """
    Stateless wrapper. Each method is a single Alpaca API call.
    Raises on unrecoverable errors; logs and returns None on transient ones.
    """

    def __init__(self, paper: bool = True):
        self.paper = paper
        self._client = TradingClient(
            api_key=config.ALPACA_KEY,
            secret_key=config.ALPACA_SECRET,
            paper=paper,
        )
        self._data_client = StockHistoricalDataClient(
            api_key=config.ALPACA_KEY,
            secret_key=config.ALPACA_SECRET,
        )
        logger.info("AlpacaClient initialized (paper=%s)", paper)

    # ------------------------------------------------------------------
    # Account
    # ------------------------------------------------------------------

    def get_account(self) -> dict:
        """Return account dict with equity, cash, buying_power, etc."""
        acct = self._client.get_account()
        return {
            "equity":        float(acct.equity),
            "cash":          float(acct.cash),
            "buying_power":  float(acct.buying_power),
            "portfolio_value": float(acct.portfolio_value),
            "daytrade_count": int(acct.daytrade_count),
            "status":         str(acct.status),
        }

    # ------------------------------------------------------------------
    # Positions
    # ------------------------------------------------------------------

    def get_positions(self) -> Dict[str, dict]:
        """Return {ticker: {qty, market_value, avg_entry_price, unrealized_pl}} dict."""
        positions = {}
        for pos in self._client.get_all_positions():
            positions[pos.symbol] = {
                "qty":           float(pos.qty),
                "market_value":  float(pos.market_value),
                "avg_entry":     float(pos.avg_entry_price),
                "unrealized_pl": float(pos.unrealized_pl),
                "unrealized_plpc": float(pos.unrealized_plpc),
                "side":          str(pos.side),
            }
        return positions

    def get_position(self, ticker: str) -> Optional[dict]:
        """Return position dict for one ticker, or None if not held."""
        positions = self.get_positions()
        return positions.get(ticker)

    # ------------------------------------------------------------------
    # Orders
    # ------------------------------------------------------------------

    def market_buy(self, ticker: str, notional: float) -> Optional[str]:
        """
        Submit a market order to buy `notional` dollars of `ticker`.
        Returns order_id on success, None on failure.
        """
        try:
            req = MarketOrderRequest(
                symbol=ticker,
                notional=round(notional, 2),
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY,
            )
            order = self._client.submit_order(req)
            logger.info("BUY %s $%.2f -> order %s", ticker, notional, order.id)
            return str(order.id)
        except Exception as exc:
            logger.error("market_buy %s failed: %s", ticker, exc)
            return None

    def market_sell(self, ticker: str, qty: float) -> Optional[str]:
        """Submit a market sell order for `qty` shares."""
        try:
            req = MarketOrderRequest(
                symbol=ticker,
                qty=qty,
                side=OrderSide.SELL,
                time_in_force=TimeInForce.DAY,
            )
            order = self._client.submit_order(req)
            logger.info("SELL %s qty=%.4f -> order %s", ticker, qty, order.id)
            return str(order.id)
        except Exception as exc:
            logger.error("market_sell %s failed: %s", ticker, exc)
            return None

    def close_position(self, ticker: str) -> Optional[str]:
        """Close entire position in `ticker`. Returns order_id or None."""
        try:
            req = ClosePositionRequest()
            order = self._client.close_position(ticker, close_options=req)
            logger.info("CLOSE %s -> order %s", ticker, order.id)
            return str(order.id)
        except Exception as exc:
            logger.error("close_position %s failed: %s", ticker, exc)
            return None

    def cancel_all_orders(self) -> int:
        """Cancel all open orders. Returns count cancelled."""
        try:
            cancelled = self._client.cancel_orders()
            n = len(cancelled)
            logger.info("Cancelled %d open orders", n)
            return n
        except Exception as exc:
            logger.error("cancel_all_orders failed: %s", exc)
            return 0

    def get_open_orders(self) -> list:
        """Return list of open order dicts."""
        req = GetOrdersRequest(status=QueryOrderStatus.OPEN)
        orders = self._client.get_orders(req)
        return [
            {
                "id":     str(o.id),
                "symbol": str(o.symbol),
                "qty":    float(o.qty or 0),
                "side":   str(o.side),
                "type":   str(o.order_type),
                "status": str(o.status),
            }
            for o in orders
        ]

    # ------------------------------------------------------------------
    # Quotes
    # ------------------------------------------------------------------

    def get_latest_price(self, ticker: str) -> Optional[float]:
        """Return latest bid/ask midpoint from Alpaca data."""
        try:
            req = StockLatestQuoteRequest(symbol_or_symbols=[ticker])
            quotes = self._data_client.get_stock_latest_quote(req)
            q = quotes[ticker]
            mid = (float(q.bid_price) + float(q.ask_price)) / 2
            return mid
        except Exception as exc:
            logger.warning("get_latest_price %s failed: %s", ticker, exc)
            return None

    # ------------------------------------------------------------------
    # Checks
    # ------------------------------------------------------------------

    def is_market_open(self) -> bool:
        """Return True if US market is currently open."""
        try:
            clock = self._client.get_clock()
            return bool(clock.is_open)
        except Exception:
            return False

    def verify_connection(self) -> bool:
        """Ping the account endpoint; return True if OK."""
        try:
            acct = self.get_account()
            logger.info(
                "Connected: equity $%.2f  cash $%.2f  status %s",
                acct["equity"], acct["cash"], acct["status"],
            )
            return True
        except Exception as exc:
            logger.error("Connection failed: %s", exc)
            return False
