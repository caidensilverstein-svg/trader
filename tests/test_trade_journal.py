"""Unit tests for reporting/trade_journal.py."""

import sys, json, tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from reporting.trade_journal import (
    load_trades_from_log,
    strategy_statistics,
    format_trade_journal,
    Trade,
    StrategyStats,
)


def _write_log(events: list, path: str):
    with open(path, "w") as f:
        for ev in events:
            f.write(json.dumps(ev) + "\n")


class TestLoadTradesFromLog:

    def test_empty_file_returns_empty(self):
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            f.write("")
            fname = f.name
        trades = load_trades_from_log(fname)
        assert trades == []

    def test_nonexistent_file_returns_empty(self):
        trades = load_trades_from_log("/tmp/nonexistent_trade_log_xyz.json")
        assert trades == []

    def test_single_round_trip_parsed(self):
        events = [
            {"action": "BUY",  "ticker": "AAPL", "strategy": "PEAD",
             "price": 180.0, "qty": 10, "date": "2024-01-10"},
            {"action": "SELL", "ticker": "AAPL", "strategy": "PEAD",
             "price": 190.0, "qty": 10, "date": "2024-01-20"},
        ]
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            for ev in events:
                f.write(json.dumps(ev) + "\n")
            fname = f.name
        trades = load_trades_from_log(fname)
        assert len(trades) == 1
        assert trades[0].ticker == "AAPL"
        assert trades[0].strategy == "PEAD"

    def test_pnl_calculated_correctly(self):
        events = [
            {"action": "BUY",  "ticker": "AAPL", "strategy": "PEAD",
             "price": 100.0, "qty": 10, "date": "2024-01-01"},
            {"action": "SELL", "ticker": "AAPL", "strategy": "PEAD",
             "price": 110.0, "qty": 10, "date": "2024-01-15"},
        ]
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            for ev in events:
                f.write(json.dumps(ev) + "\n")
            fname = f.name
        trades = load_trades_from_log(fname)
        assert len(trades) == 1
        assert abs(trades[0].pnl - 100.0) < 0.01

    def test_pnl_pct_calculated_correctly(self):
        events = [
            {"action": "BUY",  "ticker": "MSFT", "strategy": "MA_ARB",
             "price": 200.0, "qty": 5, "date": "2024-02-01"},
            {"action": "SELL", "ticker": "MSFT", "strategy": "MA_ARB",
             "price": 210.0, "qty": 5, "date": "2024-02-15"},
        ]
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            for ev in events:
                f.write(json.dumps(ev) + "\n")
            fname = f.name
        trades = load_trades_from_log(fname)
        assert len(trades) == 1
        assert abs(trades[0].pnl_pct - 5.0) < 0.01

    def test_duration_calculated_correctly(self):
        events = [
            {"action": "BUY",  "ticker": "TSLA", "strategy": "PEAD",
             "price": 300.0, "qty": 3, "date": "2024-03-01"},
            {"action": "SELL", "ticker": "TSLA", "strategy": "PEAD",
             "price": 310.0, "qty": 3, "date": "2024-03-11"},
        ]
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            for ev in events:
                f.write(json.dumps(ev) + "\n")
            fname = f.name
        trades = load_trades_from_log(fname)
        assert trades[0].duration == 10

    def test_multiple_tickers_independent(self):
        events = [
            {"action": "BUY",  "ticker": "A", "strategy": "PEAD",
             "price": 100.0, "qty": 5, "date": "2024-01-01"},
            {"action": "BUY",  "ticker": "B", "strategy": "PEAD",
             "price": 50.0, "qty": 10, "date": "2024-01-01"},
            {"action": "SELL", "ticker": "A", "strategy": "PEAD",
             "price": 105.0, "qty": 5, "date": "2024-01-10"},
            {"action": "SELL", "ticker": "B", "strategy": "PEAD",
             "price": 48.0, "qty": 10, "date": "2024-01-10"},
        ]
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            for ev in events:
                f.write(json.dumps(ev) + "\n")
            fname = f.name
        trades = load_trades_from_log(fname)
        assert len(trades) == 2

    def test_loss_trade_negative_pnl(self):
        events = [
            {"action": "BUY",  "ticker": "XYZ", "strategy": "PEAD",
             "price": 100.0, "qty": 10, "date": "2024-04-01"},
            {"action": "SELL", "ticker": "XYZ", "strategy": "PEAD",
             "price": 90.0, "qty": 10, "date": "2024-04-15"},
        ]
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            for ev in events:
                f.write(json.dumps(ev) + "\n")
            fname = f.name
        trades = load_trades_from_log(fname)
        assert trades[0].pnl < 0


class TestStrategyStatistics:

    def _make_trades(self):
        return [
            Trade("A", "PEAD", "LONG", "2024-01-01", "2024-01-10",
                  100, 110, 10, 100.0, 10.0, 9),
            Trade("B", "PEAD", "LONG", "2024-01-05", "2024-01-20",
                  100, 90, 10, -100.0, -10.0, 15),
            Trade("C", "MA_ARB", "LONG", "2024-02-01", "2024-02-10",
                  200, 205, 5, 25.0, 2.5, 9),
        ]

    def test_returns_one_stat_per_strategy(self):
        stats = strategy_statistics(self._make_trades())
        names = {s.strategy for s in stats}
        assert "PEAD" in names
        assert "MA_ARB" in names

    def test_win_rate_pead_50(self):
        stats = strategy_statistics(self._make_trades())
        pead = next(s for s in stats if s.strategy == "PEAD")
        assert abs(pead.win_rate - 50.0) < 0.01

    def test_total_pnl_correct(self):
        stats = strategy_statistics(self._make_trades())
        pead = next(s for s in stats if s.strategy == "PEAD")
        assert abs(pead.total_pnl - 0.0) < 0.01

    def test_ma_arb_100_win_rate(self):
        stats = strategy_statistics(self._make_trades())
        ma = next(s for s in stats if s.strategy == "MA_ARB")
        assert ma.win_rate == 100.0

    def test_empty_returns_empty(self):
        assert strategy_statistics([]) == []

    def test_sorted_by_total_pnl_descending(self):
        stats = strategy_statistics(self._make_trades())
        pnls = [s.total_pnl for s in stats]
        assert pnls == sorted(pnls, reverse=True)


class TestFormatTradeJournal:

    def _make_stats(self):
        return [StrategyStats("PEAD", 10, 6, 4, 60.0, 150.0, 1500.0, 8.5, 500.0, -200.0, 200.0)]

    def test_contains_header(self):
        r = format_trade_journal([], [], n_show=5)
        assert "TRADE JOURNAL" in r

    def test_empty_trades_shows_no_trades_message(self):
        r = format_trade_journal([], [])
        assert "No completed trades" in r

    def test_strategy_stats_shown(self):
        r = format_trade_journal([], self._make_stats())
        assert "PEAD" in r
        assert "60.0" in r

    def test_contains_win_rate(self):
        r = format_trade_journal([], self._make_stats())
        assert "%" in r
