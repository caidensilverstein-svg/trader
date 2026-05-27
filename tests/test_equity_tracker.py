"""Unit tests for core/equity_tracker.py."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import pytest
from unittest.mock import patch

import config


@pytest.fixture(autouse=True)
def tmp_state(tmp_path, monkeypatch):
    """Redirect state directory to temp path."""
    monkeypatch.setattr(config, "STATE_DIR", str(tmp_path))
    # Re-import after patching so _HIST_FILE is resolved fresh
    import importlib
    import core.equity_tracker as et
    et._HIST_FILE = tmp_path / "equity_history.json"


from core.equity_tracker import (
    record_equity,
    get_equity_series,
    get_equity_history,
    days_tracked,
)


class TestRecordEquity:

    def test_records_single_snapshot(self):
        record_equity(100_000, 5_000, 95_000)
        assert days_tracked() == 1

    def test_series_returns_correct_value(self):
        record_equity(100_000, 5_000, 95_000)
        series = get_equity_series()
        assert len(series) == 1
        assert series[0] == 100_000

    def test_same_day_overwrites(self):
        record_equity(100_000, 5_000, 95_000)
        record_equity(101_000, 5_000, 96_000)  # same day
        assert days_tracked() == 1
        assert get_equity_series()[0] == 101_000

    def test_multiple_days_accumulate(self):
        from datetime import date, timedelta
        dates = ["2026-01-01", "2026-01-02", "2026-01-03"]
        for i, d in enumerate(dates):
            with patch("core.equity_tracker.datetime") as mock_dt:
                mock_dt.now.return_value.strftime.return_value = d
                mock_dt.now.return_value.isoformat.return_value = d + "T00:00:00+00:00"
                record_equity(100_000 + i * 1_000, 5_000, 95_000)
        assert days_tracked() == 3

    def test_history_contains_required_keys(self):
        record_equity(100_000, 5_000, 95_000)
        hist = get_equity_history()
        assert len(hist) == 1
        for key in ("date", "ts", "equity", "cash", "pv"):
            assert key in hist[0], f"Missing key: {key}"

    def test_equity_field_correct(self):
        record_equity(123_456.78, 10_000, 113_456.78)
        hist = get_equity_history()
        assert hist[0]["equity"] == 123_456.78
        assert hist[0]["cash"] == 10_000

    def test_empty_history_returns_empty_series(self):
        assert get_equity_series() == []
        assert days_tracked() == 0


class TestCircuitBreaker:
    """Circuit breaker logic is in OrderManager, tested here in isolation."""

    def _make_om(self):
        from unittest.mock import MagicMock
        from execution.order_manager import OrderManager
        mock_client = MagicMock()
        return OrderManager(mock_client)

    def test_ok_when_no_drawdown(self):
        om = self._make_om()
        assert om.check_circuit_breaker(0.0) == "ok"

    def test_ok_when_small_drawdown(self):
        om = self._make_om()
        assert om.check_circuit_breaker(-0.05) == "ok"

    def test_review_at_10pct(self):
        om = self._make_om()
        assert om.check_circuit_breaker(-0.10) == "review"

    def test_reduce_at_15pct(self):
        om = self._make_om()
        assert om.check_circuit_breaker(-0.15) == "reduce"

    def test_halt_at_20pct(self):
        om = self._make_om()
        result = om.check_circuit_breaker(-0.20)
        assert result == "halt"

    def test_halt_sets_circuit_open(self):
        om = self._make_om()
        om.check_circuit_breaker(-0.20)
        assert om.is_trading_halted() is True

    def test_ok_clears_circuit_open(self):
        om = self._make_om()
        om.check_circuit_breaker(-0.20)  # halt first
        om.check_circuit_breaker(0.0)   # should clear
        assert om.is_trading_halted() is False

    def test_circuit_breaker_thresholds_ordered(self):
        # review < reduce < halt (all negative fractions)
        assert config.CIRCUIT_REVIEW_DD > config.CIRCUIT_REDUCE_DD > config.CIRCUIT_HALT_DD
