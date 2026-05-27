"""Unit tests for strategies/ma_monitor.py."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from unittest.mock import patch, MagicMock

import config


@pytest.fixture(autouse=True)
def isolated_state(tmp_path, monkeypatch):
    """Redirect state files to temp directory."""
    monkeypatch.setattr(config, "MA_FILE",  str(tmp_path / "ma_state.json"))
    monkeypatch.setattr(config, "LOG_FILE", str(tmp_path / "trades.json"))


from strategies.ma_monitor import (
    add_deal,
    get_spread,
    get_ma_status,
    check_ma_exits,
)


class TestAddDeal:

    def test_adds_cash_deal(self):
        deal = add_deal("TGT", "BigCo", 150.0)
        assert deal["ticker"] == "TGT"
        assert deal["deal_price"] == 150.0
        assert deal["status"] == "ANNOUNCED"

    def test_rejects_non_cash_deal(self):
        deal = add_deal("TGT", "BigCo", 150.0, deal_type="STOCK")
        assert deal == {}

    def test_duplicate_ignored(self):
        add_deal("TGT", "BigCo", 150.0)
        deal2 = add_deal("TGT", "BigCo", 155.0)  # different price
        assert deal2["deal_price"] == 150.0  # returns original

    def test_deadline_set_from_expected_close(self):
        from datetime import datetime, timezone, timedelta
        deal = add_deal("TGT", "BigCo", 150.0, expected_close_days=90)
        deadline = deal["deadline"]
        assert deadline is not None
        expected = (datetime.now(timezone.utc) + timedelta(days=90)).date().isoformat()
        assert deadline == expected

    def test_position_size_from_config(self):
        deal = add_deal("TGT", "BigCo", 150.0)
        assert deal["position_size"] == config.MA_POSITION

    def test_multiple_deals_tracked_separately(self):
        add_deal("TGT1", "BigCo", 100.0)
        add_deal("TGT2", "MegaCo", 200.0)
        status = get_ma_status()
        assert status["open_count"] == 0  # neither entered yet


class TestGetSpread:

    def test_spread_positive_when_current_below_deal(self):
        add_deal("TGT", "BigCo", 150.0)
        with patch("core.data.get_current_price", return_value=140.0):
            spread = get_spread("TGT")
        assert spread is not None
        assert spread > 0.05  # ~7% spread

    def test_spread_negative_if_current_above_deal(self):
        add_deal("TGT", "BigCo", 150.0)
        with patch("core.data.get_current_price", return_value=155.0):
            spread = get_spread("TGT")
        assert spread < 0

    def test_returns_none_for_unknown_ticker(self):
        spread = get_spread("ZZZZUNKNOWN")
        assert spread is None

    def test_spread_calculation_correct(self):
        add_deal("TGT", "BigCo", 100.0)
        with patch("core.data.get_current_price", return_value=95.0):
            spread = get_spread("TGT")
        # spread = (100 - 95) / 95 = 5.26%
        assert abs(spread - 0.0526) < 0.001


class TestGetMAStatus:

    def test_empty_status(self):
        status = get_ma_status()
        assert status["open_count"] == 0
        assert status["closed_count"] == 0
        assert status["win_rate"] == 0

    def test_win_rate_zero_when_no_closed(self):
        add_deal("TGT", "BigCo", 150.0)
        status = get_ma_status()
        assert status["win_rate"] == 0


class TestCheckMAExits:

    def _enter_deal(self, ticker, deal_price, current_price):
        """Add a deal and simulate it being entered."""
        from core.utils import load_state, save_state
        add_deal(ticker, "BigCo", deal_price)
        state = load_state(config.MA_FILE, {"deals": {}})
        state["deals"][ticker]["status"] = "ENTERED"
        state["deals"][ticker]["entry_price"] = current_price
        save_state(config.MA_FILE, state)

    def test_no_exit_when_within_parameters(self):
        self._enter_deal("TGT", 150.0, 140.0)
        mock_om = MagicMock()
        mock_client = MagicMock()
        with patch("core.data.get_current_price", return_value=142.0):
            closed = check_ma_exits(mock_om, mock_client, dry_run=True)
        assert len(closed) == 0

    def test_profit_target_exit(self):
        self._enter_deal("TGT", 150.0, 140.0)
        mock_om = MagicMock()
        mock_client = MagicMock()
        # 95% of 150 = 142.5, current = 145 -> should exit
        with patch("core.data.get_current_price", return_value=145.0):
            closed = check_ma_exits(mock_om, mock_client, dry_run=True)
        assert len(closed) == 1
        assert "Price target" in closed[0]["close_reason"]

    def test_stop_loss_exit(self):
        self._enter_deal("TGT", 150.0, 140.0)
        mock_om = MagicMock()
        mock_client = MagicMock()
        # Down 15% from entry (140 * 0.85 = 119) -> stop loss at -10%
        with patch("core.data.get_current_price", return_value=124.0):
            closed = check_ma_exits(mock_om, mock_client, dry_run=True)
        assert len(closed) == 1
        assert "Stop loss" in closed[0]["close_reason"]

    def test_pnl_positive_on_profit_exit(self):
        self._enter_deal("TGT", 150.0, 140.0)
        mock_om = MagicMock()
        mock_client = MagicMock()
        with patch("core.data.get_current_price", return_value=145.0):
            closed = check_ma_exits(mock_om, mock_client, dry_run=True)
        assert closed[0]["pnl_pct"] > 0

    def test_no_exit_when_no_deals(self):
        mock_om = MagicMock()
        mock_client = MagicMock()
        closed = check_ma_exits(mock_om, mock_client, dry_run=True)
        assert closed == []
