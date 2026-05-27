"""
Integration tests: live Alpaca connection + live market data.
These tests hit real APIs. Run with: pytest tests/test_integration.py -v
Marked as 'integration' so they can be excluded from CI: pytest -m "not integration"
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

import config


@pytest.mark.integration
class TestAlpacaConnection:

    def test_connects_and_gets_account(self):
        from execution.alpaca_client import AlpacaClient
        client = AlpacaClient(paper=True)
        assert client.verify_connection()

    def test_account_has_expected_fields(self):
        from execution.alpaca_client import AlpacaClient
        client = AlpacaClient(paper=True)
        acct = client.get_account()
        assert "equity" in acct
        assert "cash" in acct
        assert acct["equity"] > 0

    def test_get_positions_returns_dict(self):
        from execution.alpaca_client import AlpacaClient
        client = AlpacaClient(paper=True)
        positions = client.get_positions()
        assert isinstance(positions, dict)

    def test_market_open_is_bool(self):
        from execution.alpaca_client import AlpacaClient
        client = AlpacaClient(paper=True)
        result = client.is_market_open()
        assert isinstance(result, bool)

    def test_get_latest_price_spy(self):
        from execution.alpaca_client import AlpacaClient
        client = AlpacaClient(paper=True)
        price = client.get_latest_price("SPY")
        # SPY is roughly in a normal range
        assert price is None or (10 < price < 10000)


@pytest.mark.integration
class TestMarketData:

    def test_spy_vix_fetch(self):
        from core.data import get_spy_vix
        spy, vix = get_spy_vix("1y")
        assert len(spy) > 200
        assert len(vix) > 200
        assert all(v > 0 for v in spy)
        assert all(v > 0 for v in vix)

    def test_current_vix_reasonable(self):
        from core.data import get_current_vix
        vix = get_current_vix()
        # VIX is almost always between 10 and 80
        assert 5 < vix < 100

    def test_regime_from_live_data(self):
        from core.data import get_spy_vix
        from core.regime import regime_from_history, REGIMES
        spy, vix = get_spy_vix("2y")
        regime = regime_from_history(spy, vix)
        assert regime in REGIMES

    def test_etf_prices_fetch(self):
        from core.data import get_etf_prices
        df = get_etf_prices(["AVUV", "AVDV", "DBMF"], period="6mo")
        assert len(df) > 100
        assert all(col in df.columns for col in ["AVUV", "AVDV", "DBMF"])


@pytest.mark.integration
class TestSystemRun:

    def test_etf_manager_dry_run(self):
        """Full ETF manager run with dry_run=True — no orders submitted."""
        from execution.alpaca_client import AlpacaClient
        from execution.order_manager import OrderManager
        from strategies.etf_manager import run_etf_manager
        client = AlpacaClient(paper=True)
        om = OrderManager(client)
        result = run_etf_manager(client, om, dry_run=True)
        assert "regime" in result
        assert "bsc_scalar" in result
        assert 0.5 <= result["bsc_scalar"] <= 2.0

    def test_condor_signal_dry_run(self):
        """Condor signal generation — no orders."""
        from core.data import get_current_vix, get_current_price
        from core.data import get_spy_vix
        from core.regime import regime_from_history
        from strategies.iron_condor import open_condor_signal
        spy, vix = get_spy_vix("2y")
        regime = regime_from_history(spy, vix)
        current_vix = get_current_vix()
        spx = get_current_price("SPY") * 10
        # Just checks it doesn't crash — result depends on VIX
        result = open_condor_signal(spx, current_vix, regime)
        # result is either a dict or None (VIX out of range)
        assert result is None or isinstance(result, dict)

    def test_weekly_report_dry_run(self):
        """Weekly report generation without sending email."""
        from reporting.email_reporter import format_weekly_report
        regime_data  = {"regime": "BULL", "vix": 17.0, "spy_mom_60d": 5.2, "dd_from_peak": -0.5}
        account_data = {"equity": 100000, "cash": 25000, "buying_power": 50000, "portfolio_value": 100000}
        etf_status   = {"bsc_scalar": 0.5, "eff_qmom_wt": 9.0, "last_rebalance": "2026-05-27"}
        condor_status = {"open_count": 0, "closed_count": 2, "win_rate": 75.0, "total_pnl": 150.0}
        pead_status  = {"open_count": 1, "closed_count": 3, "win_rate": 66.7, "open_positions": ["APPS"]}
        ma_status    = {"open_count": 0, "closed_count": 1, "win_rate": 100.0, "open_deals": []}
        report = format_weekly_report(
            regime_data, account_data, etf_status, condor_status, pead_status, ma_status, 10
        )
        assert "WEEKLY PORTFOLIO REPORT" in report
        assert "BULL" in report
        assert "100,000.00" in report
