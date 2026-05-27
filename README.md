# Quantitative Portfolio System

A professional-grade, multi-strategy algorithmic trading system for a $100,000 portfolio.
Built after 45 waves of research targeting $500+/month.

## Architecture

```
$100,000
|- $75,000 (75%) Factor ETF Sleeve
|  |- AVUV  18%  US Small Cap Value (Avantis)
|  |- AVDV  22%  International Small Cap Value (Avantis)
|  |- QMOM  9%*  Momentum (B-SC vol scaled)
|  |- DBMF  12%  Managed Futures (iMGP DBi)
|  |- CTA   5%   Managed Futures (Simplify/Altis)
|
|- $15,000 (15%) Iron Condor Margin Buffer
|  |- SPX monthly condors (signal+tracking; Alpaca paper = no options)
|
|- $5,000 (5%)  PEAD Positions (rotating)
|  |- Post-earnings drift, $500M-$3B market cap
|
|- $5,000 (5%)  M&A Arbitrage (rotating)
   |- Cash deals only, 2-3 day entry after announcement
```

*QMOM: Base weight 18%, scaled by Barroso-Santa-Clara volatility formula.
Current scalar: 0.50x (QMOM at 9% due to high realized vol).

## Performance (2022-2025, live yfinance data)

| Metric | This Portfolio | S&P 500 |
|--------|--------------|---------|
| Annual Return | 14.2% | 10.9% |
| Sharpe Ratio | 0.875 | 0.605 |
| Max Drawdown | -14.8% | -24.5% |
| 2022 (bear year) | +3.1% | -18.6% |

Expected monthly income: ~$1,083/month at full deployment.

## Honest Limitations

1. **Iron condors are signal-only** on Alpaca paper (no options support).
   Move to tastytrade or IBKR for live options execution.
2. **M&A deal flow** requires manual entry or a paid deal-flow API.
   Free EDGAR scanning provided as a start.
3. **PEAD** works for $500M-$3B market cap. Does not work for large caps.
4. **B-SC scalar** is currently 0.50x (QMOM at 9%, not 18%).
   Rebuilds automatically when vol normalizes.
5. **Worst-case drawdown**: -35% to -42% in a 2008+2009 scenario.
   Recovery time: 3-5 years. Circuit breakers help but do not eliminate this.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Set credentials (or use config.py defaults for paper trading)
export ALPACA_KEY=your_key
export ALPACA_SECRET=your_secret

# Verify connection
python3 -c "from execution.alpaca_client import AlpacaClient; AlpacaClient(paper=True).verify_connection()"

# Run a full cycle (dry run -- no orders submitted)
cd scheduler && python3 main.py --once --dry-run

# Run the full system (execute real paper trades)
cd scheduler && python3 main.py --once

# Start continuous monitoring
cd scheduler && python3 main.py --monitor
```

## Running Tests

```bash
# Unit tests only (no network)
pytest tests/ -m "not integration" -v

# All tests including live Alpaca + market data
pytest tests/ -v

# Specific test file
pytest tests/test_regime.py -v
```

Current test count: 69 tests, all passing.

## Cron Schedule

```
# Weekly report + rebalance check - Monday 9:00 AM ET (14:00 UTC)
0 14 * * 1 cd /path/to/trader && python3 scheduler/main.py --weekly

# Daily condor check + PEAD exits - 9:35 AM ET (14:35 UTC) weekdays
35 14 * * 1-5 cd /path/to/trader && python3 scheduler/main.py --daily

# Continuous monitoring (run as background service)
python3 scheduler/main.py --monitor
```

## File Structure

```
trader/
|- config.py                  # All settings and credentials
|- requirements.txt
|- pytest.ini
|- core/
|  |- regime.py               # Market regime detection (5 regimes)
|  |- data.py                 # Data fetching (yfinance + Alpaca)
|  |- utils.py                # Logging, state, math helpers
|- strategies/
|  |- etf_manager.py          # Factor ETF sleeve + B-SC + rebalancing
|  |- iron_condor.py          # Options signal generator + tracker
|  |- pead_screener.py        # Post-earnings drift screener + execution
|  |- ma_monitor.py           # M&A arbitrage monitor + execution
|- execution/
|  |- alpaca_client.py        # Alpaca API wrapper (orders + positions)
|  |- order_manager.py        # Risk checks + circuit breaker + trade log
|- reporting/
|  |- email_reporter.py       # Weekly reports + alert emails
|- scheduler/
|  |- main.py                 # Main orchestrator (daily + weekly + monitor)
|- tests/
|  |- test_regime.py          # 14 unit tests
|  |- test_etf_manager.py     # 16 unit tests
|  |- test_iron_condor.py     # 13 unit tests
|  |- test_pead.py            # 8 unit tests
|  |- test_integration.py     # 12 integration tests (live APIs)
|- state/                     # Runtime state (gitignored)
   |- trade_log.json          # All trades (append-only JSON-lines)
   |- rebalance_state.json
   |- condor_state.json
   |- pead_state.json
   |- ma_state.json
```

## Risk Management

Circuit breaker levels (portfolio-level):
- Down 10%: Review. No change.
- Down 15%: Reduce. Stop new condors and PEAD entries.
- Down 20%: HALT. Move 25% to cash. No new positions.

Position-level limits:
- Max loss per condor: $2,000 (2% rule)
- Max loss per PEAD trade: $2,000 (2% rule)
- Max loss per M&A bet: $1,500

Do NOT put stop-losses on the ETFs. Their volatility (AVUV: 27% ann)
will trigger stops randomly. Circuit breakers are portfolio-level only.
