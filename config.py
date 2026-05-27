"""
Central configuration for the Quant Trading System.
All credentials, target weights, and tunable parameters live here.
"""
import os
from dataclasses import dataclass, field
from typing import Dict, Optional

# ---------------------------------------------------------------------------
# Credentials — override with environment variables in production
# ---------------------------------------------------------------------------
ALPACA_KEY    = os.getenv("ALPACA_KEY",    "PKTP2COCQZYF6FF5YGDRZ5KZIO")
ALPACA_SECRET = os.getenv("ALPACA_SECRET", "ChKq2QGXL5FEns8y3TBHftmcdEYUJ2USLJD7GMy6hpL8")
ALPACA_BASE   = os.getenv("ALPACA_BASE",   "https://paper-api.alpaca.markets")
PAPER_TRADING = True  # Always True until explicitly changed

EMAIL_SENDER   = os.getenv("EMAIL_SENDER",   "caidensilverstein@gmail.com")
EMAIL_RECEIVER = os.getenv("EMAIL_RECEIVER", "caidensilverstein@gmail.com")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD", "ahoe xrte evwv nkcr")

# ---------------------------------------------------------------------------
# Portfolio constants
# ---------------------------------------------------------------------------
TOTAL_CAPITAL = 100_000.0  # Starting capital

# Target weights (as fractions, sum = 0.75 for ETF sleeve)
ETF_TARGET_WEIGHTS: Dict[str, float] = {
    "AVUV": 0.18,  # US Small Cap Value (Avantis)
    "AVDV": 0.22,  # International Small Cap Value (Avantis)
    "QMOM": 0.18,  # Momentum (Alpha Architect) -- B-SC scales this weekly
    "DBMF": 0.12,  # Managed Futures CTA replication (iMGP DBi)
    "CTA":  0.05,  # Managed Futures Simplify/Altis
}
# 0.25 reserved: options margin + PEAD + M&A + cash buffer

# Rebalancing trigger: drift from target before we act
REBALANCE_DRIFT_THRESHOLD = 0.05  # 5% absolute drift

# ---------------------------------------------------------------------------
# Barroso-Santa-Clara (B-SC) momentum scaling
# ---------------------------------------------------------------------------
BSC_TARGET_VOL   = 0.12   # 12% target annual volatility for QMOM
BSC_LOOKBACK_DAYS = 126   # 6-month rolling window
BSC_MIN_SCALAR    = 0.50  # Never go below 50% of target
BSC_MAX_SCALAR    = 2.00  # Never go above 200% of target

# ---------------------------------------------------------------------------
# Regime detection thresholds
# ---------------------------------------------------------------------------
REGIME_VIX_CRISIS   = 30.0   # VIX above this = BEAR_CRISIS
REGIME_DRAWDOWN_CRISIS = -0.20  # 20% drawdown from 52-week high = BEAR_CRISIS
REGIME_MOM_WINDOW   = 60     # Days for SPY momentum signal
REGIME_VIX_HIGH     = 20.0   # VIX below this is "low"

# Regime-based position size multipliers (fraction of full allocation)
REGIME_ETF_MULT: Dict[str, float] = {
    "BULL":        1.00,
    "MILD_BULL":   0.90,
    "SIDEWAYS":    0.80,
    "BEAR":        0.65,
    "BEAR_CRISIS": 0.50,
}
REGIME_CONDOR_MULT: Dict[str, float] = {
    "BULL":        1.00,
    "MILD_BULL":   0.75,
    "SIDEWAYS":    0.50,
    "BEAR":        0.25,
    "BEAR_CRISIS": 0.00,
}
REGIME_PEAD_MULT: Dict[str, float] = {
    "BULL":        1.00,
    "MILD_BULL":   0.75,
    "SIDEWAYS":    0.50,
    "BEAR":        0.25,
    "BEAR_CRISIS": 0.00,
}

# ---------------------------------------------------------------------------
# Iron condor parameters (signal only — Alpaca paper doesn't support options)
# ---------------------------------------------------------------------------
CONDOR_VIX_MIN    = 15.0   # Skip below this
CONDOR_VIX_MAX    = 35.0   # Skip above this
CONDOR_DTE_TARGET = 38     # Target days to expiration (30-45 range)
CONDOR_DELTA_SHORT = 0.16  # Short strike delta (16-delta)
CONDOR_WING_POINTS = 35    # Wing width in SPX points
CONDOR_PROFIT_TARGET = 0.50  # Close at 50% of credit
CONDOR_LOSS_TARGET   = 2.00  # Close at 200% of credit (2x credit paid)
CONDOR_DTE_EXIT      = 21   # Close at 21 DTE regardless
CONDOR_CAPITAL_PCT   = 0.15  # 15% of capital as margin buffer

# VIX-based sizing multipliers for condors
CONDOR_VIX_MULT = {
    (15, 20): 1.00,
    (20, 25): 0.75,
    (25, 35): 0.25,
}

# ---------------------------------------------------------------------------
# PEAD (Post-Earnings Announcement Drift) parameters
# ---------------------------------------------------------------------------
PEAD_SURPRISE_MIN  = 0.15   # Minimum EPS surprise (15%)
PEAD_MCAP_MIN      = 500e6  # Minimum market cap $500M
PEAD_MCAP_MAX      = 3e9    # Maximum market cap $3B
PEAD_VOLUME_MIN    = 1e6    # Minimum average daily dollar volume
PEAD_GAP_MIN       = 0.02   # Minimum gap-up on announcement day (2%)
PEAD_POSITION_MIN  = 2_000  # Minimum position size
PEAD_POSITION_MAX  = 5_000  # Maximum position size
PEAD_MAX_POSITIONS = 3      # Maximum simultaneous positions
PEAD_HOLD_DAYS     = 45     # Maximum hold period (trading days)
PEAD_STOP_LOSS     = -0.07  # Stop loss (-7%)

# ---------------------------------------------------------------------------
# M&A Arbitrage parameters
# ---------------------------------------------------------------------------
MA_DEAL_MIN   = 500e6   # Minimum deal size $500M
MA_DEAL_MAX   = 10e9    # Maximum deal size $10B
MA_SPREAD_MIN = 0.01    # Minimum spread 1%
MA_SPREAD_MAX = 0.15    # Maximum spread 15% (above = too risky)
MA_HOLD_MAX   = 120     # Maximum hold days
MA_POSITION   = 2_500   # Position size per deal
MA_EXIT_PCT   = 0.95    # Exit when stock reaches 95% of deal price

# ---------------------------------------------------------------------------
# Risk management / circuit breakers
# ---------------------------------------------------------------------------
CIRCUIT_REVIEW_DD  = -0.10  # -10% drawdown: review
CIRCUIT_REDUCE_DD  = -0.15  # -15% drawdown: stop new condors/PEAD
CIRCUIT_HALT_DD    = -0.20  # -20% drawdown: halt all active trades
CIRCUIT_CASH_PCT   = 0.25   # Move 25% to cash at circuit breaker

# Position-level risk limits
MAX_LOSS_PER_TRADE = 0.02   # 2% of capital max loss per trade
MAX_LOSS_CONDOR    = 2_000  # Absolute max condor loss
MAX_LOSS_PEAD      = 2_000  # Absolute max PEAD position loss
MAX_LOSS_MA        = 1_500  # Absolute max M&A arb loss

# ---------------------------------------------------------------------------
# Scheduling
# ---------------------------------------------------------------------------
WEEKLY_REPORT_DAY   = "monday"
WEEKLY_REPORT_HOUR  = 9
DAILY_CONDOR_HOUR   = 9
DAILY_CONDOR_MINUTE = 35
MONTHLY_REBAL_DAY   = 1
MONTHLY_REBAL_HOUR  = 10

# ---------------------------------------------------------------------------
# Data / paths
# ---------------------------------------------------------------------------
STATE_DIR  = "state"
LOG_FILE   = "state/trade_log.json"
REBAL_FILE = "state/rebalance_state.json"
CONDOR_FILE = "state/condor_state.json"
PEAD_FILE  = "state/pead_state.json"
MA_FILE    = "state/ma_state.json"
