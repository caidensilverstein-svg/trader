"""
Factor ETF Sleeve Manager — the core 75% allocation.

Implements:
  1. Barroso-Santa-Clara (B-SC) volatility scaling for QMOM
  2. Market regime detection (adjusts position sizes)
  3. 5%-band drift-triggered rebalancing
  4. Alpaca order execution

All logic is deterministic given the same inputs, making it fully testable.
"""

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

import config
from core import data as mdata
from core.regime import regime_summary
from core.utils import load_state, save_state, now_utc, clip
from core.momentum_timing import spy_time_series_momentum, combined_regime_signal
from execution.alpaca_client import AlpacaClient
from execution.order_manager import OrderManager
from core.factor_timing import compute_etf_momentum, apply_factor_timing

logger = logging.getLogger(__name__)

TICKERS = list(config.ETF_TARGET_WEIGHTS.keys())


# ---------------------------------------------------------------------------
# B-SC volatility scalar
# ---------------------------------------------------------------------------

def compute_bsc_scalar(qmom_prices: pd.Series) -> float:
    """
    Barroso-Santa-Clara (2015) vol-scaling for QMOM.

    scalar = target_var / realized_var_126d
    Clipped to [BSC_MIN_SCALAR, BSC_MAX_SCALAR].

    Returns
    -------
    float : Scalar applied to the base QMOM weight.
    """
    returns = qmom_prices.pct_change().dropna()
    if len(returns) < config.BSC_LOOKBACK_DAYS:
        logger.warning("Insufficient QMOM history for B-SC, using min scalar")
        return config.BSC_MIN_SCALAR

    daily_var = float(returns.tail(config.BSC_LOOKBACK_DAYS).var())
    ann_var   = daily_var * 252
    target_var = config.BSC_TARGET_VOL ** 2

    if ann_var == 0:
        return 1.0

    scalar = target_var / ann_var
    clipped = clip(scalar, config.BSC_MIN_SCALAR, config.BSC_MAX_SCALAR)
    realized_vol = (ann_var ** 0.5) * 100

    logger.info(
        "B-SC QMOM: realized_vol=%.1f%%  scalar=%.3f (clipped=%.3f)",
        realized_vol, scalar, clipped,
    )
    return clipped


# ---------------------------------------------------------------------------
# Effective target weights
# ---------------------------------------------------------------------------

def compute_effective_weights(
    bsc_scalar: float,
    regime: str,
) -> Dict[str, float]:
    """
    Apply B-SC scalar to QMOM and regime multiplier to all ETF weights.

    Parameters
    ----------
    bsc_scalar : B-SC scalar for QMOM (0.5 – 2.0)
    regime     : Current market regime string

    Returns
    -------
    dict : {ticker: effective_fraction_of_TOTAL_CAPITAL}
    """
    base = dict(config.ETF_TARGET_WEIGHTS)

    # Apply B-SC: QMOM gets scaled, extra cash stays in buffer
    qmom_effective = base["QMOM"] * bsc_scalar
    base["QMOM"] = qmom_effective

    # Apply regime multiplier to entire ETF sleeve
    mult = config.REGIME_ETF_MULT.get(regime, 1.0)
    effective = {k: v * mult for k, v in base.items()}

    total = sum(effective.values())
    logger.info(
        "Effective weights (B-SC=%.2fx regime=%s mult=%.2f): total=%.1f%%",
        bsc_scalar, regime, mult, total * 100,
    )
    return effective


# ---------------------------------------------------------------------------
# Drift detection
# ---------------------------------------------------------------------------

def compute_drift(
    current_values: Dict[str, float],
    target_weights: Dict[str, float],
    total_capital: float,
) -> Dict[str, float]:
    """
    Compute absolute drift (current_weight - target_weight) for each ETF.

    Parameters
    ----------
    current_values  : {ticker: market_value} from Alpaca
    target_weights  : {ticker: target_fraction}
    total_capital   : Portfolio total capital

    Returns
    -------
    dict : {ticker: drift_fraction}  (positive = overweight)
    """
    drift = {}
    for ticker in target_weights:
        curr_val   = current_values.get(ticker, 0.0)
        curr_wt    = curr_val / total_capital
        target_wt  = target_weights[ticker]
        drift[ticker] = curr_wt - target_wt

    return drift


def needs_rebalance(drift: Dict[str, float]) -> bool:
    """Return True if any ETF has drifted >= the threshold (inclusive)."""
    return any(abs(d) >= config.REBALANCE_DRIFT_THRESHOLD for d in drift.values())


# ---------------------------------------------------------------------------
# Rebalancing
# ---------------------------------------------------------------------------

def rebalance(
    om: OrderManager,
    client: AlpacaClient,
    target_weights: Dict[str, float],
    dry_run: bool = False,
) -> Dict[str, str]:
    """
    Execute rebalancing trades to bring portfolio to target weights.

    Sells overweight positions first, then buys underweight ones.

    Parameters
    ----------
    om            : OrderManager instance
    client        : AlpacaClient instance
    target_weights: {ticker: fraction_of_total_capital}
    dry_run       : If True, log but do not submit orders

    Returns
    -------
    dict : {ticker: 'buy'/'sell'/'hold'/'error'}
    """
    acct     = client.get_account()
    total    = acct["equity"]
    positions = client.get_positions()

    current_values = {t: positions[t]["market_value"] if t in positions else 0.0
                      for t in target_weights}

    drift = compute_drift(current_values, target_weights, total)

    logger.info("=== REBALANCE CHECK ===")
    for ticker, d in drift.items():
        logger.info("  %s: current=%.1f%%  target=%.1f%%  drift=%+.1f%%",
                    ticker,
                    current_values.get(ticker, 0) / total * 100,
                    target_weights[ticker] * 100,
                    d * 100)

    if not needs_rebalance(drift):
        logger.info("No rebalancing needed (max drift %.1f%% < threshold %.1f%%)",
                    max(abs(d) for d in drift.values()) * 100,
                    config.REBALANCE_DRIFT_THRESHOLD * 100)
        return {t: "hold" for t in target_weights}

    actions: Dict[str, str] = {}

    # Step 1: Sell overweight positions (frees up cash)
    for ticker, d in sorted(drift.items(), key=lambda x: x[1], reverse=True):
        if d > config.REBALANCE_DRIFT_THRESHOLD:
            target_val   = target_weights[ticker] * total
            current_val  = current_values.get(ticker, 0)
            sell_amount  = current_val - target_val
            sell_qty_est = sell_amount / (client.get_latest_price(ticker) or 1)

            logger.info("SELL %s  $%.2f (drift %+.1f%%)", ticker, sell_amount, d * 100)
            if not dry_run:
                pos = client.get_position(ticker)
                if pos:
                    sell_qty = min(sell_qty_est, float(pos["qty"]))
                    oid = om.sell_qty(ticker, sell_qty, "ETF_MANAGER",
                                     f"rebalance drift {d*100:+.1f}%")
                    actions[ticker] = "sell" if oid else "error"
                else:
                    actions[ticker] = "hold"
            else:
                actions[ticker] = "sell(dry)"

    # Step 2: Buy underweight positions
    acct = client.get_account()
    cash = acct["cash"]

    for ticker, d in sorted(drift.items(), key=lambda x: x[1]):
        if d <= -config.REBALANCE_DRIFT_THRESHOLD:
            target_val  = target_weights[ticker] * total
            current_val = current_values.get(ticker, 0)
            buy_amount  = min(target_val - current_val, cash * 0.95)

            if buy_amount < 10:
                actions[ticker] = "hold"
                continue

            logger.info("BUY %s  $%.2f (drift %+.1f%%)", ticker, buy_amount, d * 100)
            if not dry_run:
                # ETFs: pass max_loss=0 to disable the per-trade risk cap.
                # ETFs are diversified funds and cannot go to zero.
                oid = om.buy_notional(ticker, buy_amount, "ETF_MANAGER",
                                      f"rebalance drift {d*100:+.1f}%",
                                      max_loss=0)
                actions[ticker] = "buy" if oid else "error"
                cash -= buy_amount
            else:
                actions[ticker] = "buy(dry)"
                cash -= buy_amount

    # Save state
    state = load_state(config.REBAL_FILE, {})
    state["last_rebalance"] = now_utc()
    state["actions"] = actions
    state["drift"]   = {k: round(v, 4) for k, v in drift.items()}
    save_state(config.REBAL_FILE, state)

    return actions


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_etf_manager(
    client: AlpacaClient,
    om: OrderManager,
    dry_run: bool = False,
) -> dict:
    """
    Full ETF manager run: compute B-SC scalar, detect regime,
    compute effective weights, check drift, rebalance if needed.

    Returns
    -------
    dict : Summary for reporting
    """
    logger.info("=== ETF MANAGER RUN ===")

    # Fetch market data
    spy_hist, vix_hist = mdata.get_spy_vix("2y")
    qmom_prices = mdata.get_price_history("QMOM", "1y")

    # Regime
    reg = regime_summary(spy_hist, vix_hist)
    regime = reg["regime"]

    # Momentum timing (secondary signal, logged but does not override regime weights)
    spy_mom = spy_time_series_momentum(spy_hist)
    combined = combined_regime_signal(regime, spy_mom["composite"])
    logger.info(
        "Momentum timing: composite=%+.2f%% signal=%s combined=%s",
        spy_mom["composite"], spy_mom["signal"], combined,
    )

    # B-SC scalar
    bsc = compute_bsc_scalar(qmom_prices)

    # Effective weights (B-SC + regime)
    eff_weights = compute_effective_weights(bsc, regime)

    # Factor timing: adjust individual ETF weights by 6-month momentum
    etf_prices = {t: mdata.get_price_history(t, "1y") for t in TICKERS}
    etf_momentum = compute_etf_momentum(etf_prices)
    timed_weights, ft_multipliers = apply_factor_timing(eff_weights, etf_momentum)
    logger.info("Factor timing applied: %s", {t: f"{v*100:.1f}%" for t, v in timed_weights.items()})

    # Rebalance using factor-timed weights
    actions = rebalance(om, client, timed_weights, dry_run=dry_run)

    summary = {
        "ts":              now_utc(),
        "regime":          regime,
        "bsc_scalar":      round(bsc, 3),
        "eff_qmom_wt":     round(timed_weights.get("QMOM", 0) * 100, 1),
        "vix":             reg["vix"],
        "spy_mom_60d":     reg["spy_mom_60d"],
        "mom_composite":   spy_mom["composite"],
        "mom_signal":      spy_mom["signal"],
        "combined_signal": combined,
        "etf_momentum":    {t: round(v * 100, 1) for t, v in etf_momentum.items()},
        "ft_multipliers":  {t: round(v, 2) for t, v in ft_multipliers.items()},
        "actions":         actions,
    }

    state = load_state(config.REBAL_FILE, {})
    state.update(summary)
    save_state(config.REBAL_FILE, state)

    logger.info("ETF Manager done: %s", summary)
    return summary
