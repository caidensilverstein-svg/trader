"""
Vectorized backtesting engine.

Replays the ETF sleeve + regime logic against historical yfinance data.
No look-ahead bias: all signals are computed from data available at
each rebalance date (using only past prices).

Outputs:
  - Equity curve (daily)
  - Per-rebalance trade log
  - Summary statistics (Sharpe, max DD, Calmar, vs SPY benchmark)

Usage:
    from backtest.engine import run_backtest
    result = run_backtest(start='2018-01-01', end='2023-12-31')
    print(result['summary'])
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from core.regime import get_regime, compute_regime_indicators

logger = logging.getLogger(__name__)

# ETF universe for the backtest (same as live)
BACKTEST_TICKERS = list(config.ETF_TARGET_WEIGHTS.keys()) + ["SPY"]


def _download_prices(tickers: List[str], start: str, end: str) -> pd.DataFrame:
    """Download adjusted close prices for all tickers."""
    logger.info("Downloading price data for %s from %s to %s", tickers, start, end)
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw[["Close"]].rename(columns={"Close": tickers[0]})
    prices = prices.dropna(how="all")
    logger.info("Downloaded %d rows x %d tickers", len(prices), len(prices.columns))
    return prices


def _bsc_scalar(returns_series: pd.Series, lookback: int = 126) -> float:
    """Barroso-Santa-Clara scalar from a returns series."""
    if len(returns_series) < lookback:
        return config.BSC_MIN_SCALAR
    daily_var = float(returns_series.tail(lookback).var())
    ann_var   = daily_var * 252
    if ann_var == 0:
        return 1.0
    scalar = (config.BSC_TARGET_VOL ** 2) / ann_var
    return float(np.clip(scalar, config.BSC_MIN_SCALAR, config.BSC_MAX_SCALAR))


def _effective_weights(bsc: float, regime: str) -> Dict[str, float]:
    """Mirror of strategies/etf_manager.compute_effective_weights."""
    base = dict(config.ETF_TARGET_WEIGHTS)
    base["QMOM"] = base["QMOM"] * bsc
    mult = config.REGIME_ETF_MULT.get(regime, 1.0)
    return {k: v * mult for k, v in base.items()}


def run_backtest(
    start: str = "2018-01-01",
    end: Optional[str] = None,
    initial_capital: float = 100_000.0,
    rebalance_freq: str = "M",   # 'M' = monthly, 'W' = weekly, 'Q' = quarterly
    drift_threshold: float = 0.05,
    cost_bps: float = 3.0,       # round-trip transaction cost in basis points
) -> Dict:
    """
    Run a full historical backtest of the ETF sleeve strategy.

    Parameters
    ----------
    start            : ISO date string (YYYY-MM-DD)
    end              : ISO date string, defaults to today
    initial_capital  : Starting portfolio value in dollars
    rebalance_freq   : Pandas offset alias for rebalance schedule
    drift_threshold  : Minimum weight drift to trigger rebalance

    Returns
    -------
    dict with keys:
        equity_curve  : pd.Series (daily portfolio values)
        spy_curve     : pd.Series (SPY benchmark, same-sized)
        trade_log     : list of rebalance events
        summary       : dict of performance statistics
    """
    if end is None:
        end = datetime.today().strftime("%Y-%m-%d")

    prices = _download_prices(BACKTEST_TICKERS, start, end)

    # Drop tickers with excessive missing data
    available = [t for t in config.ETF_TARGET_WEIGHTS if t in prices.columns and prices[t].notna().sum() > 60]
    if len(available) < 3:
        return {"error": f"Insufficient data: only {available} available"}

    spy = prices["SPY"] if "SPY" in prices.columns else None
    etf_prices = prices[available].ffill()

    # VIX proxy from SPY realized vol (always use SPY, not factor ETFs)
    # Factor ETFs are more volatile which would over-trigger BEAR_CRISIS
    spy_returns = (prices["SPY"].pct_change() if "SPY" in prices.columns
                   else etf_prices.iloc[:, 0].pct_change())
    vix_proxy   = spy_returns.rolling(21).std() * 252 ** 0.5 * 100  # annualized, pct

    # SPY history for regime detection
    spy_for_regime = prices["SPY"].dropna() if "SPY" in prices.columns else None

    # --- Simulation state ---
    portfolio_value = initial_capital
    holdings: Dict[str, float] = {t: 0.0 for t in available}  # shares held
    cash = initial_capital

    equity_curve: Dict[str, float] = {}
    trade_log: List[Dict] = []
    rebalance_dates = []

    # Build rebalance schedule from available price dates
    date_index = etf_prices.index
    if rebalance_freq == "M":
        rebal_idx = date_index.to_period("M").drop_duplicates().to_timestamp("M")
        rebal_idx = pd.DatetimeIndex([
            date_index[date_index >= d].min() for d in rebal_idx if any(date_index >= d)
        ])
    elif rebalance_freq == "W":
        rebal_idx = date_index[date_index.dayofweek == 0]  # Mondays
    else:
        rebal_idx = date_index.to_period("Q").drop_duplicates().to_timestamp("Q")
        rebal_idx = pd.DatetimeIndex([
            date_index[date_index >= d].min() for d in rebal_idx if any(date_index >= d)
        ])
    rebal_set = set(rebal_idx)

    # Need at least 200 days of history for regime detection
    warmup_idx = 200

    for i, date in enumerate(date_index):
        row = etf_prices.iloc[i]

        # Mark-to-market portfolio
        holdings_value = sum(holdings[t] * row[t] for t in available if not pd.isna(row[t]))
        portfolio_value = holdings_value + cash
        equity_curve[str(date.date())] = portfolio_value

        # Rebalance check (after warmup period)
        if i >= warmup_idx and date in rebal_set:
            # --- Compute regime ---
            if spy_for_regime is not None:
                spy_slice = spy_for_regime.iloc[:i+1]
                vix_val   = float(vix_proxy.iloc[i]) if not pd.isna(vix_proxy.iloc[i]) else 20.0
                try:
                    spy_price = float(spy_slice.iloc[-1])
                    spy_ma200 = float(spy_slice.tail(200).mean())
                    # spy_mom60 as fraction (consistent with get_regime expectation)
                    spy_ret60 = float((spy_slice.iloc[-1] / spy_slice.iloc[-61]) - 1)
                    # Use 52-week (252 trading days) peak, consistent with live code
                    peak_window = min(252, len(spy_slice))
                    peak      = float(spy_slice.iloc[-peak_window:].max())
                    # dd_from_peak as fraction (get_regime expects fraction, e.g. -0.15)
                    dd        = (spy_price / peak) - 1.0
                    regime = get_regime(spy_price, spy_ma200, spy_ret60, vix_val, dd)
                except Exception:
                    regime = "BULL"
            else:
                regime = "BULL"
                vix_val = 20.0

            # --- B-SC scalar for QMOM ---
            if "QMOM" in available:
                qmom_rets = etf_prices["QMOM"].iloc[:i+1].pct_change().dropna()
                bsc = _bsc_scalar(qmom_rets)
            else:
                bsc = 1.0

            # --- Target weights ---
            target_wts = _effective_weights(bsc, regime)
            target_wts = {k: v for k, v in target_wts.items() if k in available}

            # --- Drift check ---
            current_wts = {}
            for t in target_wts:
                val = holdings[t] * row[t] if not pd.isna(row[t]) else 0.0
                current_wts[t] = val / portfolio_value if portfolio_value > 0 else 0.0

            max_drift = max(abs(current_wts.get(t, 0) - w) for t, w in target_wts.items())
            needs_rebal = max_drift >= drift_threshold

            if needs_rebal:
                # Apply transaction costs: cost_bps bp on gross turnover
                turnover = sum(abs(current_wts.get(t, 0) - w) for t, w in target_wts.items())
                cost = portfolio_value * turnover * (cost_bps / 10_000)
                portfolio_value -= cost  # deduct before rebalancing

                new_holdings = {}
                for t, wt in target_wts.items():
                    if t in row and not pd.isna(row[t]):
                        target_val = portfolio_value * wt
                        new_holdings[t] = target_val / row[t]
                    else:
                        new_holdings[t] = holdings[t]

                holdings = {t: new_holdings.get(t, 0.0) for t in available}
                cash = portfolio_value - sum(holdings[t] * row[t] for t in available if not pd.isna(row[t]))

                rebalance_dates.append(str(date.date()))
                trade_log.append({
                    "date":      str(date.date()),
                    "regime":    regime,
                    "bsc":       round(bsc, 3),
                    "drift":     round(max_drift, 4),
                    "turnover":  round(turnover, 4),
                    "cost_usd":  round(cost, 2),
                    "target_weights": {k: round(v, 4) for k, v in target_wts.items()},
                })

    # --- Build equity series ---
    eq_series = pd.Series(equity_curve)
    eq_series.index = pd.to_datetime(eq_series.index)

    # --- SPY benchmark (buy and hold) ---
    if spy is not None:
        spy_aligned = spy.reindex(eq_series.index).ffill().dropna()
        spy_curve   = initial_capital * (spy_aligned / spy_aligned.iloc[0])
    else:
        spy_curve = eq_series  # fallback

    # --- Performance metrics ---
    summary = _compute_summary(eq_series, spy_curve, trade_log)
    summary["start"]           = start
    summary["end"]             = end
    summary["initial_capital"] = initial_capital
    summary["rebalance_count"] = len(trade_log)
    summary["total_costs_usd"] = round(sum(t.get("cost_usd", 0) for t in trade_log), 2)
    summary["cost_bps"]        = cost_bps

    return {
        "equity_curve": eq_series,
        "spy_curve":    spy_curve,
        "trade_log":    trade_log,
        "summary":      summary,
    }


def _compute_summary(eq: pd.Series, spy: pd.Series, trade_log: List) -> Dict:
    """Compute standard metrics for both strategy and benchmark."""
    def _metrics(s: pd.Series) -> Dict:
        rets = s.pct_change().dropna()
        total_ret  = (s.iloc[-1] / s.iloc[0]) - 1
        n_days     = len(rets)
        ann_ret    = (1 + total_ret) ** (252 / max(n_days, 1)) - 1
        ann_vol    = float(rets.std() * 252 ** 0.5)
        excess     = rets - (0.05 / 252)
        sharpe     = float(excess.mean() / excess.std() * 252 ** 0.5) if excess.std() > 0 else 0.0
        # Max drawdown
        rolling_max = s.cummax()
        dd_series   = (s / rolling_max) - 1
        max_dd      = float(dd_series.min())
        calmar      = float(ann_ret / abs(max_dd)) if max_dd != 0 else float("inf")
        return {
            "total_return": round(total_ret * 100, 2),
            "ann_return":   round(ann_ret * 100, 2),
            "ann_vol":      round(ann_vol * 100, 2),
            "sharpe":       round(sharpe, 3),
            "max_dd":       round(max_dd * 100, 2),
            "calmar":       round(calmar, 3),
        }

    strat_m = _metrics(eq)
    bench_m = _metrics(spy.reindex(eq.index).ffill())

    # Regime distribution from trade log
    regimes = [t["regime"] for t in trade_log]
    regime_counts = {}
    for r in set(regimes):
        regime_counts[r] = regimes.count(r)

    return {
        "strategy":       strat_m,
        "benchmark_spy":  bench_m,
        "alpha_ann":      round(strat_m["ann_return"] - bench_m["ann_return"], 2),
        "regime_counts":  regime_counts,
        "n_trading_days": len(eq),
    }


def format_backtest_report(result: Dict) -> str:
    """ASCII-safe report from backtest results."""
    if "error" in result:
        return f"BACKTEST ERROR: {result['error']}"

    s = result["summary"]
    strat = s["strategy"]
    bench = s["benchmark_spy"]

    lines = [
        "=" * 72,
        "BACKTEST RESULTS",
        f"Period: {s['start']} to {s['end']}",
        f"Initial Capital: ${s['initial_capital']:,.0f}",
        "=" * 72,
        "",
        f"{'Metric':<20} {'Strategy':>12} {'SPY (BnH)':>12} {'Edge':>10}",
        "-" * 56,
        f"{'Total Return':<20} {strat['total_return']:>11.1f}% {bench['total_return']:>11.1f}% {strat['total_return']-bench['total_return']:>+9.1f}%",
        f"{'Ann. Return':<20} {strat['ann_return']:>11.1f}% {bench['ann_return']:>11.1f}% {strat['ann_return']-bench['ann_return']:>+9.1f}%",
        f"{'Ann. Volatility':<20} {strat['ann_vol']:>11.1f}% {bench['ann_vol']:>11.1f}%",
        f"{'Sharpe Ratio':<20} {strat['sharpe']:>12.3f} {bench['sharpe']:>12.3f} {strat['sharpe']-bench['sharpe']:>+10.3f}",
        f"{'Max Drawdown':<20} {strat['max_dd']:>11.1f}% {bench['max_dd']:>11.1f}%",
        f"{'Calmar Ratio':<20} {strat['calmar']:>12.3f} {bench['calmar']:>12.3f}",
        "-" * 56,
        f"{'Rebalances':<20} {s['rebalance_count']:>12}",
        f"{'Trading Days':<20} {s['n_trading_days']:>12}",
        "",
        "REGIME DISTRIBUTION",
        "-" * 40,
    ]
    for regime, count in sorted(s.get("regime_counts", {}).items()):
        lines.append(f"  {regime:<20} {count:>4} rebalances")

    total_costs = s.get("total_costs_usd", 0)
    cost_bps    = s.get("cost_bps", 3)
    lines += [
        "",
        f"Annual Alpha vs SPY  : {s['alpha_ann']:+.2f}%",
        f"Transaction Costs    : ${total_costs:,.2f} total ({cost_bps:.1f} bps/rebalance)",
        "=" * 72,
    ]
    return "\n".join(lines)
