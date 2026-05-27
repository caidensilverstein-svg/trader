"""
Factor return attribution for the ETF sleeve.

Computes contribution to portfolio return from each ETF/factor over any period.
Answers: "How much did value vs momentum vs managed futures contribute?"

Uses simple arithmetic attribution (not Brinson-Hood-Beebower, which requires
sector benchmarks). For each ETF:
    Contribution = avg_weight * ETF_return

Sum of contributions should equal portfolio return (before transaction costs).
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yfinance as yf

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import config

logger = logging.getLogger(__name__)

FACTOR_LABELS = {
    "AVUV": "US Small-Cap Value",
    "AVDV": "Intl Small-Cap Value",
    "QMOM": "US Momentum (B-SC scaled)",
    "DBMF": "Managed Futures",
    "CTA":  "Trend Following",
}


def compute_attribution(
    start: str,
    end: Optional[str] = None,
    base_weights: Dict[str, float] = None,
) -> Dict:
    """
    Compute factor attribution for the given period.

    Parameters
    ----------
    start        : ISO date (start of period)
    end          : ISO date (end of period), defaults to today
    base_weights : Override default weights (defaults to config.ETF_TARGET_WEIGHTS)

    Returns
    -------
    dict with per-ticker attribution and portfolio total
    """
    if end is None:
        from datetime import datetime
        end = datetime.today().strftime("%Y-%m-%d")

    if base_weights is None:
        base_weights = dict(config.ETF_TARGET_WEIGHTS)
        # Apply current B-SC scalar (0.5x for QMOM at current vol)
        base_weights["QMOM"] *= 0.5

    tickers = list(base_weights.keys())
    raw = yf.download(tickers + ["SPY"], start=start, end=end,
                      auto_adjust=True, progress=False)

    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw[["Close"]].rename(columns={"Close": tickers[0]})

    prices = prices.dropna(how="all")
    if len(prices) < 5:
        return {"error": f"Insufficient data: {len(prices)} rows"}

    attribution = {}
    total_contribution = 0.0

    for ticker in tickers:
        if ticker not in prices.columns:
            continue
        series = prices[ticker].dropna()
        if len(series) < 2:
            continue
        total_return = float((series.iloc[-1] / series.iloc[0]) - 1) * 100
        weight       = base_weights.get(ticker, 0)
        contribution = weight * total_return

        attribution[ticker] = {
            "ticker":       ticker,
            "factor":       FACTOR_LABELS.get(ticker, ticker),
            "weight":       round(weight * 100, 1),
            "total_return": round(total_return, 2),
            "contribution": round(contribution, 2),  # in return percentage points
        }
        total_contribution += contribution

    # SPY benchmark for comparison
    spy_return = None
    if "SPY" in prices.columns:
        spy_s = prices["SPY"].dropna()
        if len(spy_s) >= 2:
            spy_return = round(float((spy_s.iloc[-1] / spy_s.iloc[0]) - 1) * 100, 2)

    return {
        "start":                start,
        "end":                  end,
        "per_ticker":           attribution,
        "total_contribution":   round(total_contribution, 2),
        "spy_return":           spy_return,
        "weights_used":         {k: round(v * 100, 1) for k, v in base_weights.items()},
    }


def format_attribution_report(result: Dict) -> str:
    if "error" in result:
        return f"ATTRIBUTION ERROR: {result['error']}"

    lines = [
        "=" * 72,
        f"FACTOR ATTRIBUTION: {result['start']} to {result['end']}",
        "=" * 72,
        "",
        f"{'Ticker':<8} {'Factor':<28} {'Weight':>7} {'Return':>8} {'Contribution':>13}",
        "-" * 70,
    ]

    for ticker, d in sorted(result["per_ticker"].items(),
                            key=lambda x: abs(x[1]["contribution"]), reverse=True):
        lines.append(
            f"{d['ticker']:<8} {d['factor']:<28} {d['weight']:>6.1f}%  "
            f"{d['total_return']:>7.1f}%  {d['contribution']:>+11.2f}pp"
        )

    lines += [
        "-" * 70,
        f"{'TOTAL':<36} {'':>7}   {'':>7}   {result['total_contribution']:>+11.2f}pp",
        "",
    ]

    if result["spy_return"] is not None:
        diff = result["total_contribution"] - result["spy_return"]
        lines.append(f"SPY Benchmark Return  : {result['spy_return']:+.2f}%")
        lines.append(f"Strategy Contribution : {result['total_contribution']:+.2f}pp")
        lines.append(f"Difference vs SPY     : {diff:+.2f}pp")

    lines += [
        "",
        "Note: Contribution = Weight x Return. Sums to weighted portfolio return.",
        "Actual portfolio return will differ slightly due to rebalancing and costs.",
        "=" * 72,
    ]
    return "\n".join(lines)
