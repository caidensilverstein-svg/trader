"""
Multi-benchmark performance comparison.

Compares the strategy against standard portfolio benchmarks:
  1. SPY (S&P 500 market cap weight)
  2. 60/40 (60% SPY + 40% AGG bonds)
  3. Equal-weight (20% each: SPY, IWM, EFA, AGG, GLD)
  4. Risk parity (inverse vol weights)

For each benchmark computes: total return, Sharpe, max DD, Calmar,
and alpha (outperformance).

Academic basis: Markowitz (1952) mean-variance; 
Swensen (2000) "Pioneering Portfolio Management"
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    name:         str
    total_return: float   # %
    ann_return:   float   # %
    ann_vol:      float   # %
    sharpe:       float
    max_dd:       float   # % (negative)
    calmar:       float
    alpha_vs_spy: float   # pp outperformance over SPY (annualized)


def _metrics(equity: pd.Series) -> Dict[str, float]:
    """Compute standard metrics for an equity curve."""
    if len(equity) < 20:
        return {}
    rets = equity.pct_change().dropna()
    n = len(rets)
    total_ret  = float(equity.iloc[-1] / equity.iloc[0] - 1) * 100
    ann_ret    = ((1 + total_ret / 100) ** (252 / n) - 1) * 100
    ann_vol    = float(rets.std() * np.sqrt(252)) * 100
    sharpe     = ann_ret / ann_vol if ann_vol > 0 else 0.0
    cummax     = equity.cummax()
    max_dd     = float((equity / cummax - 1).min()) * 100
    calmar     = ann_ret / abs(max_dd) if max_dd != 0 else float("inf")
    return {
        "total_return": round(total_ret, 2),
        "ann_return":   round(ann_ret, 2),
        "ann_vol":      round(ann_vol, 2),
        "sharpe":       round(sharpe, 3),
        "max_dd":       round(max_dd, 2),
        "calmar":       round(calmar, 3),
    }


def build_benchmark_equity_curves(
    price_data: Dict[str, pd.Series],
    start_value: float = 100_000.0,
) -> Dict[str, pd.Series]:
    """
    Construct benchmark equity curves from raw price data.

    Parameters
    ----------
    price_data : {ticker: price_series} for SPY, AGG, IWM, EFA, GLD
    start_value: initial portfolio value

    Returns
    -------
    {benchmark_name: equity_series}
    """
    curves = {}
    spy = price_data.get("SPY")
    agg = price_data.get("AGG")
    iwm = price_data.get("IWM")
    efa = price_data.get("EFA")
    gld = price_data.get("GLD")

    if spy is None:
        return curves

    # SPY pure
    spy_rets = spy.pct_change().dropna()
    curves["SPY (100%)"] = start_value * (1 + spy_rets).cumprod()

    # 60/40: monthly rebalanced
    if agg is not None:
        idx = spy.index.intersection(agg.index)
        s60 = spy.loc[idx].pct_change().dropna()
        b40 = agg.loc[idx].pct_change().dropna()
        idx2 = s60.index.intersection(b40.index)
        combined_ret = 0.60 * s60.loc[idx2] + 0.40 * b40.loc[idx2]
        curves["60/40 (SPY/AGG)"] = start_value * (1 + combined_ret).cumprod()

    # Equal weight 5-asset
    available = {}
    for t, ps in [("SPY", spy), ("IWM", iwm), ("EFA", efa), ("AGG", agg), ("GLD", gld)]:
        if ps is not None:
            available[t] = ps

    if len(available) >= 3:
        # Common index
        all_idx = None
        for ps in available.values():
            idx = ps.pct_change().dropna().index
            all_idx = idx if all_idx is None else all_idx.intersection(idx)

        n_assets = len(available)
        eq_ret = sum(ps.pct_change().reindex(all_idx) for ps in available.values()) / n_assets
        eq_ret = eq_ret.dropna()
        curves[f"Equal Weight ({n_assets} assets)"] = start_value * (1 + eq_ret).cumprod()

    return curves


def compute_benchmark_comparison(
    strategy_equity: pd.Series,
    benchmark_curves: Dict[str, pd.Series],
) -> List[BenchmarkResult]:
    """
    Compare strategy against all benchmarks.

    Returns list of BenchmarkResult objects, sorted by Sharpe descending.
    """
    results = []

    # Strategy first
    strat_m = _metrics(strategy_equity)
    if not strat_m:
        return results

    strat_spy_ann = None

    # Add strategy
    results.append(BenchmarkResult(
        name="OUR STRATEGY",
        total_return=strat_m["total_return"],
        ann_return=strat_m["ann_return"],
        ann_vol=strat_m["ann_vol"],
        sharpe=strat_m["sharpe"],
        max_dd=strat_m["max_dd"],
        calmar=strat_m["calmar"],
        alpha_vs_spy=0.0,  # filled in after SPY computed
    ))

    spy_ann = None
    for name, curve in benchmark_curves.items():
        # Align to strategy dates
        idx = strategy_equity.index.intersection(curve.index)
        if len(idx) < 60:
            continue

        m = _metrics(curve.loc[idx])
        if not m:
            continue

        alpha = strat_m["ann_return"] - m["ann_return"]
        results.append(BenchmarkResult(
            name=name,
            total_return=m["total_return"],
            ann_return=m["ann_return"],
            ann_vol=m["ann_vol"],
            sharpe=m["sharpe"],
            max_dd=m["max_dd"],
            calmar=m["calmar"],
            alpha_vs_spy=round(alpha, 2),
        ))
        if "SPY" in name and "%" in name:
            spy_ann = m["ann_return"]

    # Update strategy alpha vs SPY
    if spy_ann is not None:
        results[0].alpha_vs_spy = round(strat_m["ann_return"] - spy_ann, 2)

    return sorted(results, key=lambda r: r.sharpe, reverse=True)


def format_benchmark_report(results: List[BenchmarkResult]) -> str:
    """Format benchmark comparison as ASCII table."""
    if not results:
        return "Benchmark comparison data unavailable."

    lines = [
        "=" * 95,
        "MULTI-BENCHMARK PERFORMANCE COMPARISON",
        "(Swensen 2000 diversification framework + CAPM alpha analysis)",
        "=" * 95,
        f"{'Benchmark':<28} {'AnnRet':>8} {'Vol':>6} {'Sharpe':>8} "
        f"{'MaxDD':>8} {'Calmar':>8} {'vs SPY':>8}",
        "-" * 80,
    ]
    for r in results:
        marker = " <--" if r.name == "OUR STRATEGY" else ""
        calmar_str = f"{r.calmar:.2f}" if r.calmar != float("inf") else "inf "
        alpha_str  = f"{r.alpha_vs_spy:+.1f}pp" if r.alpha_vs_spy != 0 else "  base"
        lines.append(
            f"{r.name:<28} {r.ann_return:>+7.1f}% {r.ann_vol:>5.1f}% "
            f"{r.sharpe:>+8.3f} {r.max_dd:>7.1f}% {calmar_str:>8} {alpha_str:>8}{marker}"
        )
    lines += [
        "",
        "Alpha (vs SPY): strategy annualized return minus SPY annualized return",
        "Calmar: annualized return / |max drawdown| (higher = better risk-adjusted)",
        "=" * 95,
    ]
    return "\n".join(lines)
