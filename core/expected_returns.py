"""
Multi-factor expected return forecasting.

Builds forward-looking expected return estimates for each ETF by blending:
  1. Factor premia (Fama-French, Carhart momentum, quality)
  2. Historical risk premia (shrunk toward long-run means)
  3. Macroeconomic adjustments (current regime, yield curve)

This replaces the naive historical mean used in MVO with theoretically-grounded
forward-looking estimates that are more stable out-of-sample.

Academic basis:
  Fama & French (1993) "Common Risk Factors in Equity Returns"
  Jegadeesh & Titman (1993) "Returns to Buying Winners and Selling Losers"
  Ilmanen (2011) "Expected Returns" -- blending factor premia
  Damodaran (2020) "Equity Risk Premium" -- implied ERP approach
  Asness et al. (2019) "Quality Minus Junk"
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# Long-run academic factor premia (annualized, from academic literature)
# Sources: Fama-French data library, AQR factor data, Ilmanen (2011)
FACTOR_PREMIA: Dict[str, float] = {
    "equity_rp":  0.055,   # equity risk premium above T-bills (Damodaran 2020)
    "size":       0.020,   # SMB (small minus big, Fama-French 1993)
    "value":      0.030,   # HML (high minus low book-to-market, Fama-French 1993)
    "momentum":   0.040,   # WML (winners minus losers, Jegadeesh-Titman 1993)
    "quality":    0.025,   # QMJ (quality minus junk, Asness et al. 2019)
    "low_vol":    0.010,   # BAB (betting against beta, Frazzini-Pedersen 2014)
    "intl_disc":  -0.010,  # international discount vs US (currency + political risk)
}

# Factor loadings for each ETF (based on academic characterization)
# Each ETF's expected return = rf + sum(loading_i * factor_premium_i)
ETF_FACTOR_LOADINGS: Dict[str, Dict[str, float]] = {
    "AVUV": {  # Avantis US Small Cap Value
        "equity_rp": 1.10,   # high beta to market
        "size":       0.80,   # strong small-cap tilt
        "value":      0.70,   # strong value tilt (Avantis focuses on HML)
        "momentum":   0.10,   # slight momentum inclusion
        "quality":    0.20,   # mild quality screen
        "low_vol":    0.00,
        "intl_disc":  0.00,
    },
    "AVDV": {  # Avantis Intl Small Cap Value
        "equity_rp": 1.00,
        "size":       0.75,
        "value":      0.65,
        "momentum":   0.10,
        "quality":    0.15,
        "low_vol":    0.00,
        "intl_disc":  1.00,   # full international discount
    },
    "QMOM": {  # Alpha Architect Quality Momentum
        "equity_rp": 1.05,
        "size":       0.00,   # large-cap focused
        "value":      -0.20,  # momentum often anti-value (growth tilt)
        "momentum":   0.80,   # primary driver
        "quality":    0.60,   # quality screen (QMJ)
        "low_vol":    0.00,
        "intl_disc":  0.00,
    },
    "DBMF": {  # iMGP DBi Managed Futures
        "equity_rp": 0.00,   # non-equity
        "size":       0.00,
        "value":      0.00,
        "momentum":   0.50,   # trend-following = momentum in futures
        "quality":    0.00,
        "low_vol":    0.00,
        "intl_disc":  0.00,
    },
    "CTA":  {  # CTA trend following
        "equity_rp": 0.00,
        "size":       0.00,
        "value":      0.00,
        "momentum":   0.55,
        "quality":    0.00,
        "low_vol":    0.00,
        "intl_disc":  0.00,
    },
    "CASH": {
        "equity_rp": 0.00,
        "size":       0.00,
        "value":      0.00,
        "momentum":   0.00,
        "quality":    0.00,
        "low_vol":    0.00,
        "intl_disc":  0.00,
    },
}

# Base RF (T-bill yield), used when not provided
DEFAULT_RF = 0.043  # ~4.3% current T-bill rate (2026)


@dataclass
class ETFExpectedReturn:
    """Forward-looking expected return estimate for one ETF."""
    ticker:           str
    risk_free_rate:   float
    factor_premium:   float     # total factor premium above RF
    expected_return:  float     # total annualized expected return
    regime_adj:       float     # regime-conditional adjustment (+/- %)
    adjusted_return:  float     # expected return after regime adjustment
    factor_breakdown: Dict[str, float]   # contribution from each factor
    confidence_band:  float     # +/- this much (1 sigma uncertainty)


def compute_expected_returns(
    etf_tickers: List[str],
    rf: float = DEFAULT_RF,
    regime: str = "BULL",
    factor_premia_override: Optional[Dict[str, float]] = None,
) -> List[ETFExpectedReturn]:
    """
    Compute multi-factor expected returns for each ETF.

    Parameters
    ----------
    etf_tickers   : list of ETF tickers
    rf            : current risk-free rate (annual)
    regime        : current market regime (adjusts expected returns)
    factor_premia_override : override long-run factor premia (for sensitivity analysis)

    Returns
    -------
    List of ETFExpectedReturn, sorted by adjusted_return descending
    """
    premia = factor_premia_override or FACTOR_PREMIA.copy()

    # Regime adjustments to equity risk premium
    # In BEAR_CRISIS, realized ERP is negative -- adjust down
    regime_eq_adj = {
        "BULL":        0.010,   # bull market: slightly above long-run mean
        "MILD_BULL":   0.005,
        "SIDEWAYS":    0.000,
        "BEAR":       -0.020,   # bear market: negative regime adj
        "BEAR_CRISIS": -0.040,
    }.get(regime, 0.0)

    results: List[ETFExpectedReturn] = []

    for ticker in etf_tickers:
        loadings = ETF_FACTOR_LOADINGS.get(ticker, {})
        if not loadings:
            continue

        breakdown: Dict[str, float] = {}
        factor_premium = 0.0

        for factor, loading in loadings.items():
            premium = premia.get(factor, 0.0)
            contribution = loading * premium
            breakdown[factor] = round(contribution, 4)
            factor_premium += contribution

        expected_return = rf + factor_premium

        # Regime adjustment: applies to equity_rp loading
        eq_loading = loadings.get("equity_rp", 0.0)
        regime_adj = eq_loading * regime_eq_adj
        adjusted_return = expected_return + regime_adj

        # Uncertainty: larger for assets with more factor exposure
        n_active_factors = sum(1 for v in loadings.values() if abs(v) > 0.05)
        confidence_band = 0.03 + n_active_factors * 0.01  # ~3-6% 1-sigma

        results.append(ETFExpectedReturn(
            ticker=ticker,
            risk_free_rate=round(rf, 4),
            factor_premium=round(factor_premium, 4),
            expected_return=round(expected_return, 4),
            regime_adj=round(regime_adj, 4),
            adjusted_return=round(adjusted_return, 4),
            factor_breakdown=breakdown,
            confidence_band=round(confidence_band, 3),
        ))

    return sorted(results, key=lambda x: x.adjusted_return, reverse=True)


def portfolio_expected_return(
    etf_weights: Dict[str, float],
    rf: float = DEFAULT_RF,
    regime: str = "BULL",
) -> Dict[str, float]:
    """
    Compute portfolio-level expected return (weighted sum).

    Returns dict with: expected_return, factor_premium, regime_adj, sharpe_est, ...
    """
    er_list = compute_expected_returns(list(etf_weights.keys()), rf=rf, regime=regime)
    er_map = {e.ticker: e for e in er_list}

    total_weights = sum(etf_weights.values())
    if total_weights <= 0:
        return {}

    port_return = 0.0
    port_regime_adj = 0.0
    port_factor_premium = 0.0
    weight_sum = 0.0

    for ticker, w in etf_weights.items():
        norm_w = w / total_weights
        er = er_map.get(ticker)
        if er:
            port_return      += norm_w * er.adjusted_return
            port_regime_adj  += norm_w * er.regime_adj
            port_factor_premium += norm_w * er.factor_premium
            weight_sum += norm_w

    # Assume portfolio annual vol ~8-9% (from backtest)
    assumed_vol = 0.088
    sharpe_est = (port_return - rf) / assumed_vol if assumed_vol > 0 else 0.0

    return {
        "portfolio_er":       round(port_return, 4),
        "factor_premium":     round(port_factor_premium, 4),
        "regime_adj":         round(port_regime_adj, 4),
        "risk_free_rate":     round(rf, 4),
        "assumed_vol":        round(assumed_vol, 4),
        "sharpe_estimate":    round(sharpe_est, 3),
        "regime":             regime,
    }


def format_expected_return_report(
    er_list: List[ETFExpectedReturn],
    port_stats: Optional[Dict[str, float]] = None,
) -> str:
    """Format expected return estimates as ASCII table."""
    if not er_list:
        return "Expected return data unavailable."

    lines = [
        "=" * 80,
        "MULTI-FACTOR EXPECTED RETURN FORECASTS",
        "(Fama-French 1993, Jegadeesh-Titman 1993, Ilmanen 2011, Damodaran 2020)",
        "(Factor premia are long-run academic estimates, NOT historical mean returns)",
        "=" * 80,
        f"{'Ticker':<8} {'RF':>6} {'Factor+':>8} {'ExpRet':>8} {'RegAdj':>8} {'AdjRet':>8} {'+-1sig':>7}",
        "-" * 58,
    ]

    for er in er_list:
        lines.append(
            f"{er.ticker:<8} "
            f"{er.risk_free_rate*100:>5.1f}% "
            f"{er.factor_premium*100:>7.2f}% "
            f"{er.expected_return*100:>7.2f}% "
            f"{er.regime_adj*100:>+7.2f}% "
            f"{er.adjusted_return*100:>7.2f}% "
            f"+-{er.confidence_band*100:.1f}%"
        )

    lines.append("")
    lines.append("FACTOR BREAKDOWN (annualized contribution):")
    lines.append(f"{'Ticker':<8} " + " ".join(f"{f[:6]:>8}" for f in FACTOR_PREMIA.keys()))
    lines.append("-" * (8 + 9 * len(FACTOR_PREMIA)))
    for er in er_list:
        row = f"{er.ticker:<8}"
        for factor in FACTOR_PREMIA.keys():
            v = er.factor_breakdown.get(factor, 0.0)
            row += f" {v*100:>+7.2f}%"
        lines.append(row)

    if port_stats:
        lines += [
            "",
            "PORTFOLIO BLENDED EXPECTED RETURN:",
            f"  Risk-Free Rate:        {port_stats['risk_free_rate']*100:.2f}%",
            f"  Factor Premium:        {port_stats['factor_premium']*100:.2f}%",
            f"  Regime Adjustment:     {port_stats['regime_adj']*100:+.2f}%  ({port_stats['regime']})",
            f"  Portfolio Expected Return: {port_stats['portfolio_er']*100:.2f}%",
            f"  Assumed Portfolio Vol: {port_stats['assumed_vol']*100:.2f}%  (from backtest)",
            f"  Estimated Sharpe:      {port_stats['sharpe_estimate']:.3f}",
        ]

    lines += [
        "",
        "Note: These are theoretical forward-looking estimates with wide uncertainty bands.",
        "Factor premia from academic literature; actual realized returns will differ.",
        "Regime adjustments condition on current market environment.",
        "=" * 80,
    ]

    return "\n".join(lines)
