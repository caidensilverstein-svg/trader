"""
Iron Condor Greeks calculation and monitoring.

Computes theoretical Black-Scholes Greeks for the iron condor positions:
  Delta, Gamma, Theta, Vega for each leg and net position.

Since Alpaca paper trading doesn't support options, these are SIGNAL-ONLY
positions tracked for learning and reporting. Greeks help understand:
  - Delta: directional exposure (we want near-zero for neutral condors)
  - Theta: daily time decay earned (positive for short condors)
  - Vega: volatility exposure (negative for short condors)
  - Gamma: rate of delta change (negative = risk of large moves)

Academic basis:
  Black & Scholes (1973) "The Pricing of Options and Corporate Liabilities"
  Taleb (1997) "Dynamic Hedging" -- practical Greeks implementation
"""

import logging
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def _norm_cdf(x: float) -> float:
    """Standard normal CDF."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_pdf(x: float) -> float:
    """Standard normal PDF."""
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def bs_call_greeks(
    S: float,    # underlying price
    K: float,    # strike
    T: float,    # time to expiry (years)
    r: float,    # risk-free rate (annual)
    sigma: float, # implied volatility (annual)
) -> Dict[str, float]:
    """Black-Scholes call option Greeks."""
    if T <= 0 or sigma <= 0:
        return {"delta": 1.0 if S > K else 0.0, "gamma": 0.0,
                "theta": 0.0, "vega": 0.0, "price": max(0, S - K)}

    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    delta = _norm_cdf(d1)
    gamma = _norm_pdf(d1) / (S * sigma * math.sqrt(T))
    theta = (-(S * _norm_pdf(d1) * sigma / (2 * math.sqrt(T))) - r * K * math.exp(-r * T) * _norm_cdf(d2)) / 365
    vega  = S * _norm_pdf(d1) * math.sqrt(T) / 100  # per 1% change in vol
    price = S * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)

    return {
        "delta": round(delta, 4),
        "gamma": round(gamma, 6),
        "theta": round(theta, 4),
        "vega":  round(vega, 4),
        "price": round(max(0, price), 2),
    }


def bs_put_greeks(
    S: float, K: float, T: float, r: float, sigma: float
) -> Dict[str, float]:
    """Black-Scholes put option Greeks (via put-call parity)."""
    call = bs_call_greeks(S, K, T, r, sigma)
    delta = call["delta"] - 1.0
    return {
        "delta": round(delta, 4),
        "gamma": call["gamma"],
        "theta": call["theta"],
        "vega":  call["vega"],
        "price": round(max(0, call["price"] - S + K * math.exp(-r * T)), 2),
    }


@dataclass
class CondorGreeks:
    """Net Greeks for a 4-leg iron condor position."""
    underlying:     str
    spot:           float
    put_long_K:     float
    put_short_K:    float
    call_short_K:   float
    call_long_K:    float
    dte:            int       # days to expiry
    iv:             float     # implied vol
    net_delta:      float
    net_gamma:      float
    net_theta:      float     # daily $ earned from time decay
    net_vega:       float     # $ change per 1% IV move
    max_profit:     float     # $ credit received
    max_loss:       float     # $ max loss
    breakeven_low:  float
    breakeven_high: float
    prob_profit:    float     # theoretical probability


def compute_condor_greeks(
    underlying: str,
    spot: float,
    put_long_K: float,
    put_short_K: float,
    call_short_K: float,
    call_long_K: float,
    dte: int,
    iv: float,
    contracts: int = 1,
    r: float = 0.05,
) -> CondorGreeks:
    """
    Compute net Greeks for an iron condor position.

    Iron Condor = short put spread + short call spread
      +1 put at put_long_K (long outer put)
      -1 put at put_short_K (short inner put)
      -1 call at call_short_K (short inner call)
      +1 call at call_long_K (long outer call)
    """
    T = max(dte / 365, 1 / 365)
    mult = 100 * contracts  # standard options multiplier

    # Long put
    lp = bs_put_greeks(spot, put_long_K,  T, r, iv)
    # Short put
    sp = bs_put_greeks(spot, put_short_K, T, r, iv)
    # Short call
    sc = bs_call_greeks(spot, call_short_K, T, r, iv)
    # Long call
    lc = bs_call_greeks(spot, call_long_K,  T, r, iv)

    # Net: +lp -sp -sc +lc
    net_delta = (lp["delta"] - sp["delta"] - sc["delta"] + lc["delta"]) * mult
    net_gamma = (lp["gamma"] - sp["gamma"] - sc["gamma"] + lc["gamma"]) * mult
    net_theta = (lp["theta"] - sp["theta"] - sc["theta"] + lc["theta"]) * mult
    net_vega  = (lp["vega"]  - sp["vega"]  - sc["vega"]  + lc["vega"] ) * mult

    # P&L
    credit_received = (sp["price"] - lp["price"] + sc["price"] - lc["price"]) * mult
    put_spread_width  = (put_short_K - put_long_K) * mult
    call_spread_width = (call_long_K - call_short_K) * mult
    max_loss = -(max(put_spread_width, call_spread_width) - credit_received)

    # Breakevens
    breakeven_low  = put_short_K  - credit_received / mult
    breakeven_high = call_short_K + credit_received / mult

    # Probability of profit (spot stays between short strikes at expiry)
    d_low  = (math.log(spot / breakeven_low) + (r - 0.5 * iv**2) * T) / (iv * math.sqrt(T))
    d_high = (math.log(spot / breakeven_high) + (r - 0.5 * iv**2) * T) / (iv * math.sqrt(T))
    prob_profit = max(0.0, min(1.0, _norm_cdf(d_high) - (1 - _norm_cdf(-d_low))))

    return CondorGreeks(
        underlying=underlying,
        spot=spot,
        put_long_K=put_long_K,
        put_short_K=put_short_K,
        call_short_K=call_short_K,
        call_long_K=call_long_K,
        dte=dte,
        iv=iv,
        net_delta=round(net_delta, 4),
        net_gamma=round(net_gamma, 6),
        net_theta=round(net_theta, 2),
        net_vega=round(net_vega, 2),
        max_profit=round(credit_received, 2),
        max_loss=round(max_loss, 2),
        breakeven_low=round(breakeven_low, 2),
        breakeven_high=round(breakeven_high, 2),
        prob_profit=round(prob_profit * 100, 1),
    )


def format_condor_greeks(cg: CondorGreeks) -> str:
    """Format iron condor Greeks as ASCII report."""
    lines = [
        "=" * 65,
        f"IRON CONDOR GREEKS -- {cg.underlying}  (Signal-only, no live options)",
        "=" * 65,
        f"Spot: ${cg.spot:.2f}   IV: {cg.iv*100:.1f}%   DTE: {cg.dte}d",
        "",
        f"STRIKES:  {cg.put_long_K:.0f}P / {cg.put_short_K:.0f}P / "
        f"{cg.call_short_K:.0f}C / {cg.call_long_K:.0f}C",
        "",
        "NET GREEKS (per position):",
        f"  Delta:   {cg.net_delta:+.4f}  (near-zero = good neutrality)",
        f"  Gamma:   {cg.net_gamma:+.6f}  (negative = hurt by large moves)",
        f"  Theta:   {cg.net_theta:+.2f}/day  (positive = time decay in our favor)",
        f"  Vega:    {cg.net_vega:+.2f}/1%IV  (negative = hurt by vol spike)",
        "",
        "P&L PROFILE:",
        f"  Max Profit:      ${cg.max_profit:+.2f}  (keep if spot stays between short strikes)",
        f"  Max Loss:        ${cg.max_loss:+.2f}  (if spot blows through outer strike)",
        f"  Breakeven Low:   ${cg.breakeven_low:.2f}",
        f"  Breakeven High:  ${cg.breakeven_high:.2f}",
        f"  P(Profit):       {cg.prob_profit:.1f}%",
        "",
        "Note: Iron condors are SIGNAL-ONLY (Alpaca paper does not support options).",
        "=" * 65,
    ]
    return "\n".join(lines)
