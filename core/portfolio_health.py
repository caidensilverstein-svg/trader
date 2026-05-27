"""
Portfolio health check: aggregates all key risk metrics into a single status.

Synthesizes results from: VaR, drawdown, regime, sector exposure, liquidity,
momentum signals, and correlation into a single GREEN/YELLOW/RED/HALT status
with actionable recommendations.

This is the "dashboard" view that a portfolio manager would check daily.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class HealthCheck:
    """A single health check item."""
    category:      str
    name:          str
    value:         str
    status:        str    # GREEN, YELLOW, RED
    threshold:     str    # what the threshold is
    action:        str    # recommended action if not GREEN


@dataclass
class PortfolioHealthReport:
    """Full portfolio health report."""
    overall_status:  str                # GREEN / YELLOW / RED / HALT
    checks:          List[HealthCheck]
    n_green:         int
    n_yellow:        int
    n_red:           int
    priority_action: str                # most urgent action needed
    timestamp:       str


def run_health_checks(
    drawdown_pct:      float,         # current drawdown from peak (negative)
    var_95_pct:        float,         # 1-day VaR at 95% (as fraction)
    regime:            str,           # current market regime
    hhi_sector:        float,         # HHI sector concentration
    avg_correlation:   float,         # average pairwise ETF correlation
    portfolio_sharpe:  float,         # trailing 1-year Sharpe ratio
    momentum_penalties: int,          # number of ETFs on PENALTY signal
    days_to_exit:      float,         # liquidity: days to exit largest position
    vol_scalar:        float,         # current B-SC volatility scalar
    portfolio_value:   float = 100_000,  # current portfolio NAV
    timestamp:         str = "now",
) -> PortfolioHealthReport:
    """
    Run all portfolio health checks.

    Returns a comprehensive health report with GREEN/YELLOW/RED status for
    each dimension and an overall portfolio status.
    """
    checks: List[HealthCheck] = []

    # 1. Drawdown check
    dd_pct = drawdown_pct * 100  # convert to %
    if dd_pct > -10:
        dd_status = "GREEN"
        dd_action = "No action needed"
    elif dd_pct > -15:
        dd_status = "YELLOW"
        dd_action = "Monitor closely; no new entries"
    elif dd_pct > -20:
        dd_status = "RED"
        dd_action = "Close PEAD/M&A positions; reduce risk"
    else:
        dd_status = "RED"
        dd_action = "HALT all activity; send alert"

    checks.append(HealthCheck(
        category="Drawdown", name="Current Drawdown",
        value=f"{dd_pct:.1f}%",
        status=dd_status,
        threshold="OK:<-10% | REVIEW:-10% to -15% | REDUCE:-15% to -20% | HALT:<-20%",
        action=dd_action,
    ))

    # 2. VaR check
    var_dollar = var_95_pct * portfolio_value
    if var_dollar < 1_000:
        var_status = "GREEN"
        var_action = "VaR within target"
    elif var_dollar < 2_000:
        var_status = "YELLOW"
        var_action = "VaR elevated; monitor position sizes"
    else:
        var_status = "RED"
        var_action = "Reduce position sizes to bring VaR below $1,500"

    checks.append(HealthCheck(
        category="Risk", name="1-Day VaR (95%)",
        value=f"${var_dollar:,.0f}",
        status=var_status,
        threshold="GREEN:<$1,000 | YELLOW:<$2,000 | RED:>$2,000",
        action=var_action,
    ))

    # 3. Regime check
    if regime in ("BULL", "MILD_BULL"):
        reg_status = "GREEN"
        reg_action = "Aggressive allocation appropriate"
    elif regime == "SIDEWAYS":
        reg_status = "YELLOW"
        reg_action = "Reduce momentum ETFs, maintain value"
    elif regime == "BEAR":
        reg_status = "YELLOW"
        reg_action = "B-SC scalar and regime reduction active; hold"
    else:  # BEAR_CRISIS
        reg_status = "RED"
        reg_action = "Maximum defense; halt new trades"

    checks.append(HealthCheck(
        category="Regime", name="Market Regime",
        value=regime,
        status=reg_status,
        threshold="GREEN:BULL/MILD_BULL | YELLOW:SIDEWAYS/BEAR | RED:BEAR_CRISIS",
        action=reg_action,
    ))

    # 4. Sector concentration
    if hhi_sector < 1_500:
        hhi_status = "GREEN"
        hhi_action = "Well diversified"
    elif hhi_sector < 2_500:
        hhi_status = "YELLOW"
        hhi_action = "Monitor sector concentration"
    else:
        hhi_status = "RED"
        hhi_action = "Reduce concentrated sector ETFs"

    checks.append(HealthCheck(
        category="Diversification", name="Sector HHI",
        value=f"{hhi_sector:.0f}",
        status=hhi_status,
        threshold="GREEN:<1500 | YELLOW:<2500 | RED:>2500",
        action=hhi_action,
    ))

    # 5. Correlation check
    if avg_correlation < 0.50:
        corr_status = "GREEN"
        corr_action = "Diversification holding"
    elif avg_correlation < 0.70:
        corr_status = "YELLOW"
        corr_action = "Correlations rising; reduce overlapping positions"
    else:
        corr_status = "RED"
        corr_action = "High correlation -- diversification failing"

    checks.append(HealthCheck(
        category="Diversification", name="Avg Pairwise Correlation",
        value=f"{avg_correlation:.3f}",
        status=corr_status,
        threshold="GREEN:<0.50 | YELLOW:<0.70 | RED:>0.70",
        action=corr_action,
    ))

    # 6. Sharpe ratio check
    if portfolio_sharpe > 0.50:
        sh_status = "GREEN"
        sh_action = "Strategy performing well"
    elif portfolio_sharpe > 0.20:
        sh_status = "YELLOW"
        sh_action = "Below target Sharpe; review factor exposures"
    else:
        sh_status = "RED"
        sh_action = "Sharpe critically low; strategy review needed"

    checks.append(HealthCheck(
        category="Performance", name="Trailing Sharpe (1yr)",
        value=f"{portfolio_sharpe:.3f}",
        status=sh_status,
        threshold="GREEN:>0.50 | YELLOW:>0.20 | RED:<0.20",
        action=sh_action,
    ))

    # 7. Momentum signal check
    if momentum_penalties == 0:
        mom_status = "GREEN"
        mom_action = "All ETFs in positive momentum"
    elif momentum_penalties <= 1:
        mom_status = "YELLOW"
        mom_action = f"{momentum_penalties} ETF on PENALTY; weight reduced -20%"
    else:
        mom_status = "RED"
        mom_action = f"{momentum_penalties} ETFs on PENALTY; significant weight reduction"

    checks.append(HealthCheck(
        category="Momentum", name="ETFs on Penalty Signal",
        value=str(momentum_penalties),
        status=mom_status,
        threshold="GREEN:0 | YELLOW:1 | RED:>1",
        action=mom_action,
    ))

    # 8. Liquidity check
    if days_to_exit < 2:
        liq_status = "GREEN"
        liq_action = "Adequate liquidity"
    elif days_to_exit < 5:
        liq_status = "YELLOW"
        liq_action = "Monitor liquidity; avoid adding to concentrated positions"
    else:
        liq_status = "RED"
        liq_action = "Liquidity constraint; reduce position size"

    checks.append(HealthCheck(
        category="Liquidity", name="Days to Exit (largest position)",
        value=f"{days_to_exit:.1f}d",
        status=liq_status,
        threshold="GREEN:<2d | YELLOW:<5d | RED:>5d",
        action=liq_action,
    ))

    # 9. B-SC scalar check
    if vol_scalar >= 0.80:
        bsc_status = "GREEN"
        bsc_action = "Full allocation appropriate"
    elif vol_scalar >= 0.50:
        bsc_status = "YELLOW"
        bsc_action = f"B-SC reduced to {vol_scalar:.0%}; vol elevated vs target"
    else:
        bsc_status = "RED"
        bsc_action = f"B-SC at {vol_scalar:.0%}; significant deleveraging active"

    checks.append(HealthCheck(
        category="Sizing", name="B-SC Volatility Scalar",
        value=f"{vol_scalar:.0%}",
        status=bsc_status,
        threshold="GREEN:>80% | YELLOW:50-80% | RED:<50%",
        action=bsc_action,
    ))

    # Tally
    n_green  = sum(1 for c in checks if c.status == "GREEN")
    n_yellow = sum(1 for c in checks if c.status == "YELLOW")
    n_red    = sum(1 for c in checks if c.status == "RED")

    # Overall status
    if n_red >= 3 or dd_pct <= -20:
        overall = "HALT"
    elif n_red >= 1:
        overall = "RED"
    elif n_yellow >= 3:
        overall = "YELLOW"
    else:
        overall = "GREEN"

    # Priority action
    red_checks = [c for c in checks if c.status == "RED"]
    yellow_checks = [c for c in checks if c.status == "YELLOW"]
    if red_checks:
        priority_action = f"[RED] {red_checks[0].name}: {red_checks[0].action}"
    elif yellow_checks:
        priority_action = f"[YELLOW] {yellow_checks[0].name}: {yellow_checks[0].action}"
    else:
        priority_action = "All checks GREEN -- continue normal operations"

    return PortfolioHealthReport(
        overall_status=overall,
        checks=checks,
        n_green=n_green,
        n_yellow=n_yellow,
        n_red=n_red,
        priority_action=priority_action,
        timestamp=timestamp,
    )


def format_health_report(report: PortfolioHealthReport) -> str:
    """Format health check report as ASCII."""
    status_icon = {"GREEN": "[OK]", "YELLOW": "[!!]", "RED": "[XX]", "HALT": "[!!HALT!!]"}
    icon = status_icon.get(report.overall_status, "[??]")

    lines = [
        "=" * 75,
        f"PORTFOLIO HEALTH CHECK  {report.timestamp}",
        f"OVERALL STATUS: {icon} {report.overall_status}",
        f"  Green: {report.n_green} | Yellow: {report.n_yellow} | Red: {report.n_red}",
        f"  Priority Action: {report.priority_action}",
        "=" * 75,
        f"{'Category':<18} {'Check':<28} {'Value':>10} {'Status':>8}",
        "-" * 68,
    ]

    for c in report.checks:
        icon_c = status_icon.get(c.status, "[??]")
        lines.append(
            f"{c.category:<18} {c.name:<28} {c.value:>10} {icon_c:>8}"
        )
        if c.status != "GREEN":
            lines.append(f"  Action: {c.action}")
            lines.append(f"  Threshold: {c.threshold}")

    lines += [
        "=" * 75,
    ]

    return "\n".join(lines)
