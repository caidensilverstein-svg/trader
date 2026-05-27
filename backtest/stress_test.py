"""
Historical stress scenario analysis.

Applies historical crisis return distributions to the current portfolio
to estimate maximum loss under each scenario. Also computes portfolio
performance during those periods using the actual backtest equity curve.

Scenarios:
  - COVID crash (Feb-Mar 2020): -34% SPY in 33 days
  - 2022 rate spike (Jan-Oct 2022): -25% SPY + bond correlation breakdown
  - 2008 GFC (Oct 2007 - Mar 2009): -57% SPY over 17 months
  - Dot-com bust (Mar 2000 - Oct 2002): -49% SPY over 30 months
  - 1987 Black Monday: -22.6% in a single day

Methodology: Historical scenario approach (BIS 2005 "Stress Testing
at Major Financial Institutions")
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class StressScenario:
    name:           str
    period:         str       # e.g. "Feb 2020 - Mar 2020"
    spy_return_pct: float     # SPY return during scenario
    spy_duration_d: int       # calendar days of scenario
    portfolio_return_pct: Optional[float]  # actual portfolio return if available
    estimated_loss_usd:   float            # estimated $ loss on $100k
    drawdown_vs_spy:      Optional[float]  # portfolio DD / SPY DD ratio (alpha capture)


# Historical market crisis parameters (SPY total return, calendar days)
HISTORICAL_SCENARIOS: List[Dict] = [
    {
        "name": "COVID Crash",
        "period": "Feb 19 - Mar 23, 2020",
        "spy_return_pct": -33.9,
        "spy_duration_d": 33,
        "start_date": "2020-02-19",
        "end_date": "2020-03-23",
    },
    {
        "name": "2022 Rate Spike",
        "period": "Jan - Oct 2022",
        "spy_return_pct": -24.5,
        "spy_duration_d": 303,
        "start_date": "2022-01-03",
        "end_date": "2022-10-12",
    },
    {
        "name": "GFC 2008",
        "period": "Oct 2007 - Mar 2009",
        "spy_return_pct": -56.8,
        "spy_duration_d": 517,
        "start_date": "2007-10-09",
        "end_date": "2009-03-09",
    },
    {
        "name": "Dot-Com Bust",
        "period": "Mar 2000 - Oct 2002",
        "spy_return_pct": -49.1,
        "spy_duration_d": 929,
        "start_date": "2000-03-24",
        "end_date": "2002-10-09",
    },
    {
        "name": "Black Monday",
        "period": "Oct 19, 1987",
        "spy_return_pct": -22.6,
        "spy_duration_d": 1,
        "start_date": None,  # pre-backtest
        "end_date": None,
    },
    {
        "name": "Euro Crisis",
        "period": "May - Oct 2011",
        "spy_return_pct": -21.6,
        "spy_duration_d": 154,
        "start_date": "2011-04-29",
        "end_date": "2011-10-03",
    },
    {
        "name": "Flash Crash",
        "period": "May 6, 2010",
        "spy_return_pct": -9.2,
        "spy_duration_d": 1,
        "start_date": "2010-05-06",
        "end_date": "2010-05-06",
    },
]


def run_stress_scenarios(
    equity_curve: pd.Series,
    beta: float = 0.75,
    initial_value: float = 100_000.0,
) -> List[StressScenario]:
    """
    Estimate portfolio performance under each historical stress scenario.

    For scenarios within the backtest period, uses actual equity curve returns.
    For historical scenarios outside the period, estimates using portfolio beta.

    Parameters
    ----------
    equity_curve  : Backtest daily equity values indexed by date
    beta          : Portfolio market beta (0.75 for 75% equity allocation)
    initial_value : Current portfolio size in $

    Returns
    -------
    List of StressScenario objects, sorted by estimated loss
    """
    results = []
    equity_curve = equity_curve.copy()
    equity_curve.index = pd.to_datetime(equity_curve.index)

    for sc in HISTORICAL_SCENARIOS:
        start = sc.get("start_date")
        end   = sc.get("end_date")
        spy_ret = sc["spy_return_pct"]

        port_ret = None
        if start and end:
            try:
                s = pd.Timestamp(start)
                e = pd.Timestamp(end)
                seg = equity_curve[(equity_curve.index >= s) & (equity_curve.index <= e)]
                if len(seg) >= 2:
                    port_ret = round(float(seg.iloc[-1] / seg.iloc[0] - 1) * 100, 2)
            except Exception:
                pass

        if port_ret is None:
            # Estimate: portfolio_return ≈ beta * spy_return (downside beta)
            # Add 20% cushion from defensive mechanisms (regime detection, BSC)
            defensive_factor = 0.80 if spy_ret < -15 else 0.90
            port_ret_est = spy_ret * beta * defensive_factor
        else:
            port_ret_est = port_ret

        loss_usd = initial_value * (port_ret_est / 100)
        ratio = round(port_ret_est / spy_ret, 3) if spy_ret != 0 else None

        results.append(StressScenario(
            name=sc["name"],
            period=sc["period"],
            spy_return_pct=spy_ret,
            spy_duration_d=sc["spy_duration_d"],
            portfolio_return_pct=port_ret,  # actual if available
            estimated_loss_usd=round(loss_usd, 0),
            drawdown_vs_spy=ratio,
        ))

    return sorted(results, key=lambda s: s.estimated_loss_usd)


def stress_test_summary(scenarios: List[StressScenario]) -> Dict:
    """Aggregate summary of stress test results."""
    if not scenarios:
        return {}

    losses = [s.estimated_loss_usd for s in scenarios]
    worst = min(scenarios, key=lambda s: s.estimated_loss_usd)
    best_capture = [s for s in scenarios if s.drawdown_vs_spy is not None]

    avg_ratio = (float(np.mean([s.drawdown_vs_spy for s in best_capture]))
                 if best_capture else None)

    return {
        "n_scenarios":      len(scenarios),
        "worst_scenario":   worst.name,
        "worst_loss_usd":   worst.estimated_loss_usd,
        "worst_loss_pct":   round(worst.estimated_loss_usd / 100_000 * 100, 1),
        "avg_ratio_to_spy": round(avg_ratio, 3) if avg_ratio else None,
    }


def format_stress_report(scenarios: List[StressScenario]) -> str:
    """Format stress test results as ASCII table."""
    if not scenarios:
        return "No stress test data available."

    smry = stress_test_summary(scenarios)

    lines = [
        "=" * 90,
        "HISTORICAL STRESS TEST SCENARIOS",
        "(Portfolio behavior in major market crises)",
        "=" * 90,
        f"{'Scenario':<18} {'Period':<25} {'SPY Ret':>8} {'Port Ret':>9} "
        f"{'Est Loss':>10} {'Ratio':>7}",
        "-" * 80,
    ]
    for s in scenarios:
        port_str = f"{s.portfolio_return_pct:+.1f}%" if s.portfolio_return_pct is not None else f"est {s.estimated_loss_usd/1000:+.0f}k"
        ratio_str = f"{s.drawdown_vs_spy:.2f}x" if s.drawdown_vs_spy else "---"
        lines.append(
            f"{s.name:<18} {s.period:<25} {s.spy_return_pct:>+7.1f}% {port_str:>9} "
            f"${s.estimated_loss_usd:>+8,.0f} {ratio_str:>7}"
        )
    lines += [
        "",
        f"Worst scenario: {smry['worst_scenario']} (estimated loss: "
        f"${smry['worst_loss_usd']:+,.0f} = {smry['worst_loss_pct']:+.1f}%)",
    ]
    if smry.get("avg_ratio_to_spy"):
        lines.append(
            f"Avg portfolio/SPY ratio: {smry['avg_ratio_to_spy']:.2f}x "
            f"(< 1.0 means portfolio lost LESS than SPY)"
        )
    lines += [
        "",
        "Note: Portfolio return = actual backtest result where period overlaps;",
        "      otherwise estimated as beta * SPY_return * 0.80 (defensive factor).",
        "=" * 90,
    ]
    return "\n".join(lines)
