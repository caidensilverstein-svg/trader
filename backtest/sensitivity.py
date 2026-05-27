"""
Parameter sensitivity analysis for the ETF sleeve backtest.

Tests how key parameters affect strategy performance so we can identify
which assumptions the strategy is robust to and which are fragile.

Parameters scanned:
  - drift_threshold: [0.03, 0.04, 0.05, 0.06, 0.07]
  - bsc_lookback:    [63, 84, 126]
  - bsc_target_vol:  [0.10, 0.12, 0.15]
  - bsc_min_scalar:  [0.40, 0.50, 0.60]

Methodology:
  - Each parameter is varied independently (one-at-a-time)
  - Baseline is the production configuration
  - Results compared by Calmar ratio and max drawdown
  - Sensitive parameters flagged when result changes > 20% from baseline

Academic basis: Pardo (2008) "The Evaluation and Optimization of Trading Strategies"
"""

import itertools
import logging
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Tuple, Optional

import config
from backtest.engine import run_backtest

logger = logging.getLogger(__name__)


@dataclass
class SensitivityResult:
    param_name:   str
    param_value:  float
    calmar:       float
    max_dd:       float
    total_return: float
    n_rebalances: int
    calmar_delta: float = 0.0   # vs baseline
    dd_delta:     float = 0.0   # vs baseline


def _extract_metrics(result: dict) -> dict:
    """Extract flat metrics from the nested backtest result dict."""
    summary = result.get("summary", {})
    strat   = summary.get("strategy", {})
    return {
        "calmar":       strat.get("calmar", 0) or 0,
        "max_drawdown": (strat.get("max_dd", 0) or 0) / 100,  # convert pct to fraction
        "total_return": (strat.get("total_return", 0) or 0) / 100,
        "n_rebalances": summary.get("rebalance_count", 0),
    }


_PARAM_TO_KWARG = {
    "REBALANCE_DRIFT_THRESHOLD": "drift_threshold",
}


def _run_with_override(param_name: str, param_value: float) -> dict:
    """
    Run backtest with a single parameter overridden.

    Parameters that map directly to run_backtest() kwargs are passed as kwargs.
    Parameters that must come from config are temporarily overridden there.
    """
    if param_name in _PARAM_TO_KWARG:
        result = run_backtest(**{_PARAM_TO_KWARG[param_name]: param_value})
        return _extract_metrics(result)

    # Config override path (BSC parameters etc.)
    original = getattr(config, param_name, None)
    try:
        setattr(config, param_name, param_value)
        result = run_backtest()
        return _extract_metrics(result)
    finally:
        if original is not None:
            setattr(config, param_name, original)


def run_one_at_a_time(
    parameters: Optional[Dict[str, List[float]]] = None,
) -> Dict[str, List[SensitivityResult]]:
    """
    One-at-a-time sensitivity analysis.

    Varies each parameter across its range while holding others at baseline.

    Parameters
    ----------
    parameters : {config_attr_name: [value1, value2, ...]}
                 Defaults to production-relevant parameter set.

    Returns
    -------
    dict : {param_name: [SensitivityResult, ...]}
    """
    if parameters is None:
        parameters = {
            "REBALANCE_DRIFT_THRESHOLD": [0.03, 0.04, 0.05, 0.06, 0.07],
            "BSC_LOOKBACK_DAYS":         [63, 84, 126],
            "BSC_TARGET_VOL":            [0.10, 0.12, 0.15],
            "BSC_MIN_SCALAR":            [0.40, 0.50, 0.60],
        }

    logger.info("Running sensitivity analysis over %d parameter sets...",
                sum(len(v) for v in parameters.values()))

    # Get baseline
    baseline_raw    = run_backtest()
    baseline        = _extract_metrics(baseline_raw)
    baseline_calmar = baseline["calmar"]
    baseline_dd     = baseline["max_drawdown"]
    logger.info("Baseline: Calmar=%.3f  MaxDD=%.2f%%",
                baseline_calmar, baseline_dd * 100)

    results: Dict[str, List[SensitivityResult]] = {}

    for param_name, values in parameters.items():
        param_results = []
        for val in values:
            try:
                res    = _run_with_override(param_name, val)
                calmar = res["calmar"]
                dd     = res["max_drawdown"]
                sr = SensitivityResult(
                    param_name   = param_name,
                    param_value  = val,
                    calmar       = round(calmar, 3),
                    max_dd       = round(dd, 4),
                    total_return = round(res["total_return"], 4),
                    n_rebalances = res["n_rebalances"],
                    calmar_delta = round((calmar - baseline_calmar) / max(abs(baseline_calmar), 1e-6), 3),
                    dd_delta     = round(dd - baseline_dd, 4),
                )
                logger.debug("  %s=%s: Calmar=%.3f (delta %+.1f%%)",
                             param_name, val, calmar, sr.calmar_delta * 100)
                param_results.append(sr)
            except Exception as exc:
                logger.error("Sensitivity run failed for %s=%s: %s", param_name, val, exc)

        results[param_name] = param_results

    return results


def fragile_parameters(
    results: Dict[str, List[SensitivityResult]],
    threshold: float = 0.20,
) -> List[str]:
    """
    Identify parameters where any value causes Calmar to change > threshold.

    Parameters
    ----------
    threshold : Fractional change in Calmar (default 20%)

    Returns
    -------
    List of parameter names that are 'fragile'.
    """
    fragile = []
    for param, param_results in results.items():
        max_delta = max(abs(r.calmar_delta) for r in param_results)
        if max_delta > threshold:
            fragile.append(param)
    return fragile


def format_sensitivity_report(
    results: Dict[str, List[SensitivityResult]],
    baseline_calmar: float,
) -> str:
    """Format sensitivity analysis as ASCII table."""
    lines = [
        "=" * 70,
        "PARAMETER SENSITIVITY ANALYSIS",
        f"Baseline Calmar: {baseline_calmar:.3f}",
        "=" * 70,
        "",
    ]

    for param, param_results in results.items():
        lines.append(f"Parameter: {param}")
        lines.append(f"{'Value':>10} {'Calmar':>8} {'CalmarDelta':>12} {'MaxDD':>8} {'NRebal':>7}")
        lines.append("-" * 50)
        for r in param_results:
            prod_flag = " <-- PROD" if abs(r.calmar_delta) < 0.001 else ""
            lines.append(
                f"{r.param_value:>10.4f} {r.calmar:>8.3f} {r.calmar_delta:>+11.1%} "
                f"{r.max_dd:>8.1%} {r.n_rebalances:>7}{prod_flag}"
            )
        lines.append("")

    fragile = fragile_parameters(results)
    if fragile:
        lines.append(f"FRAGILE PARAMETERS (>20% Calmar change): {', '.join(fragile)}")
    else:
        lines.append("ROBUST: No parameter causes >20% change in Calmar ratio")

    lines += ["", "=" * 70]
    return "\n".join(lines)
