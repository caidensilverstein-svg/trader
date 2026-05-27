"""
Factor exposure decomposition via OLS regression.

Regresses portfolio daily returns against systematic factor proxies to
estimate beta loadings (factor exposures):
  - Market beta (SPY as market proxy)
  - Size factor (IWM/SPY spread as size proxy)
  - Value factor (IVE/IVW spread as value proxy)
  - Momentum factor (MTUM price change momentum)
  - Quality factor (QUAL as quality proxy)
  - Low volatility factor (USMV as low-vol proxy)

Methodology: Fama & French (1993, 2015), Carhart (1997) 4-factor extension.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

FACTOR_TICKERS = {
    "market":   "SPY",
    "size":     "IWM",     # small-cap proxy (relative to SPY = market)
    "value":    "IVE",     # S&P 500 value ETF
    "growth":   "IVW",     # S&P 500 growth ETF
    "momentum": "MTUM",    # iShares momentum factor ETF
    "quality":  "QUAL",    # iShares quality factor ETF
    "low_vol":  "USMV",    # iShares min-vol ETF
}


@dataclass
class FactorExposure:
    factor:      str
    beta:        float   # regression coefficient
    t_stat:      float   # t-statistic
    significant: bool    # |t-stat| > 2 (approx 95% CI)
    proxy:       str     # ticker used as factor proxy


def compute_factor_exposures(
    portfolio_returns: pd.Series,
    factor_data: Dict[str, pd.Series],
    min_overlap: int = 63,
) -> List[FactorExposure]:
    """
    Run OLS regression of portfolio returns on factor returns.

    Parameters
    ----------
    portfolio_returns : Daily portfolio returns (not %)
    factor_data       : {factor_name: daily_returns_series}
    min_overlap       : Minimum overlapping observations required

    Returns
    -------
    List of FactorExposure objects, one per factor
    """
    if portfolio_returns is None or len(portfolio_returns) < min_overlap:
        return []

    # Align all series on common dates
    combined = pd.DataFrame({"portfolio": portfolio_returns})
    for name, series in factor_data.items():
        combined[name] = series
    combined = combined.dropna()

    if len(combined) < min_overlap:
        logger.warning("Only %d overlapping obs for factor regression", len(combined))
        return []

    y = combined["portfolio"].values
    X_raw = combined.drop(columns=["portfolio"]).values
    n, k = X_raw.shape

    # Add intercept
    X = np.column_stack([np.ones(n), X_raw])

    try:
        betas, residuals, rank, _ = np.linalg.lstsq(X, y, rcond=None)
    except np.linalg.LinAlgError:
        return []

    # Compute standard errors
    y_hat = X @ betas
    resid = y - y_hat
    df_err = n - k - 1
    if df_err < 1:
        return []
    mse = float(np.sum(resid ** 2) / df_err)
    XtX_inv = np.linalg.pinv(X.T @ X)
    se = np.sqrt(mse * np.diag(XtX_inv))

    factor_names = list(combined.drop(columns=["portfolio"]).columns)
    results = []
    for i, fname in enumerate(factor_names):
        beta_i = float(betas[i + 1])    # skip intercept
        se_i   = float(se[i + 1])
        t_stat = beta_i / se_i if se_i > 0 else 0.0
        results.append(FactorExposure(
            factor=fname,
            beta=round(beta_i, 4),
            t_stat=round(t_stat, 2),
            significant=(abs(t_stat) >= 2.0),
            proxy=FACTOR_TICKERS.get(fname, fname),
        ))

    return sorted(results, key=lambda x: abs(x.t_stat), reverse=True)


def build_factor_returns(
    factor_prices: Dict[str, pd.Series],
    market_prices: Optional[pd.Series] = None,
) -> Dict[str, pd.Series]:
    """
    Convert factor price series into factor return series.
    SMB = IWM return - SPY return (size minus market)
    HML = IVE return - IVW return (value minus growth)
    Others: raw ETF returns
    """
    factor_rets = {}
    pct = {name: prices.pct_change().dropna()
           for name, prices in factor_prices.items() if prices is not None}

    # Pure market beta
    if "market" in pct:
        factor_rets["market_beta"] = pct["market"]

    # SMB: small minus big
    if "size" in pct and "market" in pct:
        idx = pct["size"].index.intersection(pct["market"].index)
        factor_rets["size_smb"] = pct["size"].loc[idx] - pct["market"].loc[idx]

    # HML: value minus growth
    if "value" in pct and "growth" in pct:
        idx = pct["value"].index.intersection(pct["growth"].index)
        factor_rets["value_hml"] = pct["value"].loc[idx] - pct["growth"].loc[idx]

    # Momentum (raw factor ETF return)
    if "momentum" in pct:
        factor_rets["momentum"] = pct["momentum"]

    # Quality (raw factor ETF return)
    if "quality" in pct:
        factor_rets["quality"] = pct["quality"]

    # Low-vol (raw factor ETF return)
    if "low_vol" in pct:
        factor_rets["low_vol"] = pct["low_vol"]

    return factor_rets


def format_factor_report(
    exposures: List[FactorExposure],
    portfolio_label: str = "Portfolio",
) -> str:
    """Format factor exposures as ASCII table."""
    if not exposures:
        return "Factor exposure data unavailable."

    lines = [
        "=" * 70,
        f"FACTOR EXPOSURE DECOMPOSITION -- {portfolio_label}",
        "(OLS Regression: Fama-French 5-Factor + Momentum)",
        "=" * 70,
        f"{'Factor':<18} {'Beta':>8} {'T-Stat':>8} {'Sig':>5} {'Proxy ETF':>12}",
        "-" * 55,
    ]
    for e in exposures:
        sig = "***" if abs(e.t_stat) >= 3.0 else ("** " if abs(e.t_stat) >= 2.0 else "   ")
        lines.append(
            f"{e.factor:<18} {e.beta:>+8.4f} {e.t_stat:>+8.2f} {sig:>5} {e.proxy:>12}"
        )
    lines += [
        "",
        "Significance: *** |t| >= 3.0   ** |t| >= 2.0  (approx 99% / 95%)",
        "Beta > 0: portfolio co-moves positively with factor",
        "Beta < 0: portfolio hedges against factor",
        "=" * 70,
    ]
    return "\n".join(lines)
