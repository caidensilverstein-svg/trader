"""
Email reporting module.

Sends weekly status emails and alert emails.
Uses Gmail SMTP with app password.
All content is plain ASCII — no Unicode special characters to avoid
rendering issues in email clients.
"""

import logging
import smtplib
from datetime import datetime, timezone
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Dict, List, Optional

import config

logger = logging.getLogger(__name__)


def _send_email(subject: str, body: str, attachments: Optional[List[str]] = None) -> bool:
    """
    Send an email via Gmail SMTP.

    Parameters
    ----------
    subject     : Email subject line (ASCII only)
    body        : Plain text body (ASCII only)
    attachments : Optional list of file paths to attach

    Returns
    -------
    bool : True if sent successfully
    """
    # Sanitize: replace any problematic characters
    subject = subject.encode("ascii", "replace").decode("ascii")
    body    = body.encode("ascii", "replace").decode("ascii")

    msg = MIMEMultipart()
    msg["From"]    = config.EMAIL_SENDER
    msg["To"]      = config.EMAIL_RECEIVER
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    if attachments:
        for fpath in attachments:
            p = Path(fpath)
            if p.exists():
                with open(p, "rb") as f:
                    part = MIMEApplication(f.read(), Name=p.name)
                part["Content-Disposition"] = f'attachment; filename="{p.name}"'
                msg.attach(part)
                logger.debug("Attached %s (%.1f KB)", p.name, p.stat().st_size / 1024)
            else:
                logger.warning("Attachment not found: %s", fpath)

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as srv:
            srv.login(config.EMAIL_SENDER, config.EMAIL_PASSWORD)
            srv.sendmail(config.EMAIL_SENDER, config.EMAIL_RECEIVER, msg.as_string())
        logger.info("Email sent: %s", subject)
        return True
    except Exception as exc:
        logger.error("Email send failed: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Weekly report
# ---------------------------------------------------------------------------

def format_weekly_report(
    regime_data: dict,
    account_data: dict,
    etf_status: dict,
    condor_status: dict,
    pead_status: dict,
    ma_status: dict,
    trade_count: int = 0,
) -> str:
    """
    Build the weekly report body (plain ASCII, human readable).

    No em-dashes, no fancy chars. Just clean text.
    """
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    bsc  = etf_status.get("bsc_scalar", "N/A")
    qmom = etf_status.get("eff_qmom_wt", "N/A")

    # Account section
    equity     = account_data.get("equity", 0)
    cash       = account_data.get("cash", 0)
    buy_power  = account_data.get("buying_power", 0)
    pv         = account_data.get("portfolio_value", equity)

    # Condor section
    c_open  = condor_status.get("open_count", 0)
    c_close = condor_status.get("closed_count", 0)
    c_wr    = condor_status.get("win_rate", 0)
    c_pnl   = condor_status.get("total_pnl", 0)

    # PEAD section
    p_open  = pead_status.get("open_count", 0)
    p_close = pead_status.get("closed_count", 0)
    p_wr    = pead_status.get("win_rate", 0)

    # M&A section
    m_open  = ma_status.get("open_count", 0)
    m_close = ma_status.get("closed_count", 0)
    m_wr    = ma_status.get("win_rate", 0)

    vix    = regime_data.get("vix", "N/A")
    regime = regime_data.get("regime", "N/A")
    mom60  = regime_data.get("spy_mom_60d", "N/A")
    dd     = regime_data.get("dd_from_peak", "N/A")

    condor_action = (
        "ACTIVE -- open condor if new month"
        if isinstance(vix, (int, float)) and 15 <= vix <= 35
        else "SKIP -- VIX out of tradeable range"
    )

    body = f"""
================================================================================
WEEKLY PORTFOLIO REPORT
Generated: {now}
================================================================================

ACCOUNT SUMMARY
---------------
Portfolio Value : ${pv:>12,.2f}
Equity          : ${equity:>12,.2f}
Cash            : ${cash:>12,.2f}
Buying Power    : ${buy_power:>12,.2f}

MARKET CONDITIONS
-----------------
Regime          : {regime}
VIX             : {vix}
SPY 60d Mom     : {mom60}%
DD from Peak    : {dd}%

FACTOR ETF SLEEVE
-----------------
B-SC QMOM Scalar: {bsc}x
Effective QMOM  : {qmom}%  (base 18% * scalar)
Last Rebalance  : {etf_status.get('last_rebalance', 'N/A')}

IRON CONDORS (SIGNAL ONLY -- Alpaca paper does not support options)
-------------------------------------------------------------------
Open Condors    : {c_open}
Closed Condors  : {c_close}
Win Rate        : {c_wr}%
Total Signal P&L: ${c_pnl:>10,.2f}  (tracked, not executed)
Action          : {condor_action}

PEAD POSITIONS
--------------
Open Positions  : {p_open}
Closed Trades   : {p_close}
Win Rate        : {p_wr}%
Active Tickers  : {', '.join(pead_status.get('open_positions', [])) or 'None'}

M&A ARBITRAGE
-------------
Open Positions  : {m_open}
Closed Trades   : {m_close}
Win Rate        : {m_wr}%
Active Deals    : {', '.join(ma_status.get('open_deals', [])) or 'None'}

ACTIONS THIS WEEK
-----------------
Total Trades    : {trade_count}
{_format_condor_action(vix, regime)}

RISK STATUS
-----------
{_format_risk_status(equity)}

================================================================================
This is an automated report from the Quant Portfolio System.
Target: $500+/month | System: 4-layer factor + options + PEAD + M&A
================================================================================
"""
    return body.strip()


def _format_condor_action(vix, regime):
    if not isinstance(vix, (int, float)):
        return "Iron Condor: Cannot determine (VIX unavailable)"
    if vix < 15:
        return f"Iron Condor: SKIP -- VIX {vix:.1f} below 15 minimum"
    elif vix > 35:
        return f"Iron Condor: SKIP -- VIX {vix:.1f} above 35 maximum"
    elif regime == "BEAR_CRISIS":
        return f"Iron Condor: SKIP -- BEAR_CRISIS regime"
    else:
        mult = 1.0 if vix < 20 else (0.75 if vix < 25 else 0.25)
        return f"Iron Condor: OPEN at {mult:.0%} size -- VIX {vix:.1f} regime {regime}"


def _format_risk_status(equity: float) -> str:
    if equity < 80_000:
        return "WARNING: Portfolio down >20% from $100k -- circuit breaker ACTIVE"
    elif equity < 85_000:
        return "CAUTION: Portfolio down >15% -- reduce new positions"
    elif equity < 90_000:
        return "REVIEW: Portfolio down >10% -- monitor closely"
    else:
        return "OK: Portfolio within normal operating range"


def send_weekly_report(
    regime_data: dict,
    account_data: dict,
    etf_status: dict,
    condor_status: dict,
    pead_status: dict,
    ma_status: dict,
    trade_count: int = 0,
    attachments: Optional[List[str]] = None,
) -> bool:
    """Send the weekly portfolio report email."""
    body = format_weekly_report(
        regime_data, account_data, etf_status,
        condor_status, pead_status, ma_status, trade_count,
    )
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    subject  = f"Portfolio Weekly Report -- {date_str}"
    return _send_email(subject, body, attachments)


def send_alert(event_type: str, details: str) -> bool:
    """Send a one-line alert email for important events."""
    subject = f"ALERT: {event_type}"
    body = f"""
PORTFOLIO ALERT
===============
Event    : {event_type}
Details  : {details}
Time     : {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}

This is an automated alert from the Quant Portfolio System.
"""
    return _send_email(subject, body.strip())


def send_progress_update(
    wave: int,
    subject_suffix: str,
    body: str,
    attachments: Optional[List[str]] = None,
) -> bool:
    """Send a progress/status update email during system execution."""
    subject = f"Portfolio System Update #{wave} -- {subject_suffix}"
    return _send_email(subject, body, attachments)
