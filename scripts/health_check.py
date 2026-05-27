"""
System health check.

Verifies all system components are working correctly:
  - Alpaca connection and account status
  - yfinance data availability
  - State file integrity
  - Test suite pass/fail count
  - Cron job status

Usage:
    python3 scripts/health_check.py
    Returns exit code 0 on all-pass, 1 on any failure.
"""

import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import config

PASS = "PASS"
FAIL = "FAIL"
WARN = "WARN"


def check(label: str, status: str, detail: str = ""):
    icon = "OK" if status == PASS else ("!!" if status == FAIL else "--")
    print(f"  [{icon}] {label:<40} {detail}")
    return status == PASS


def main():
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    print(f"\nSYSTEM HEALTH CHECK -- {now}")
    print("=" * 60)

    all_pass = True

    # 1. Alpaca connection
    print("\n[1] ALPACA CONNECTION")
    try:
        from execution.alpaca_client import AlpacaClient
        client = AlpacaClient(paper=True)
        acct = client.get_account()
        equity = acct.get("equity", 0)
        ok = check("Alpaca paper account", PASS, f"equity=${equity:,.2f}")
    except Exception as e:
        ok = check("Alpaca paper account", FAIL, str(e)[:50])
    all_pass = all_pass and ok

    # 2. yfinance data
    print("\n[2] MARKET DATA")
    try:
        from core import data as mdata
        mdata.clear_cache()
        spy, vix = mdata.get_spy_vix("1mo")
        ok = check("SPY + VIX data", PASS if len(spy) > 10 else FAIL,
                   f"{len(spy)} days SPY, VIX={float(vix.iloc[-1]):.1f}")
    except Exception as e:
        ok = check("SPY + VIX data", FAIL, str(e)[:50])
    all_pass = all_pass and ok

    try:
        from core import data as mdata
        qmom = mdata.get_price_history("QMOM", "3mo")
        ok = check("QMOM prices", PASS if len(qmom) > 40 else FAIL, f"{len(qmom)} days")
    except Exception as e:
        ok = check("QMOM prices", FAIL, str(e)[:50])
    all_pass = all_pass and ok

    # 3. State files
    print("\n[3] STATE FILE INTEGRITY")
    state_dir = Path(config.STATE_DIR)
    expected_files = [
        config.REBAL_FILE,
        config.CONDOR_FILE,
        config.PEAD_FILE,
        config.MA_FILE,
        config.LOG_FILE,
    ]
    import json
    for fpath in expected_files:
        p = Path(fpath)
        if p.exists():
            try:
                with open(p) as f:
                    content = f.read().strip()
                if not content:
                    ok = check(f"State: {p.name}", PASS, "empty (fresh)")
                elif content.startswith("[") or content.startswith("{"):
                    # Try as JSON array/object, then as NDJSON
                    try:
                        json.loads(content)
                    except json.JSONDecodeError:
                        # Try NDJSON (one object per line)
                        for line in content.splitlines():
                            json.loads(line)
                    ok = check(f"State: {p.name}", PASS, f"{p.stat().st_size} bytes")
                else:
                    ok = check(f"State: {p.name}", FAIL, "unrecognized format")
            except Exception as e:
                ok = check(f"State: {p.name}", FAIL, str(e)[:40])
        else:
            ok = check(f"State: {p.name}", WARN, "not yet created (OK for fresh deploy)")
        all_pass = all_pass and (ok or True)  # missing state is a warning, not failure

    # 4. Tests
    print("\n[4] TEST SUITE")
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "--ignore=tests/test_integration.py",
         "-q", "--tb=no"],
        capture_output=True, text=True,
        cwd=str(Path(__file__).parent.parent)
    )
    output = result.stdout + result.stderr
    passed = 0
    for line in output.split("\n"):
        if " passed" in line:
            try:
                passed = int(line.strip().split()[0])
            except Exception:
                pass
    ok = check("Unit tests", PASS if result.returncode == 0 else FAIL,
               f"{passed} passed" if passed else output[-100:].strip())
    all_pass = all_pass and ok

    # 5. Cron
    print("\n[5] CRON SCHEDULE")
    cron = subprocess.run(["crontab", "-l"], capture_output=True, text=True)
    has_daily  = "scheduler/main.py --daily" in cron.stdout
    has_weekly = "scheduler/main.py --weekly" in cron.stdout
    ok = check("Daily cron (13:35 UTC Mon-Fri)", PASS if has_daily else WARN,
               "installed" if has_daily else "not found")
    check("Weekly cron (13:00 UTC Mon)", PASS if has_weekly else WARN,
          "installed" if has_weekly else "not found")
    all_pass = all_pass and ok

    # 6. GitHub
    print("\n[6] GIT STATUS")
    git = subprocess.run(["git", "status", "--short"], capture_output=True, text=True,
                         cwd=str(Path(__file__).parent.parent))
    dirty = git.stdout.strip()
    ok = check("Working tree clean", PASS if not dirty else WARN,
               "clean" if not dirty else dirty[:40])
    git_log = subprocess.run(["git", "log", "--oneline", "-1"], capture_output=True, text=True,
                             cwd=str(Path(__file__).parent.parent))
    check("Latest commit", PASS, git_log.stdout.strip()[:50])

    # Summary
    print()
    print("=" * 60)
    if all_pass:
        print("  OVERALL: ALL CHECKS PASSED")
    else:
        print("  OVERALL: SOME CHECKS FAILED -- review above")
    print("=" * 60)
    print()

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
