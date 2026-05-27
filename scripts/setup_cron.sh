#!/bin/bash
# Setup cron jobs for the trading system
# Schedules:
#   Daily:   9:35 AM ET = 13:35 UTC (Mon-Fri)
#   Weekly:  9:00 AM ET = 13:00 UTC (Monday only)
# Log rotation: each run appends to state/cron.log

TRADER_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="$(which python3)"
CRON_LOG="$TRADER_DIR/state/cron.log"

echo "Setting up cron for trader at: $TRADER_DIR"
echo "Python: $PYTHON"

# Remove existing trader cron entries
crontab -l 2>/dev/null | grep -v "trader/scheduler" > /tmp/current_cron.txt || true

# Add new entries
cat >> /tmp/current_cron.txt << EOF

# Quant Portfolio System
# Daily: Mon-Fri 9:35 AM ET (13:35 UTC)
35 13 * * 1-5 cd $TRADER_DIR && $PYTHON scheduler/main.py --daily >> $CRON_LOG 2>&1

# Weekly: Monday 9:00 AM ET (13:00 UTC) -- ETF rebalance + report
0 13 * * 1 cd $TRADER_DIR && $PYTHON scheduler/main.py --weekly >> $CRON_LOG 2>&1
EOF

crontab /tmp/current_cron.txt
rm /tmp/current_cron.txt

echo "Cron jobs installed:"
crontab -l | grep -A2 "Quant Portfolio"
