#!/usr/bin/env bash
# Start the UBI-Fights download + verify + extract (setup/download_ubi_fights.py) detached from the terminal.
#
# Usage (from anywhere):
#   bash setup/download_ubi_fights.sh               # full run, 8 parallel byte-range connections
#   bash setup/download_ubi_fights.sh --workers 12  # extra args go to the Python script
#   tail -f data/logs/ubi_fights_download.log
#   kill -- -"$(cat data/logs/ubi_fights_download.pid)"   # stop
#
# The Python script is resumable, so rerunning this launcher after a stop or crash continues where it left off.
# A lock file prevents two copies from running at once. Aborts if the data disk has under 60 GB free (the
# Python script then does the exact check: free space minus (remaining archive bytes + 8.3 GB extracted)
# must stay >= 60 GB).
set -euo pipefail

REPO=.
DATA=data
PY="$REPO/.venv/bin/python"
NAME=ubi_fights_download
LOG="$DATA/logs/$NAME.log"
LOCK="$DATA/logs/.$NAME.lock"
PIDFILE="$DATA/logs/$NAME.pid"
MIN_FREE_GB=60

mkdir -p "$DATA/logs"
free_gb=$(df -BG --output=avail "$DATA" | tail -1 | tr -dc '0-9')
if (( free_gb < MIN_FREE_GB )); then
  echo "ABORT: only ${free_gb} GB free on $(df --output=target "$DATA" | tail -1), need >= ${MIN_FREE_GB} GB" >&2
  exit 2
fi
if ! flock -n "$LOCK" true; then
  echo "already running (lock $LOCK held); see $LOG" >&2
  exit 1
fi

cd "$REPO"
echo "==== $(date '+%Y-%m-%d %H:%M:%S') launch: $PY -u setup/download_ubi_fights.py $*" >> "$LOG"
nohup setsid flock -n "$LOCK" "$PY" -u setup/download_ubi_fights.py "$@" >> "$LOG" 2>&1 < /dev/null &
echo $! > "$PIDFILE"
echo "started pid $(cat "$PIDFILE") (process group leader); log: $LOG"
