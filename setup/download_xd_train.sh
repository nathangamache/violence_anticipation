#!/usr/bin/env bash
# Start the XD-Violence TRAIN split download (setup/download_xd_train.py) detached from the terminal.
#
# Usage (from anywhere):
#   bash setup/download_xd_train.sh                 # full run, 4 parallel files
#   bash setup/download_xd_train.sh --workers 6     # extra args go to the Python script
#   tail -f data/logs/xd_train_download.log
#   kill -- -"$(cat data/logs/xd_train_download.pid)"   # stop
#
# The Python script is resumable, so rerunning this launcher after a stop or crash continues where it left off.
# A lock file prevents two copies from running at once. Aborts if the data disk has under 60 GB free (the
# Python script then does the exact check: free space minus the bytes all queued and in-flight files still
# owe must stay >= 60 GB, before every file and every 64 MB written).
set -euo pipefail

REPO=.
DATA=data
PY="$REPO/.venv/bin/python"
NAME=xd_train_download
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
echo "==== $(date '+%Y-%m-%d %H:%M:%S') launch: $PY -u setup/download_xd_train.py $*" >> "$LOG"
nohup setsid flock -n "$LOCK" "$PY" -u setup/download_xd_train.py "$@" >> "$LOG" 2>&1 < /dev/null &
echo $! > "$PIDFILE"
echo "started pid $(cat "$PIDFILE") (process group leader); log: $LOG"
