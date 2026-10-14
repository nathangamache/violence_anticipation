#!/usr/bin/env bash
# Hard-caps this user's CPU footprint on the shared server by pinning all of the user's processes to a
# subset of cores (default 0-11 = 12 of 16 cores, about 75%). Re-applied periodically so new processes are caught.
# Usage: tools/cpu_governor.sh [cores=0-11] [interval_sec=30]
CORES=${1:-0-11}
INTERVAL=${2:-30}
while true; do
  for p in $(pgrep -u "$(id -u)"); do
    taskset -a -cp "$CORES" "$p" >/dev/null 2>&1
  done
  sleep "$INTERVAL"
done
