#!/usr/bin/env bash
# Bulk feature extraction queue: three lanes of run_extract steps, detached and resumable.
# Spec, lanes, monitoring and ETAs: docs/FEATURES.md, section "Bulk extraction run v1".
# Usage: tools/extract_queue.sh start | status | stop
# Smoke test: EXTRACT_LIMIT=1 EXTRACT_FEATURE_ROOT=<dir> EXTRACT_LOG=<dir> tools/extract_queue.sh start
set -u

REPO=.
PY=$REPO/.venv/bin/python
DATA=data
LOG=${EXTRACT_LOG:-$DATA/logs/extract}
STATE=$LOG/queue
MANIFEST=${EXTRACT_MANIFEST:-$REPO/benchmark/extract_manifest_v1.csv}
AUDIO_MANIFEST=$STATE/manifest_audio.csv
FEATURE_ROOT=${EXTRACT_FEATURE_ROOT:-$DATA/features}
CORES=${EXTRACT_CORES:-0-11}
STALL_SEC=${EXTRACT_STALL_SEC:-5400}
RETRY_SLEEP=${EXTRACT_RETRY_SLEEP:-600}
EXTRA=()
[ -n "${EXTRACT_FEATURE_ROOT:-}" ] && EXTRA+=(--feature_root "$EXTRACT_FEATURE_ROOT")
[ -n "${EXTRACT_LIMIT:-}" ] && EXTRA+=(--limit "$EXTRACT_LIMIT")
[ -n "${EXTRACT_RETRY_CRASHED:-}" ] && EXTRA+=(--retry_crashed)

# lane step extractor outputs shards device threads manifest sources planned_x_realtime
STEPS="gpu0 visual visual vjepa2_l,siglip2,siglip2_prompts 2 cuda:0 2 all - 35
gpu0 audio_events audio_events ast_events,ast_emb,clap_prompts 2 cuda:0 2 audio - 120
gpu0 voice voice voice_affect,e2v_emb 2 cuda:0 2 audio - 90
gpu0 speech_tox speech_tox speech_tox 2 cuda:0 2 audio xd,yt,ucf 12
gpu1 pose pose pose_raw,pose_interact 3 cuda:1 2 all - 40
cpu motion motion motion 4 cpu 1 all - 100
cpu audio_dsp audio_dsp audio_dsp 4 cpu 1 audio - 50"

note() {
  echo "$(date '+%F %T') $1: $2" >> "$STATE/status.log"
  echo "$(date '+%F %T') $2" > "$STATE/$1.status"
}

run_shard() {
  local lane=$1 name=$2 ext=$3 i=$4 n=$5 dev=$6 thr=$7 man=$8 src=$9 attempt pid rc=0
  local log=$LOG/${lane}_${name}_shard$i.log args=()
  [ "$src" != - ] && args=(--sources "$src")
  for attempt in 1 2 3; do
    echo "=== $(date '+%F %T') attempt $attempt" >> "$log"
    OMP_NUM_THREADS=$thr MKL_NUM_THREADS=$thr OPENBLAS_NUM_THREADS=$thr NUMEXPR_NUM_THREADS=$thr \
      NUMBA_NUM_THREADS=$thr TOKENIZERS_PARALLELISM=false \
      taskset -c "$CORES" "$PY" -m anticipation.features.run_extract --extractor "$ext" \
      --manifest "$man" --keep_order --shard "$i" --num_shards "$n" --device "$dev" --cpu_threads "$thr" \
      "${args[@]}" "${EXTRA[@]}" >> "$log" 2>&1 < /dev/null &
    pid=$!
    while kill -0 "$pid" 2>/dev/null; do
      sleep 30
      if (( $(date +%s) - $(stat -c %Y "$log") > STALL_SEC )); then
        echo "=== $(date '+%F %T') no output for $STALL_SEC s, killing $pid" >> "$log"
        kill -KILL "$pid" 2>/dev/null
      fi
    done
    wait "$pid"; rc=$?
    echo "=== $(date '+%F %T') exit $rc" >> "$log"
    (( rc <= 1 )) && return "$rc"
    (( rc == 3 && attempt < 3 )) && sleep "$RETRY_SLEEP"
  done
  return "$rc"
}

run_step() {
  local lane=$1 name=$2 ext=$3 n=$5 dev=$6 thr=$7 which=$8 src=$9 man=$MANIFEST t0 i p rc=0 pids=()
  [ "$which" = audio ] && man=$AUDIO_MANIFEST
  t0=$(date +%s)
  echo "$t0 $name" > "$STATE/$lane.step"
  note "$lane" "running $name ($n shards, $dev)"
  for ((i = 0; i < n; i++)); do
    run_shard "$lane" "$name" "$ext" "$i" "$n" "$dev" "$thr" "$man" "$src" &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p" || rc=1; done
  note "$lane" "finished $name rc=$rc after $(( ($(date +%s) - t0) / 60 )) min"
}

run_lane() {
  local lane=$1 l name ext outs n dev thr which src x
  while read -r l name ext outs n dev thr which src x; do
    [ "$l" = "$lane" ] && run_step "$l" "$name" "$ext" "$outs" "$n" "$dev" "$thr" "$which" "$src"
  done <<< "$STEPS"
  note "$lane" "lane done"
}

queue_main() {
  ps -o pgid= $$ | tr -d ' ' > "$STATE/queue.pgid"
  "$PY" -c "import csv, sys
r = list(csv.DictReader(open(sys.argv[1])))
w = csv.DictWriter(open(sys.argv[2], 'w', newline=''), fieldnames=list(r[0]))
w.writeheader()
w.writerows(x for x in r if x['has_audio'] == 'True')" "$MANIFEST" "$AUDIO_MANIFEST" || exit 1
  note queue "started (manifest $MANIFEST, feature root $FEATURE_ROOT)"
  run_lane gpu0 &
  { run_lane gpu1; run_lane cpu; } &
  wait
  note queue "all lanes done"
}

running() {
  [ -f "$STATE/queue.pgid" ] && pgrep -g "$(cat "$STATE/queue.pgid")" > /dev/null
}

progress() {
  STEPS="$STEPS" "$PY" - "$MANIFEST" "$AUDIO_MANIFEST" "$FEATURE_ROOT" "$STATE" "$LOG" "${EXTRACT_FEATURE_ROOT:+1}" <<'EOF'
import csv, glob, json, os, sys, time
man, audio_man, root, state, log, scratch = sys.argv[1:7]
rows = {k: list(csv.DictReader(open(p))) for k, p in (("all", man), ("audio", audio_man)) if os.path.exists(p)}
steps = [ln.split() for ln in os.environ["STEPS"].splitlines()]
cur = {}
for lane in ("gpu0", "gpu1", "cpu"):
    p = os.path.join(state, lane + ".step")
    if os.path.exists(p):
        t0, name = open(p).read().split()
        cur[lane] = (name, float(t0))
lane_eta = {}
print(f"{'step':13s} {'videos done':>17s} {'hours done':>17s} {'fail':>5s} {'x realtime':>11s} {'ETA h':>7s}")
for lane, name, ext, outs, n, dev, thr, which, src, planned in steps:
    sel = rows.get(which, [])
    if src != "-":
        sel = [r for r in sel if r["source"] in src.split(",")]
    done = [r for r in sel if all(os.path.exists(os.path.join(root, o, r["video_id"] + ".npz")) for o in outs.split(","))]
    h_all = sum(float(r["duration_sec"]) for r in sel) / 3600
    h_done = sum(float(r["duration_sec"]) for r in done) / 3600
    logdir = os.path.join(root, "_logs", ext) if scratch else os.path.join(log, ext)
    fails, vsec, rate, tag = set(), 0.0, float(planned), "plan"
    for f in glob.glob(os.path.join(logdir, f"shard*of{n}.jsonl")):
        for ln in open(f):
            r = json.loads(ln)
            if r.get("status") in ("fail", "crash"):
                fails.add(r["video_id"])
            elif r.get("status") == "ok" and cur.get(lane, ("", 0))[0] == name and \
                    time.mktime(time.strptime(r["time"], "%Y-%m-%dT%H:%M:%S")) >= cur[lane][1]:
                vsec += float(r.get("duration_sec") or 0)
    fails -= {r["video_id"] for r in done}
    if cur.get(lane, ("", 0))[0] == name and vsec > 0:
        rate, tag = vsec / max(time.time() - cur[lane][1], 1.0), "meas"
    eta = 0.0 if len(done) == len(sel) else (h_all - h_done) / rate
    lane_eta[lane] = lane_eta.get(lane, 0.0) + eta
    print(f"{name:13s} {len(done):7d} / {len(sel):<7d} {h_done:7.1f} / {h_all:<7.1f} {len(fails):5d} "
          f"{rate:6.1f} {tag} {eta:7.1f}")
print("lane ETA h: " + ", ".join(f"{k} {v:.1f}" for k, v in lane_eta.items())
      + f" (cpu starts after gpu1: {lane_eta.get('gpu1', 0) + lane_eta.get('cpu', 0):.1f} h from now)")
EOF
}

case "${1:-}" in
  start)
    if running; then echo "queue already running (pgid $(cat "$STATE/queue.pgid"))"; exit 1; fi
    [ -f "$MANIFEST" ] || { echo "missing manifest $MANIFEST"; exit 1; }
    mkdir -p "$STATE"
    rm -f "$STATE/queue.pgid" "$STATE"/*.step
    cd "$REPO" || exit 1
    EXTRACT_LOG=$LOG setsid nohup nice -n 10 taskset -c "$CORES" bash "$0" _run >> "$STATE/queue.log" 2>&1 < /dev/null &
    sleep 2
    echo "started, pgid $(cat "$STATE/queue.pgid"); logs in $LOG, status: $0 status"
    ;;
  _run)
    queue_main
    ;;
  status)
    if running; then echo "queue running (pgid $(cat "$STATE/queue.pgid"))"; else echo "queue not running"; fi
    for l in queue gpu0 gpu1 cpu; do [ -f "$STATE/$l.status" ] && echo "$l: $(cat "$STATE/$l.status")"; done
    progress
    ;;
  stop)
    running || { echo "queue not running"; exit 0; }
    pg=$(cat "$STATE/queue.pgid")
    kill -TERM -- "-$pg"
    for _ in $(seq 60); do pgrep -g "$pg" > /dev/null || break; sleep 1; done
    kill -KILL -- "-$pg" 2> /dev/null
    note queue "stopped"
    echo "stopped"
    ;;
  *)
    sed -n '2,5p' "$0"; exit 1
    ;;
esac
