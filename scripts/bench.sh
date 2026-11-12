#!/usr/bin/env bash
# Standardized maximum-capacity benchmark for the four models (gRPC), gates B0-B4.
#
#   Cloud249: bash scripts/bench.sh cloud <model>           # benchmark receiver :8004 / gRPC :50071
#   native:   bash scripts/bench.sh model <model>           # B1->B2->B3 adapter capacity (Edge models on
#                                                           #   Edge246, Cloud models on Cloud249)
#   Edge246:  bash scripts/bench.sh pipeline <model>        # B1->B2 inflight sweep->B3 to Cloud ByteTrack
#   any:      bash scripts/bench.sh report <dir> [cloud-resources.jsonl]
#   Edge:     bash scripts/bench.sh parity <edge model>       # dump boxes; compare with scripts/edge_parity.py
#   Edges:    BENCH_START=HH:MM:SS BENCH_FPS_EST=<fps> bash scripts/bench.sh contention <model>  # see contention()
#   model:    yolov8n | yolo11s | rtdetrv2 | lwdetr
#
# Both scopes are unpaced: they measure the maximum, not a target FPS.
# Override sizes with BENCH_FRAMES (default 10000), BENCH_REPS (3), BENCH_INFLIGHT ("1 2 4 8");
# BENCH_PREFETCH (default 4, 0 = baseline serial decode) for the pipeline source.
set -euo pipefail

ROLE="${1:-}"
MODEL="${2:-}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source "$ROOT/scripts/models.sh"

HTTP_PORT=8004
GRPC_PORT=50071
SMOKE_FRAMES=60
SWEEP_FRAMES=1000
FRAMES="${BENCH_FRAMES:-10000}"
REPS="${BENCH_REPS:-3}"
INFLIGHT_LIST="${BENCH_INFLIGHT:-1 2 4 8}"
WARMUP=30
CACHE_FRAMES="${BENCH_CACHE_FRAMES:-64}"  # 1080p source: 64 x ~6.2 MB stays under --cache-mib 512
CLIP_SHA=59a29f8913fc91a1ae9a835ffec182967f34204d3dee89c00c98e050f97b4b91
CLOUD_VIDEO="${CLOUD_VIDEO:-/home/stu0/bench_videos/traffic_1min.mp4}"
CAMPAIGN="${BENCH_CAMPAIGN:-$(date +%Y%m%d_%H%M%S)}"
OUT="$ROOT/runs/bench/$CAMPAIGN"
USAGE="Usage: bash scripts/bench.sh cloud|model|pipeline|contention <yolov8n|yolo11s|rtdetrv2|lwdetr> | report <dir> [cloud-resources]"

# Generous finite bound: 0.2 s per frame (>= 5 FPS) plus setup.
timeout_for() { echo $(( $1 * $2 / 5 + 300 )); }

# ---- Cloud benchmark receiver ------------------------------------------------
cloud() {
  local py="${BENCH_PYTHON:-$CLOUD_PYTHON_DEFAULT}"
  "$py" -B scripts/verify_release_snapshot.py || fail B0 "release manifest"
  for port in "$HTTP_PORT" "$GRPC_PORT"; do
    port_open 127.0.0.1 "$port" && fail B0 "port $port busy; stop the previous benchmark receiver first"
  done
  export PYTHONPATH="$ROOT"
  [ "$PLACEMENT" = cloud ] && check_artifact B0
  export_cloud_receiver "$GRPC_PORT"
  if [ "$PLACEMENT" = cloud ]; then export_model_runtime "$MODEL" || fail B0 "model runtime"; fi
  mkdir -p "$OUT"
  export CLOUD_BENCHMARK=1 CLOUD_CONTROL_V1=0 CLOUD_PREVIEW_FPS=30 CLOUD_FRAME_CACHE_TTL_S=30
  export CLOUD_RUN_ID="bench-$MODEL" CLOUD_BENCHMARK_RESOURCES="$OUT/cloud-resources-$MODEL-$(date +%H%M%S).jsonl"
  [ "$PLACEMENT" = cloud ] && cuda_smoke "$py" B0
  ensure_mediamtx B0
  gate B0 "PASS benchmark receiver $CLOUD_RUN_ID; Cloud resources -> $CLOUD_BENCHMARK_RESOURCES"
  "$py" -B -m uvicorn web.cloud_app:create_app --factory --host 0.0.0.0 --port "$HTTP_PORT"
}

# ---- Shared preflight for measuring roles -------------------------------------
preflight() {  # host-local video path, host (edge|cloud) running this role
  PY="${BENCH_PYTHON:-$([ "$2" = edge ] && echo "$EDGE_PYTHON_DEFAULT" || echo "$CLOUD_PYTHON_DEFAULT")}"
  "$PY" -B scripts/verify_release_snapshot.py || fail B0 "release manifest"
  check_sha "$1" "$CLIP_SHA" B0
  export PYTHONPATH="$ROOT"
  export_model_runtime "$MODEL" || fail B0 "model runtime"
  [ "$2" = edge ] && apply_edge_sched B0 "$OUT/$MODEL"
  return 0
}

passed() {  # dir reps -> exit 0 only when every expected repetition passed
  "$PY" -B - "$1" "$2" <<'PY'
import json, sys
from pathlib import Path
summaries = [json.loads(p.read_text()) for p in Path(sys.argv[1]).rglob("summary.json")]
if len(summaries) != int(sys.argv[2]):
    print(f"  expected {sys.argv[2]} repetitions, found {len(summaries)}")
    sys.exit(1)
bad = [s.get("fatal_error") or s.get("failure_reasons") for s in summaries if s.get("execution_status") != "PASS"]
for s in summaries:
    print(f"  rep{s['repetition']}: {s['execution_status']} completed={s['completed']}/{s['measured_frames']} "
          f"fps={s.get('effective_fps') or 0:.2f} bytetrack_fps={s.get('bytetrack_completion_fps') or 0:.2f}")
if not summaries or bad:
    print(f"  failures: {bad or 'no summary written'}")
    sys.exit(1)
PY
}

# ---- Model capacity on the model's native host -------------------------------
model_step() {  # gate name frames reps
  local dir="$OUT/$MODEL/model/$2" video="$VIDEO"
  [ "$PLACEMENT" = cloud ] && video="$CLOUD_VIDEO"
  local args=(--adapter "$BENCH_ADAPTER" --artifact "$ARTIFACT" --video "$video" --node-id "$NODE"
    --model-key "$MODEL_KEY" --frames "$3" --warmup-frames "$WARMUP" --repetitions "$4"
    --cache-frames "$CACHE_FRAMES" --cache-mib 512 --imgsz "$IMGSZ" --layout "$LAYOUT"
    --confidence "$CONFIDENCE" --nms-iou "$NMS_IOU" --out "$dir"
    --run-timeout-s "$(timeout_for "$3" "$4")")
  [ "$MODEL" = yolo11s ] && args+=(--allow-diagnostic)
  gate "$1" "RUN model capacity $MODEL frames=$3 reps=$4 -> $dir"
  "$PY" -B -m benchmark model "${args[@]}" >"$dir.log" 2>&1 || true
  passed "$dir" "$4" || fail "$1" "model capacity; see $dir.log"
  gate "$1" "PASS"
}

model() {
  if [ "$PLACEMENT" = edge ]; then NODE=$NODE_ID; preflight "$VIDEO" edge
  else NODE=cloud-249; preflight "$CLOUD_VIDEO" cloud; cuda_smoke "$PY" B0; fi
  check_artifact B0
  gate B0 "PASS $MODEL model capacity preflight"
  mkdir -p "$OUT/$MODEL/model"
  model_step B1 smoke "$SMOKE_FRAMES" 1
  model_step B2 1k "$SWEEP_FRAMES" 1
  model_step B3 "${FRAMES}x$REPS" "$FRAMES" "$REPS"
  report_dir "$OUT/$MODEL/model"
}

# ---- Contention: several Edges load the same Cloud model at once --------------
# Edges differ ~10x in FPS, so equal frame counts would leave the slow Edge running alone.
# Every Edge instead sizes each repetition to the same wall-clock duration and starts it in
# the same wall-clock slot (start + k * (duration + gap)); hosts must share NTP time.
#   BENCH_START=HH:MM:SS (today, required)  BENCH_FPS_EST=<this Edge's single-Edge FPS> (required)
#   BENCH_DURATION_S (120)  BENCH_GAP_S (120: slack for contention slowdown)  BENCH_SCENARIO (label, e.g. 3edge)
contention() {
  : "${BENCH_START:?set BENCH_START=HH:MM:SS (same on every Edge)}"
  : "${BENCH_FPS_EST:?set BENCH_FPS_EST to the single-Edge FPS of $MODEL on this Edge}"
  local duration="${BENCH_DURATION_S:-120}" gap="${BENCH_GAP_S:-120}" start frames rep slot dir
  start=$(date -d "today $BENCH_START" +%s) || fail B0 "BENCH_START must be HH:MM:SS"
  frames=$(awk -v f="$BENCH_FPS_EST" -v d="$duration" 'BEGIN { printf "%d", f * d }')
  [ "$frames" -ge 100 ] || fail B0 "BENCH_FPS_EST x BENCH_DURATION_S gives only $frames frames"
  OUT="$ROOT/runs/bench/contention_${BENCH_SCENARIO:-multi}_$(date +%Y%m%d)_${BENCH_START//:/}"
  preflight "$VIDEO" edge
  [ "$PLACEMENT" = edge ] && check_artifact B0
  cloud_ready
  mkdir -p "$OUT/$MODEL/pipeline"
  gate B0 "PASS contention $MODEL: $REPS x ${duration}s (~$frames frames) from $BENCH_START, gap ${gap}s"
  for rep in $(seq 1 "$REPS"); do
    slot=$(( start + (rep - 1) * (duration + gap) ))
    # Contention slows Edges, so a repetition can overrun its slot; start late rather than skip.
    if [ "$(date +%s)" -le "$slot" ]; then
      gate "B$rep" "waiting for $(date -d "@$slot" +%T)"
      sleep $(( slot - $(date +%s) ))
    else
      gate "B$rep" "LATE: slot $(date -d "@$slot" +%T) passed, starting now $(date +%T)"
    fi
    # Overload losing a few frames is a contention result, not a reason to skip later slots.
    ( pipeline_step "B$rep" "contention-rep$rep-inflight2" "$frames" 1 2 )       || gate "B$rep" "FAIL recorded; continuing with the next slot"
  done
  report_dir "$OUT/$MODEL/pipeline"
}

# ---- Pipeline capacity Edge -> Cloud ByteTrack -------------------------------
pipeline_step() {  # gate name frames reps inflight
  local dir="$OUT/$MODEL/pipeline/$2"
  local args=(--mode "$MODE" --profile pipeline_capacity --transport grpc --model-key "$MODEL_KEY"
    --video "$VIDEO" --loop-source --source-prefetch "${BENCH_PREFETCH:-4}" --node-id "$NODE_ID" --camera-id traffic-01
    --grpc-target "$CLOUD_HOST:$GRPC_PORT" --rtsp-base "rtsp://$CLOUD_HOST:$RTSP_PORT"
    --frames "$3" --warmup-frames "$WARMUP" --repetitions "$4" --queue-depth 16 --max-inflight "$5"
    --imgsz "$IMGSZ" --layout "$LAYOUT" --confidence "$CONFIDENCE" --nms-iou "$NMS_IOU"
    --phase-timeout-s "$(timeout_for "$3" 1)" --run-timeout-s "$(timeout_for "$3" "$4")" --out "$dir")
  if [ "$PLACEMENT" = edge ]; then
    # Edge results need no input pixels in Cloud; preview media is excluded from capacity.
    args+=(--artifact "$ARTIFACT" --media-mode none --priming-frames 0 --padding-frames 0)
    [ "$MODEL" = yolo11s ] && args+=(--allow-diagnostic)
  else
    # Cloud input pixels travel by RTSP; media-fps is a timestamp base, not pacing.
    args+=(--media-mode rtsp --media-fps 30 --width "$WIDTH" --height "$HEIGHT"
           --bitrate-kbps "$BITRATE_KBPS" --priming-frames 30 --padding-frames 30)
  fi
  gate "$1" "RUN pipeline $MODEL frames=$3 reps=$4 inflight=$5 -> $dir"
  "$PY" -B -m benchmark pipeline "${args[@]}" >"$dir.log" 2>&1 || true
  passed "$dir" "$4" || fail "$1" "pipeline; see $dir.log"
  gate "$1" "PASS"
}

cloud_ready() {  # the matching benchmark receiver must be up on Cloud249
  local health
  health="$(curl -fsS --max-time 5 "http://$CLOUD_HOST:$HTTP_PORT/healthz")" \
    || fail B0 "Cloud benchmark receiver unreachable; run 'bash scripts/bench.sh cloud $MODEL' on Cloud249"
  HEALTH="$health" RUN_ID="bench-$MODEL" MODES="$CLOUD_MODES" "$PY" -B -c '
import json, os, sys
h = json.loads(os.environ["HEALTH"])
ok = (h.get("status") == "ok" and h.get("run_id") == os.environ["RUN_ID"]
      and h.get("cloud_model_modes") == json.loads(os.environ["MODES"]))
sys.exit(0 if ok else f"wrong Cloud receiver: {h}")' || fail B0 "Cloud must run 'bash scripts/bench.sh cloud $MODEL'"
  port_open "$CLOUD_HOST" "$GRPC_PORT" || fail B0 "Cloud gRPC $GRPC_PORT unreachable"
}

pipeline() {
  preflight "$VIDEO" edge  # pipeline always drives from Edge, also for Cloud models
  [ "$PLACEMENT" = edge ] && check_artifact B0
  cloud_ready
  gate B0 "PASS $MODEL pipeline preflight"
  mkdir -p "$OUT/$MODEL/pipeline"
  pipeline_step B1 smoke "$SMOKE_FRAMES" 1 4
  local best="" n
  for n in $INFLIGHT_LIST; do
    pipeline_step B2 "1k-inflight$n" "$SWEEP_FRAMES" 1 "$n"
  done
  "$PY" -B -m benchmark.report "$OUT/$MODEL/pipeline" --expected-reps 1 --out "$OUT/$MODEL/pipeline/sweep" >/dev/null || true
  best="$("$PY" -B - "$OUT/$MODEL/pipeline/sweep/report.json" "$SWEEP_FRAMES" <<'PY'
import json, sys
rows = [r for r in json.load(open(sys.argv[1])) if r["measured_frames"] == int(sys.argv[2])
        and r["profile"].startswith("pipeline_capacity") and r["gate"] != "FAIL" and r["effective_fps_mean"]]
print(max(rows, key=lambda r: r["effective_fps_mean"])["max_inflight"] if rows else "")
PY
)"
  [ -n "$best" ] || fail B2 "no passing inflight in sweep"
  gate B2 "saturating inflight=$best (highest effective FPS at ${SWEEP_FRAMES} frames)"
  pipeline_step B3 "${FRAMES}x$REPS-inflight$best" "$FRAMES" "$REPS" "$best"
  report_dir "$OUT/$MODEL/pipeline"
}

report_dir() {  # dir [cloud-resources]
  local py="${PY:-${BENCH_PYTHON:-python3}}" extra=()
  [ -n "${2:-}" ] && extra=(--cloud-resources "$2")
  set +e
  PYTHONPATH="$ROOT" "$py" -B -m benchmark.report "$1" "${extra[@]}" --expected-reps "$REPS"
  local code=$?
  set -e
  [ "$code" = 0 ] && gate B4 "PASS report $1/report.md" || gate B4 "REVIEW report $1/report.md (sweep/smoke rows are expected PROVISIONAL)"
}

# ---- Cross-Edge parity: dump this host's boxes for one Edge model --------------
parity() {
  [ "$PLACEMENT" = edge ] || fail P1 "parity is for Edge models"
  preflight "$VIDEO" edge
  mkdir -p "$ROOT/runs/parity"
  local out="$ROOT/runs/parity/${MODEL}_${NODE_ID}.json" args=()
  [ "$MODEL" = yolo11s ] && args+=(--allow-diagnostic)
  "$PY" -B scripts/edge_parity.py dump --adapter "$BENCH_ADAPTER" --artifact "$ARTIFACT" --video "$VIDEO"     --out "$out" --imgsz "$IMGSZ" --layout "$LAYOUT" --confidence "$CONFIDENCE" --nms-iou "$NMS_IOU" "${args[@]}"     || fail P1 "dump"
  gate P1 "PASS $out"
}

case "$ROLE" in
  report) [ -n "$MODEL" ] || { echo "$USAGE" >&2; exit 2; }; report_dir "$MODEL" "${3:-}"; exit 0 ;;
  cloud|model|pipeline|parity|contention) load_model "$MODEL" || { echo "$USAGE" >&2; exit 2; }; "$ROLE" ;;
  *) echo "$USAGE" >&2; exit 2 ;;
esac
