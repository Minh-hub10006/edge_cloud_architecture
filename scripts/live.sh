#!/usr/bin/env bash
# Standardized live run for the four planned models, with gates G1-G4.
#
#   Cloud249: scripts/live.sh cloud <model>    # receiver + MediaMTX, foreground
#   Edge246:  scripts/live.sh edge  <model>    # one 1800-frame trial + verdict
#   model:    yolov8n | yolo11s | rtdetrv2 | lwdetr
#
# Run from the root of a verified release lane. Every model uses the same clip,
# media profile, ports, tracker and output layout so runs are comparable.
set -euo pipefail

ROLE="${1:-}"
MODEL="${2:-}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

source "$ROOT/scripts/models.sh"
HTTP_PORT=8005
GRPC_PORT=50073
FRAMES=1800
PREVIEW_FPS=15
load_model "$MODEL" || { echo "Usage: scripts/live.sh cloud|edge yolov8n|yolo11s|rtdetrv2|lwdetr" >&2; exit 2; }
RUN_ID="live-$MODEL"

cloud() {
  local py="${LIVE_PYTHON:-$CLOUD_PYTHON_DEFAULT}"
  # G1: source, ports, dependencies, artifacts
  "$py" -B scripts/verify_release_snapshot.py || fail G1 "release manifest"
  for port in "$HTTP_PORT" "$GRPC_PORT"; do
    port_open 127.0.0.1 "$port" && fail G1 "port $port busy; stop the previous receiver first"
  done
  export PYTHONPATH="$ROOT"
  "$py" -B -c 'import av,grpc,starlette,uvicorn; from cloud_core.tracking import CloudByteTrack; CloudByteTrack().prepare()' \
    || fail G1 "receiver dependencies"
  [ "$PLACEMENT" = cloud ] && check_artifact G1
  gate G1 "PASS source/ports/dependencies/artifact ($MODEL conf=$CONFIDENCE)"

  export_cloud_receiver "$GRPC_PORT"
  export CLOUD_CONTROL_V1=0 CLOUD_BENCHMARK=0 CLOUD_PREVIEW_FPS="$PREVIEW_FPS" CLOUD_RUN_ID="$RUN_ID"
  if [ "$PLACEMENT" = cloud ]; then export_model_runtime "$MODEL" || fail G1 "model runtime"; fi

  # G2: a real CUDA kernel run, not just a provider list
  if [ "$PLACEMENT" = cloud ]; then
    cuda_smoke "$py" G2
    gate G2 "PASS CUDA kernel executed"
  else
    gate G2 "SKIP Edge model; receiver loads no GPU model"
  fi
  ensure_mediamtx G1
  gate RUN "receiver $RUN_ID on :$HTTP_PORT / gRPC :$GRPC_PORT (Ctrl+C to stop)"
  "$py" -B -m uvicorn web.cloud_app:create_app --factory --host 0.0.0.0 --port "$HTTP_PORT"
}

edge() {
  local py="${LIVE_PYTHON:-$EDGE_PYTHON_DEFAULT}"
  # G1: source, clip, artifact, matching Cloud receiver
  "$py" -B scripts/verify_release_snapshot.py || fail G1 "release manifest"
  [ -f "$VIDEO" ] || fail G1 "missing video $VIDEO"
  command -v ffmpeg >/dev/null || fail G1 "ffmpeg missing"
  [ "$PLACEMENT" = edge ] && check_artifact G1
  local health
  health="$(curl -fsS --max-time 5 "http://$CLOUD_HOST:$HTTP_PORT/healthz")" \
    || fail G1 "Cloud receiver unreachable; start 'scripts/live.sh cloud $MODEL' first"
  HEALTH="$health" RUN_ID="$RUN_ID" MODES="$CLOUD_MODES" "$py" -B -c '
import json, os, sys
h = json.loads(os.environ["HEALTH"])
ok = (h.get("status") == "ok" and h.get("transport") == "grpc" and h.get("run_id") == os.environ["RUN_ID"]
      and h.get("cloud_model_modes") == json.loads(os.environ["MODES"]))
sys.exit(0 if ok else f"wrong Cloud receiver: {h}")' || fail G1 "Cloud must run 'scripts/live.sh cloud $MODEL'"
  for port in "$GRPC_PORT" "$RTSP_PORT"; do
    port_open "$CLOUD_HOST" "$port" || fail G1 "Cloud port $port unreachable"
  done
  gate G1 "PASS source/clip/artifact/receiver ($MODEL)"

  local epoch run_dir stream=live-$MODEL
  epoch="$(date +%s%N)"
  run_dir="$ROOT/runs/$MODEL/$(date +%Y%m%d_%H%M%S)"
  mkdir -p "$run_dir"
  local url="http://$CLOUD_HOST:$HTTP_PORT/v1/streams/$NODE_ID/traffic-01/$stream/$epoch/live?fps=$PREVIEW_FPS&buffer_ms=500"
  echo "RAW_URL=$url&smooth_ms=0&review=1"
  echo "SMOOTH_URL=$url&smooth_ms=120"
  echo "DASHBOARD_URL=http://$CLOUD_HOST:$HTTP_PORT/live  (one tile per Edge; stays open across runs)"
  echo "RUN_DIR=$run_dir"
  [ "${LIVE_NO_WAIT:-0}" = 1 ] || read -rp "Open both URLs, then press Enter to publish: "

  export PYTHONPATH="$ROOT"
  [ "$PLACEMENT" = edge ] && { export_model_runtime "$MODEL" || fail G1 "model runtime"; }
  apply_edge_sched G1 "$run_dir"
  local args=(--mode "$MODE" --video "$VIDEO" --frames "$FRAMES" --transport grpc
    --grpc-target "$CLOUD_HOST:$GRPC_PORT" --rtsp-base "rtsp://$CLOUD_HOST:$RTSP_PORT"
    --node-id "$NODE_ID" --camera-id traffic-01 --stream-id "$stream" --stream-epoch "$epoch"
    --decision-id "$RUN_ID" --width "$WIDTH" --height "$HEIGHT" --preview-fps "$PREVIEW_FPS"
    --bitrate-kbps "$BITRATE_KBPS" --shutdown-padding-frames "$PREVIEW_FPS"
    --monitor --metrics-port 0 --monitor-jsonl "$run_dir/monitor.jsonl")
  case "$MODEL" in
    yolov8n) args+=(--yolo-dlc "$ARTIFACT" --yolo-imgsz "$IMGSZ" --yolo-layout "$LAYOUT" --confidence "$CONFIDENCE") ;;
    yolo11s) args+=(--yolo11s-dlc "$ARTIFACT" --yolo11s-allow-diagnostic --yolo11s-confidence "$CONFIDENCE" --yolo11s-nms-iou "$NMS_IOU") ;;
    *) args+=(--cloud-model-id "$MODEL_KEY") ;;
  esac

  # G3: live trial
  gate G3 "RUN $MODEL $FRAMES source frames at preview ${PREVIEW_FPS}FPS"
  set +e
  "$py" -B -u -m edge_agent.phase7 "${args[@]}" 2>&1 | tee "$run_dir/edge.log"
  local code=${PIPESTATUS[0]}
  set -e

  # G4: verdict from the runner's own final counters
  CODE="$code" MODEL="$MODEL" FRAMES="$FRAMES" RUN_DIR="$run_dir" "$py" -B - <<'PY'
import json, os, re
from pathlib import Path
run = Path(os.environ["RUN_DIR"])
text = (run / "edge.log").read_text(encoding="utf-8")
# Known startup race: the first Cloud tickets can expire while the Cloud RTSP tap
# connects. Only deadline_exceeded on the first 2 s of a 30 FPS source is exempt.
STARTUP_SOURCE_FRAMES = 60
rejected = re.findall(r"\[Edge inference\] frame=(\d+) request=\S+ Cloud inference (\w+)", text)
startup = sum(1 for frame, status in rejected
              if status == "deadline_exceeded" and int(frame) < STARTUP_SOURCE_FRAMES)
start = text.rfind("\n{\n")
summary = json.JSONDecoder().raw_decode(text[start + 1:])[0] if start >= 0 else {}
stats = summary.get("inference", {})
checks = {
    "exit_zero": os.environ["CODE"] == "0",
    "all_frames_captured": summary.get("captured") == int(os.environ["FRAMES"]),
    "inferred": stats.get("frames_inferred", 0) > 0,
    "no_model_errors": stats.get("model_errors") == 0,
    "no_transport_errors": stats.get("transport_errors") == 0,
    "no_cloud_rejections_after_startup": stats.get("cloud_rejections") == startup,
    "media_refs_sent": summary.get("media_ref_send_failed") == 0,
}
verdict = {"model": os.environ["MODEL"], "status": "PASS" if all(checks.values()) else "REVIEW",
           "checks": checks, "startup_rejections": startup, "summary": summary,
           "note": "Runtime/transport gate only; visual review and quality are separate."}
(run / "verdict.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")
print(f"[G4] {verdict['status']} {run / 'verdict.json'}")
failed = [name for name, ok in checks.items() if not ok]
if failed:
    print("[G4] failed checks: " + ", ".join(failed))
raise SystemExit(1 if failed else 0)
PY
}

case "$ROLE" in
  cloud) cloud ;;
  edge) edge ;;
  *) echo "Usage: scripts/live.sh cloud|edge yolov8n|yolo11s|rtdetrv2|lwdetr" >&2; exit 2 ;;
esac
