#!/usr/bin/env bash
# Adaptive control runs (control v1): each Edge switches among the four benchmarked models at
# runtime; per-window state JSONL is the DRL dataset. Gates C1-C3.
#
#   Cloud249: bash scripts/control.sh cloud                 # RT-DETRv2 + D-FINE-X + telemetry, foreground
#   Edge:     bash scripts/control.sh edge <policy>         # policy: explore | rules | cycle | drl | observe
#             or one fixed model: yolov8n | yolo11s | rtdetr | dfinex (no switching, e.g. Cloud saturation tests)
#
# v50: CONTROL_FRAME_SOURCE (jpeg: Cloud tickets carry a CONTROL_JPEG_QUALITY (85) JPEG and the H.264 preview goes
# to the Edge's own MediaMTX, started by run_test.sh; rtsp: v49 path, video pushed to the Cloud MediaMTX).
# CONTROL_PROBE_PERIOD_S (0 = off; 5 = 2 probe frames per 5 s, docs/PLAN_PROBE_VI.md).
# Cloud scheduler (docs/PLAN_CONTROL_V3_VI.md): Cloud CONTROL_SCHEDULER=fixed:<json>|rules (CONTROL_PERIOD_S, 5);
# Edges then run policy `cloud` (CONTROL_FPS_LEVELS, default 5,10,15) and follow their per-stream commands.
# Edge overrides: CONTROL_DURATION_S (300), CONTROL_SEED (node-derived), CONTROL_HOLD_S (5: minimum
# seconds per action = controller cooldown), CONTROL_FPS (15: analytics FPS, the single FPS level),
# CONTROL_ACTOR (actor JSON, required for drl), CONTROL_MODES (default all four),
# CONTROL_VIDEO + CONTROL_VIDEO_SHA (e.g. a `detrac.py playlist` clip; windows map back to scenes
# with `detrac.py windows state.jsonl clip.json`), CONTROL_START_S (0: staged runs begin this many
# seconds into the video; recorded in <run>/start_frame), CONTROL_MAX_TEMP_C (90: stop the run when the
# hottest thermal zone reaches it; Linux Edges only), CONTROL_CLOUD_INFLIGHT (2: Cloud tickets in flight),
# CONTROL_TICKET_TTL_S (1.5: result deadline after capture, checked on the Cloud clock; raise it by the skew on an
# Edge whose clock runs behind the Cloud and cannot be synced, e.g. 3.5 on 247 at -2 s).
# Cloud overrides: CLOUD_MAX_INFLIGHT (2 GPU slots), CLOUD_MAX_TEMP_C (75) and CLOUD_MAX_POWER_W (220): the GPU
# guard logs <run>/gpu.csv every 2 s and stops the receiver when the GPU is unreachable or stays over a limit 6 s.
# Plate OCR (docs/PLAN_PLATE_OCR_TICH_HOP_VI.md): Cloud CONTROL_PLATE_OCR=1 also starts cloud_core.plate_ocr_service
# on PLATE_PORT (PLATE_OCR_PYTHON = an env with easyocr, PLATE_OCR_HOME, PLATE_OCR_WORKERS 4) -> <run>/plate-ocr.jsonl;
# Edge CONTROL_PLATE=1 runs the plate path on every published frame with the host's PLATE_DLC (CONTROL_PLATE_EVERY 3,
# CONTROL_PLATE_CAP 5, CONTROL_PLATE_CONFIRM 2) -> <run>/plate.jsonl.
set -euo pipefail

ROLE="${1:-}"
POLICY="${2:-explore}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source "$ROOT/scripts/models.sh"
HTTP_PORT=8006
GRPC_PORT=50075
PLATE_PORT=50062
PREVIEW_FPS=15
RUN_ID=control
ALL_MODES=yolo_edge,yolo11s_edge,rtdetr_cloud,dfine_x_cloud

# Artifact model IDs and measured single-request capacities (Cloud249 model capacity, 10k x 3).
load_model rtdetrv2; RTDETR_ID=$MODEL_KEY
load_model dfinex; DFINEX_ID=$MODEL_KEY
CLOUD_IDS="{\"rtdetr_cloud\":\"$RTDETR_ID\",\"dfine_x_cloud\":\"$DFINEX_ID\"}"
# shortcut: D-FINE-X 85 = LW-DETR's adapter capacity (D-FINE-X model-only b1 is 100 img/s); re-measure in v50 step 15.
CAPACITIES="{\"$RTDETR_ID\":172.5,\"$DFINEX_ID\":85}"

# Stop this run when the hottest thermal zone reaches CONTROL_MAX_TEMP_C. No thermal zones (Windows 247)
# means no guard. Windows already written stay in state.jsonl.
thermal_guard() {  # run_dir
  local limit="${CONTROL_MAX_TEMP_C:-90}" t
  while sleep 5; do
    t=$(cat ${THERMAL_GLOB:-/sys/class/thermal/thermal_zone*/temp} 2>/dev/null | sort -n | tail -1 || true)
    [ -n "$t" ] && [ $(( t / 1000 )) -ge "$limit" ] || continue
    echo "[C2] THERMAL STOP $(( t / 1000 ))C >= ${limit}C" | tee "$1/thermal_stop"
    # TERM, not INT: run_test.sh starts streams with &, and non-interactive bash makes background jobs ignore
    # SIGINT (252, Oct 28: 15 THERMAL STOPs, every run went on to 97 C).
    pkill -TERM -f -- "--state-jsonl $1/state.jsonl" || true
    return 0
  done
}

gpu_guard() {  # run_dir
  local max_t="${CLOUD_MAX_TEMP_C:-75}" max_p="${CLOUD_MAX_POWER_W:-220}" over=0 line t p reason=""
  echo "timestamp, temperature.gpu, power.draw [W], utilization.gpu [%]" > "$1/gpu.csv"
  while sleep 2; do
    line=$(nvidia-smi --query-gpu=timestamp,temperature.gpu,power.draw,utilization.gpu --format=csv,noheader,nounits 2>&1) || true
    if [[ "$line" != *,*,*,* || "$line" == *rror* ]]; then
      reason="GPU unreachable: $(echo $line)"
    else
      echo "$line" >> "$1/gpu.csv"
      read -r t p < <(awk -F', *' '{print int($2), int($3)}' <<< "$line")
      if [ "$t" -ge "$max_t" ] || [ "$p" -ge "$max_p" ]; then over=$((over + 1)); else over=0; fi
      [ "$over" -ge 3 ] && reason="GPU ${t}C ${p}W over limit ${max_t}C/${max_p}W for 6 s"
    fi
    if [ -n "$reason" ]; then
      echo "[C2] GPU GUARD STOP: $reason" | tee "$1/gpu_stop"
      pkill -INT -f "uvicorn web.cloud_app:create_app" || true
      pkill -TERM -f "cloud_core.plate_ocr_service --model-home" || true
      return 0
    fi
  done
}

cloud() {
  local py="${CONTROL_PYTHON:-$CLOUD_PYTHON_DEFAULT}" model run_dir
  "$py" -B scripts/verify_release_snapshot.py || fail C1 "release manifest"
  for port in "$HTTP_PORT" "$GRPC_PORT" $([ "${CONTROL_PLATE_OCR:-0}" = 1 ] && echo "$PLATE_PORT"); do
    port_open 127.0.0.1 "$port" && fail C1 "port $port busy; stop the previous control receiver first"
  done
  export PYTHONPATH="$ROOT"
  for model in rtdetrv2 dfinex; do
    load_model "$model"; check_artifact C1; export_model_runtime "$model" || fail C1 "$model runtime"
    cuda_smoke "$py" C1
  done
  run_dir="$ROOT/runs/control/$(date +%Y%m%d_%H%M%S)_cloud"
  mkdir -p "$run_dir"
  export_cloud_receiver "$GRPC_PORT"
  export CLOUD_MODELS=rtdetrv2_small,dfine_x CLOUD_CONTROL_V1=1 CLOUD_BENCHMARK=0
  export CLOUD_MODEL_CAPACITY_RPS="$CAPACITIES" CONTROL_V1_TELEMETRY_JSONL="$run_dir/cloud-telemetry.jsonl"
  export CLOUD_PREVIEW_FPS="$PREVIEW_FPS" CLOUD_RUN_ID="$RUN_ID"
  if [ -n "${CONTROL_SCHEDULER:-}" ]; then
    export CLOUD_SCHEDULER="$CONTROL_SCHEDULER" CLOUD_SCHEDULER_PERIOD_S="${CONTROL_PERIOD_S:-5}"
    export CLOUD_SCHEDULER_LOG="$run_dir/decisions.jsonl" CLOUD_RULES_CAPACITY_RPS=85
  fi
  # JPEG streams keep their video on the Edge; RTSP (v49) streams still play from this Cloud's MediaMTX.
  [ "${CONTROL_FRAME_SOURCE:-jpeg}" = jpeg ] && export CLOUD_EDGE_WEBRTC_BASES='{"edge-246":"http://192.168.180.246:8889","edge-252":"http://192.168.180.252:8889","edge-247":"http://192.168.180.247:8889"}'
  ensure_mediamtx C1
  gate C1 "PASS both Cloud models on CUDA; telemetry -> $run_dir/cloud-telemetry.jsonl"
  gate C1 "GPU guard: slots=$CLOUD_MAX_INFLIGHT max ${CLOUD_MAX_TEMP_C:-75}C/${CLOUD_MAX_POWER_W:-220}W -> $run_dir/gpu.csv"
  echo "DASHBOARD_URL=http://$CLOUD_HOST:$HTTP_PORT/live"
  PLATE_PID=""
  trap 'kill ${PLATE_PID:-} 2>/dev/null' EXIT
  if [ "${CONTROL_PLATE_OCR:-0}" = 1 ]; then
    PYTHONPATH="$ROOT" "${PLATE_OCR_PYTHON:-/home/stu0/miniconda3/bin/python}" -B -m cloud_core.plate_ocr_service       --model-home "${PLATE_OCR_HOME:-/home/stu0/plate_ocr_home}" --port "$PLATE_PORT"       --workers "${PLATE_OCR_WORKERS:-4}" --log "$run_dir/plate-ocr.jsonl" > "$run_dir/plate-ocr.log" 2>&1 &
    PLATE_PID=$!
    # The port opens only after every worker has loaded its model.
    for _ in $(seq 90); do
      port_open 127.0.0.1 "$PLATE_PORT" && break
      kill -0 "$PLATE_PID" 2>/dev/null || fail C1 "plate OCR exited: $run_dir/plate-ocr.log"
      sleep 2
    done
    port_open 127.0.0.1 "$PLATE_PORT" || fail C1 "plate OCR not ready after 180 s: $run_dir/plate-ocr.log"
    gate C1 "PASS plate OCR :$PLATE_PORT workers=${PLATE_OCR_WORKERS:-4} -> $run_dir/plate-ocr.jsonl"
  fi
  gpu_guard "$run_dir" &
  local guard=$!
  trap 'kill "${guard:-}" ${PLATE_PID:-} 2>/dev/null' EXIT  # runs after cloud() returns, when the local is gone
  "$py" -B -m uvicorn web.cloud_app:create_app --factory --host 0.0.0.0 --port "$HTTP_PORT"
}

edge() {
  local py="${CONTROL_PYTHON:-$EDGE_PYTHON_DEFAULT}" health run_dir model
  local label="$POLICY"  # run directory and stream id keep the name typed
  case "$POLICY" in
    explore|rules|cycle|drl|observe) ;;
    cloud) CONTROL_FPS_LEVELS="${CONTROL_FPS_LEVELS:-5,10,15}" ;;
    yolov8n) CONTROL_MODES=yolo_edge; POLICY=observe ;;
    yolo11s) CONTROL_MODES=yolo11s_edge; POLICY=observe ;;
    rtdetr) CONTROL_MODES=rtdetr_cloud; POLICY=observe ;;
    dfinex) CONTROL_MODES=dfine_x_cloud; POLICY=observe ;;
    *) fail C1 "unknown policy $POLICY" ;;
  esac
  local duration="${CONTROL_DURATION_S:-300}" hold="${CONTROL_HOLD_S:-5}" fps="${CONTROL_FPS:-15}"
  local modes="${CONTROL_MODES:-$ALL_MODES}" start_s="${CONTROL_START_S:-0}"
  # Distinct default seed per Edge so simultaneous explorers do not move in lockstep.
  local seed="${CONTROL_SEED:-$(printf '%s' "$NODE_ID" | cksum | cut -d' ' -f1)}"
  "$py" -B scripts/verify_release_snapshot.py || fail C1 "release manifest"
  if [ -n "${CONTROL_VIDEO:-}" ]; then
    VIDEO="$CONTROL_VIDEO"
    check_sha "$VIDEO" "${CONTROL_VIDEO_SHA:?set CONTROL_VIDEO_SHA for CONTROL_VIDEO}" C1
  else
    check_sha "$VIDEO" 59a29f8913fc91a1ae9a835ffec182967f34204d3dee89c00c98e050f97b4b91 C1
  fi
  local source_fps
  source_fps="$("$py" -B -c 'import cv2, sys; c = cv2.VideoCapture(sys.argv[1]); print(round(c.get(cv2.CAP_PROP_FPS)) or 30)' "$VIDEO")" \
    || fail C1 "cannot read FPS of $VIDEO"
  command -v ffmpeg >/dev/null || fail C1 "ffmpeg missing"
  # Up to 60 s: a stream started while the Cloud still ran its CUDA smoke failed at once (Oct 22).
  for _ in $(seq 12); do
    health="$(curl -fsS --max-time 5 "http://$CLOUD_HOST:$HTTP_PORT/healthz")" && break
    sleep 5
  done
  [ -n "${health:-}" ] || fail C1 "control receiver unreachable; run 'bash scripts/control.sh cloud' on Cloud249"
  HEALTH="$health" "$py" -B -c '
import json, os, sys
h = json.loads(os.environ["HEALTH"])
ok = h.get("status") == "ok" and h.get("run_id") == "control" and set(h.get("cloud_model_modes") or []) == {"rtdetr_cloud", "dfine_x_cloud"}
sys.exit(0 if ok else f"wrong Cloud receiver: {h}")' || fail C1 "Cloud must run 'bash scripts/control.sh cloud'"
  port_open "$CLOUD_HOST" "$GRPC_PORT" || fail C1 "Cloud gRPC $GRPC_PORT unreachable"
  local frame_source="${CONTROL_FRAME_SOURCE:-jpeg}" rtsp_base="rtsp://$CLOUD_HOST:$RTSP_PORT"
  case "$frame_source" in
    jpeg) rtsp_base="rtsp://127.0.0.1:$RTSP_PORT"
          port_open 127.0.0.1 "$RTSP_PORT" || fail C1 "no MediaMTX on this Edge: start runs with run_test.sh (or CONTROL_FRAME_SOURCE=rtsp)" ;;
    rtsp) ;;
    *) fail C1 "CONTROL_FRAME_SOURCE must be jpeg or rtsp" ;;
  esac
  # Start on the first allowed model, so CONTROL_MODES=dfine_x_cloud pins every second to one Cloud model.
  local args=(--video "$VIDEO" --loop-source --frames $(( duration * source_fps )) --mode "${modes%%,*}"
    --start-frame $(( start_s * source_fps )) --cloud-inflight "${CONTROL_CLOUD_INFLIGHT:-2}" --ticket-ttl "${CONTROL_TICKET_TTL_S:-1.5}"
    --allowed-modes "$modes" --policy "$POLICY" --seed "$seed" --cooldown-s "$hold"
    --target-fps "$fps" --fps-levels "${CONTROL_FPS_LEVELS:-$fps}" --preview-fps "$PREVIEW_FPS"
    --width "$WIDTH" --height "$HEIGHT" --bitrate-kbps "$BITRATE_KBPS"
    --grpc-target "$CLOUD_HOST:$GRPC_PORT" --rtsp-base "$rtsp_base" --http-port "$HTTP_PORT"
    --cloud-frame-source "$frame_source" --cloud-jpeg-quality "${CONTROL_JPEG_QUALITY:-85}"
    --probe-period-s "${CONTROL_PROBE_PERIOD_S:-0}"
    --node-id "$NODE_ID" --camera-id traffic-01 --stream-id "control-$label"
    --cloud-model-ids "$CLOUD_IDS" --capability "${EDGE_CAPABILITY:?missing EDGE_CAPABILITY in host config}"
    --metrics-port 0 --run-timeout-s $(( duration + 180 )))
  [ "$POLICY" = drl ] && { [ -f "${CONTROL_ACTOR:-}" ] || fail C1 "set CONTROL_ACTOR to the actor JSON"; args+=(--actor "$CONTROL_ACTOR"); }
  export PYTHONPATH="$ROOT"
  for model in yolov8n yolo11s; do
    case ",$modes," in *",$( [ $model = yolov8n ] && echo yolo_edge || echo yolo11s_edge ),"*) ;; *) continue ;; esac
    load_model "$model"; check_artifact C1; export_model_runtime "$model" || fail C1 "$model runtime"
    if [ "$model" = yolov8n ]; then
      args+=(--yolo-dlc "$ARTIFACT" --yolo-imgsz "$IMGSZ" --yolo-layout "$LAYOUT" --confidence "$CONFIDENCE")
    else
      args+=(--yolo11s-dlc "$ARTIFACT" --yolo11s-confidence "$CONFIDENCE" --yolo11s-nms-iou "$NMS_IOU")
    fi
  done
  local plate=""
  if [ "${CONTROL_PLATE:-0}" = 1 ]; then
    [ -f "${PLATE_DLC:-}" ] || fail C1 "plate detector missing: PLATE_DLC=${PLATE_DLC:-} (host config)"
    port_open "$CLOUD_HOST" "$PLATE_PORT" || fail C1 "Cloud plate OCR :$PLATE_PORT unreachable; start the Cloud with CONTROL_PLATE_OCR=1"
    # Same NPU daemon environment as YOLOv8n (a no-op on the ONNX Edge); the plate adapter ignores YOLOv8n's graph.
    load_model yolov8n; export_model_runtime yolov8n || fail C1 "plate NPU runtime"
    [ -n "${PLATE_SNPE_GRAPH:-}" ] && export PLATE_SNPE_GRAPH YOLO11S_SNPE_DAEMON_BIN="$YOLO11S_DAEMON_BIN"
    plate="every=${CONTROL_PLATE_EVERY:-3} cap=${CONTROL_PLATE_CAP:-5} confirm=${CONTROL_PLATE_CONFIRM:-2}"
    args+=(--plate-detector "$PLATE_DLC" --plate-target "$CLOUD_HOST:$PLATE_PORT"
      --plate-every "${CONTROL_PLATE_EVERY:-3}" --plate-cap "${CONTROL_PLATE_CAP:-5}"
      --plate-confirm "${CONTROL_PLATE_CONFIRM:-2}")
  fi
  run_dir="$ROOT/runs/control/$(date +%Y%m%d_%H%M%S)_${NODE_ID}_$label"
  mkdir -p "$run_dir"
  apply_edge_sched C1 "$run_dir"
  args+=(--state-jsonl "$run_dir/state.jsonl")
  [ -n "$plate" ] && args+=(--plate-log "$run_dir/plate.jsonl")
  echo $(( start_s * source_fps )) > "$run_dir/start_frame"
  gate C1 "PASS $NODE_ID policy=$label ($POLICY) seed=$seed modes=$modes hold=${hold}s fps=$fps ${duration}s start=${start_s}s max_temp=${CONTROL_MAX_TEMP_C:-90}C cloud_inflight=${CONTROL_CLOUD_INFLIGHT:-2} ttl=${CONTROL_TICKET_TTL_S:-1.5}s frame_source=$frame_source jpeg_q=${CONTROL_JPEG_QUALITY:-85} probe_s=${CONTROL_PROBE_PERIOD_S:-0} plate=${plate:-off}"
  echo "DASHBOARD_URL=http://$CLOUD_HOST:$HTTP_PORT/live"
  gate C2 "RUN -> $run_dir"
  thermal_guard "$run_dir" &
  local guard=$!
  set +e
  # State lines already go to state.jsonl; keep the console log readable.
  "$py" -B -u -m edge_agent.control_v1 "${args[@]}" 2>&1 | grep --line-buffered -v '^\[Control v1 state\]' | tee "$run_dir/edge.log"
  local code=${PIPESTATUS[0]}
  kill "$guard" 2>/dev/null
  set -e
  # C3: the run's own summary; every allowed mode must have produced accepted frames.
  CODE="$code" MODES="$modes" POLICY="$POLICY" RUN_DIR="$run_dir" "$py" -B - <<'PY'
import json, os
from pathlib import Path
run = Path(os.environ["RUN_DIR"])
text = (run / "edge.log").read_text(encoding="utf-8")
start = text.rfind("\n{\n")
summary = json.JSONDecoder().raw_decode(text[start + 1:])[0] if start >= 0 else {}
stats, frames = summary.get("stats", {}), summary.get("frames_by_mode", {})
windows = sum(1 for _ in open(run / "state.jsonl", encoding="utf-8")) if (run / "state.jsonl").is_file() else 0
checks = {"exit_zero": os.environ["CODE"] == "0", "windows_logged": windows > 0,
          "no_thermal_stop": not (run / "thermal_stop").exists(),
          "no_processing_errors": stats.get("processing_errors", 0) == 0,
          # Only exploring policies are required to visit every allowed model.
          "every_mode_used": os.environ["POLICY"] not in ("explore", "cycle")
                             or all(frames.get(m, 0) > 0 for m in os.environ["MODES"].split(","))}
plate = summary.get("plate")
if plate is not None:  # the plate path saw frames and every crop got an OCR reply
    checks["plate_ok"] = plate.get("frames", 0) > 0 and not plate.get("errors") and not plate.get("detect_errors")
verdict = {"status": "PASS" if all(checks.values()) else "REVIEW", "checks": checks, "windows": windows,
           "frames_by_mode": frames, "stats": stats, "error_reasons": summary.get("error_reasons", {}), "plate": plate}
(run / "verdict.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")
print(f"[C3] {verdict['status']} windows={windows} frames_by_mode={frames} -> {run / 'verdict.json'}")
failed = [name for name, ok in checks.items() if not ok]
if failed:
    print("[C3] failed checks: " + ", ".join(failed) + f"; reasons={verdict['error_reasons']}")
raise SystemExit(1 if failed else 0)
PY
}

case "$ROLE" in
  cloud) cloud ;;
  edge) edge ;;
  *) echo "Usage: bash scripts/control.sh cloud | edge [explore|rules|cycle|drl|observe|cloud|yolov8n|yolo11s|rtdetr|dfinex]" >&2; exit 2 ;;
esac
