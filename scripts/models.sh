# Single source of the four-model standard, sourced by live.sh and bench.sh.
# Change values here only; live and benchmark must never drift apart.

CLOUD_HOST=192.168.180.249
RTSP_PORT=8554
WEBRTC_PORT=8889
WIDTH=1280
HEIGHT=720
BITRATE_KBPS=2500
TRACKER_OPTIONS='{"match_thresh":0.9,"track_high_thresh":0.25,"new_track_thresh":0.2}'
VEHICLE_MERGE_IOU=0.6
CLASS_ALLOWLIST=0,1,2,3,5,7  # COCO person, bicycle, car, motorcycle, bus, truck
OPENCV_VERSION=4.13.0  # Edge246 pin; decode/resize pixels differ across OpenCV builds
OPENCV_PIP=4.13.0.92
CLOUD_PYTHON_DEFAULT=/home/stu0/miniconda3/envs/qnn/bin/python
MEDIAMTX_BIN=/home/stu0/edge_video_platform/tools/mediamtx/mediamtx

# Per-Edge paths/ids. EDGE_NODE overrides; otherwise map this host's name. Cloud249 and
# unknown hosts keep the Edge246 values (they are only used by Edge roles).
case "${EDGE_NODE:-$(hostname)}" in
  edge-252|qcs6490-odk) _EDGE_HOST_FILE=edge-252.sh ;;
  edge-247|ws570-249|WS570-249) _EDGE_HOST_FILE=edge-247.sh ;;
  *) _EDGE_HOST_FILE=edge-246.sh ;;
esac
source "$(dirname "${BASH_SOURCE[0]}")/hosts/$_EDGE_HOST_FILE"

# Sets MODE PLACEMENT CLOUD_MODELS CLOUD_MODES MODEL_KEY ARTIFACT ARTIFACT_SHA
# [ARTIFACT_DATA_SHA] CONFIDENCE NMS_IOU IMGSZ LAYOUT BENCH_ADAPTER for one model.
load_model() {
  ARTIFACT_DATA_SHA=""
  NMS_IOU=0.5
  case "$1" in
    yolov8n)
      MODE=yolo_edge; PLACEMENT=edge; CLOUD_MODELS=none; CLOUD_MODES='[]'
      MODEL_KEY=${YOLOV8N_KEY:-yolov8n_320_int8_split_traffic100}; BENCH_ADAPTER=yolo_snpe
      CONFIDENCE=0.25; IMGSZ=320; LAYOUT=nchw
      ARTIFACT=$YOLOV8N_DLC
      ARTIFACT_SHA=$YOLOV8N_DLC_SHA ;;
    yolo11s)
      MODE=yolo11s_edge; PLACEMENT=edge; CLOUD_MODELS=none; CLOUD_MODES='[]'
      MODEL_KEY=${YOLO11S_KEY:-yolo11s320_w8a8_qcs6490}; BENCH_ADAPTER=yolo11s_snpe
      CONFIDENCE=0.1; IMGSZ=320; LAYOUT=${YOLO11S_LAYOUT:-nhwc}
      ARTIFACT=$YOLO11S_DLC
      ARTIFACT_SHA=$YOLO11S_DLC_SHA ;;
    rtdetrv2)
      MODE=rtdetr_cloud; PLACEMENT=cloud; CLOUD_MODELS=rtdetrv2_small; CLOUD_MODES='["rtdetr_cloud"]'
      MODEL_KEY=rtdetrv2_s_candidate_265508cc3dc2; BENCH_ADAPTER=rtdetrv2_onnx
      CONFIDENCE=0.5; IMGSZ=640; LAYOUT=nchw  # DETR has no NMS: 0.10 gave ~119 boxes/frame live
      ARTIFACT=/home/stu0/cloud_artifacts/rtdetrv2_s_candidate_20261005/rtdetrv2_s.onnx
      ARTIFACT_SHA=265508cc3dc296cee5fd21b7ff818196358a234eebdce5e6e0a9949b5998fca9
      ARTIFACT_DATA_SHA=fee52913900260b0af43d97b28fdfce10dfabe0106ffe86a372c242bec7f5202 ;;
    lwdetr)
      MODE=lwdetr_large_cloud; PLACEMENT=cloud; CLOUD_MODELS=lwdetr_large; CLOUD_MODES='["lwdetr_large_cloud"]'
      MODEL_KEY=lwdetr_large_candidate_4f2bb0afdd7f; BENCH_ADAPTER=lwdetr_onnx
      CONFIDENCE=0.5; IMGSZ=640; LAYOUT=nchw  # DETR has no NMS: 0.10 gave ~119 boxes/frame live
      ARTIFACT=/home/stu0/cloud_artifacts/lwdetr_large_candidate_20261006/lwdetr_large_coco.onnx
      ARTIFACT_SHA=4f2bb0afdd7f1a1e7ba95406b02106bcc9aefb75e2d1280293c3cbcbbfa07bff ;;
    dfinex)  # replaces LW-DETR from v50; 0.65 = best mean F1 on the UA-DETRAC train split (MODEL_COMPARE.md)
      MODE=dfine_x_cloud; PLACEMENT=cloud; CLOUD_MODELS=dfine_x; CLOUD_MODES='["dfine_x_cloud"]'
      MODEL_KEY=dfine_x_candidate_9461641566be; BENCH_ADAPTER=dfinex_onnx
      CONFIDENCE=0.65; IMGSZ=640; LAYOUT=nchw
      ARTIFACT=/home/stu0/cloud_artifacts/dfine_x_candidate_20261021/dfine_x_obj2coco.onnx
      ARTIFACT_SHA=9461641566be309b15e8e24dfd6d9283e6564186ab851c3e55d39cfaa66ddfed ;;
    *) return 1 ;;
  esac
  # One-off A/B without repackaging; the value is printed and kept in run manifests.
  CONFIDENCE="${MODEL_CONF:-$CONFIDENCE}"
}

# Environment for running a model's adapter in this process (Edge daemon or Cloud ORT).
export_model_runtime() {
  case "$ARTIFACT" in *.onnx) [ "$PLACEMENT" = edge ] && return 0 ;; esac  # GPU Edge: ORT CUDA, no SNPE
  case "$1" in
    yolov8n)
      [ -x "$SNPE_DAEMON_BIN_DEFAULT" ] || { echo "snpe-daemon missing: $SNPE_DAEMON_BIN_DEFAULT" >&2; return 1; }
      export SNPE_PERSISTENT=1 SNPE_PERSISTENT_MODE=daemon QAIRT_ROOT="$QAIRT_ROOT_DEFAULT" QAIRT_ABI \
             SNPE_DAEMON_BIN="$SNPE_DAEMON_BIN_DEFAULT"
      if [ -n "${YOLOV8N_SNPE_GRAPH:-}" ]; then  # host opts YOLOv8n into the strict v3 daemon (Edge252)
        [ -x "$YOLO11S_DAEMON_BIN" ] || { echo "v3 daemon missing: $YOLO11S_DAEMON_BIN" >&2; return 1; }
        export YOLOV8N_SNPE_GRAPH YOLO11S_SNPE_DAEMON_BIN="$YOLO11S_DAEMON_BIN"
      fi ;;
    yolo11s)
      [ -x "$YOLO11S_DAEMON_BIN" ] || { echo "YOLO11s cache-off daemon missing: $YOLO11S_DAEMON_BIN" >&2; return 1; }
      export SNPE_PERSISTENT=1 SNPE_PERSISTENT_MODE=daemon QAIRT_ROOT="$QAIRT_ROOT_DEFAULT" QAIRT_ABI \
             YOLO11S_SNPE_DAEMON_BIN="$YOLO11S_DAEMON_BIN" ;;
    rtdetrv2) export REQUIRE_CUDA=1 RTDETRV2_ALLOW_IMPORTED_CANDIDATE=1 RTDETRV2_SMALL_ONNX="$ARTIFACT"
              export RTDETRV2_CONF_THR="$CONFIDENCE" ;;
    lwdetr) export REQUIRE_CUDA=1 LWDETR_ALLOW_IMPORTED_CANDIDATE=1 LWDETR_LARGE_ONNX="$ARTIFACT"
            export LWDETR_CONF_THR="$CONFIDENCE" ;;
    dfinex) export REQUIRE_CUDA=1 DFINEX_ALLOW_IMPORTED_CANDIDATE=1 DFINEX_ONNX="$ARTIFACT"
            export DFINEX_CONF_THR="$CONFIDENCE" ;;
  esac
}

# Cloud receiver environment shared by live and benchmark receivers.
export_cloud_receiver() {  # $1=grpc port
  export CLOUD_TRANSPORT=grpc CLOUD_MODELS CLOUD_GRPC_BIND="0.0.0.0:$1"
  export CLOUD_RTSP_BASE="rtsp://127.0.0.1:$RTSP_PORT"
  export CLOUD_MEDIAMTX_WEBRTC_BASE="http://$CLOUD_HOST:$WEBRTC_PORT"
  export CLOUD_BYTETRACK_OPTIONS="$TRACKER_OPTIONS" CLOUD_VEHICLE_MERGE_IOU="$VEHICLE_MERGE_IOU"
  export CLOUD_CLASS_ALLOWLIST="$CLASS_ALLOWLIST"
  export CLOUD_MAX_INFLIGHT="${CLOUD_MAX_INFLIGHT:-2}" REQUIRE_CUDA=1  # GPU slots; 1 = gentlest load
  unset CLOUD_KAFKA_NAMESPACE CLOUD_ENABLE_OVERLAY_SMOKE
}

# Edge only: apply this host's scheduler hint to this shell (inherited by Python, FFmpeg and newly
# spawned SNPE daemons) and to already-running daemons of this model, then record the CPU state.
apply_edge_sched() {  # gate record_dir
  SCHED_PROFILE=os-default
  if [ -n "${EDGE_UCLAMP_MIN:-}" ]; then
    command -v uclampset >/dev/null || fail "$1" "EDGE_UCLAMP_MIN=$EDGE_UCLAMP_MIN but uclampset is missing"
    uclampset -m "$EDGE_UCLAMP_MIN" -p $$ || fail "$1" "uclampset on this shell"
    # Reused daemons predate this run; match ours by DLC path so other projects' daemons are untouched.
    [ "$PLACEMENT" = edge ] && for pid in $(pgrep -f -- "--dlc $ARTIFACT"); do
      uclampset -m "$EDGE_UCLAMP_MIN" -a -p "$pid" >/dev/null || fail "$1" "uclampset on daemon $pid"
    done
    SCHED_PROFILE="uclamp_min=$EDGE_UCLAMP_MIN"
  fi
  export SCHED_PROFILE
  mkdir -p "$2"
  {
    echo "node=$NODE_ID kernel=$(uname -r) sched_profile=$SCHED_PROFILE affinity=$(taskset -cp $$ 2>/dev/null | sed 's/.*: //')"
    if command -v uclampset >/dev/null; then echo "shell_uclamp=$(uclampset -p $$ 2>/dev/null | tr -s '[:space:]' ' ')"; fi
    for c in 0 4 7; do
      d=/sys/devices/system/cpu/cpu$c/cpufreq
      echo "cpu${c} governor=$(cat $d/scaling_governor 2>/dev/null) max=$(cat $d/scaling_max_freq 2>/dev/null)/$(cat $d/cpuinfo_max_freq 2>/dev/null)"
      if [ -e /sys/devices/system/cpu/cpu$c/core_ctl/min_cpus ]; then echo "core_ctl_cpu${c}_min=$(cat /sys/devices/system/cpu/cpu$c/core_ctl/min_cpus)"; fi
    done
    if [ -e /proc/sys/kernel/sched_boost ]; then echo "sched_boost=$(cat /proc/sys/kernel/sched_boost)"; fi
    t=$(cat /sys/class/thermal/thermal_zone*/temp 2>/dev/null | sort -n | tail -1 || true)
    echo "temp_start_c=$(( ${t:-0} / 1000 ))"
    echo "ffmpeg=$(ffmpeg -version 2>/dev/null | head -1)"
  } > "$2/edge_sched.txt"
  gate "$1" "scheduler $SCHED_PROFILE (recorded $2/edge_sched.txt)"
}

gate() { echo "[$1] $2"; }
fail() { echo "[$1] FAIL: $2" >&2; exit 1; }
port_open() { (exec 3<>"/dev/tcp/$1/$2") 2>/dev/null; }
check_sha() {  # path sha gate
  [ -f "$1" ] || fail "${3:-G1}" "missing $1"
  [ "$(sha256sum "$1" | cut -d' ' -f1)" = "$2" ] || fail "${3:-G1}" "SHA256 mismatch $1"
}
check_artifact() {  # gate
  check_sha "$ARTIFACT" "$ARTIFACT_SHA" "$1"
  [ -z "$ARTIFACT_DATA_SHA" ] || check_sha "$ARTIFACT.data" "$ARTIFACT_DATA_SHA" "$1"
}

# Real CUDA kernel execution for the loaded Cloud model; never a provider list only.
cuda_smoke() {  # python gate
  nvidia-smi -L >/dev/null 2>&1 || fail "$2" "nvidia-smi cannot reach the GPU (driver/GPU fault); no CPU fallback"
  CLOUD_MODELS="$CLOUD_MODELS" "$1" -B - <<'PY' || fail "$2" "CUDA adapter smoke"
import os, numpy as np
from inference import phase7_cloud_catalog
catalog = phase7_cloud_catalog(rtdetrv2_onnx=os.getenv("RTDETRV2_SMALL_ONNX"),
    lwdetr_onnx=os.getenv("LWDETR_LARGE_ONNX"), dfinex_onnx=os.getenv("DFINEX_ONNX"),
    enabled=(os.environ["CLOUD_MODELS"],))
(mode,) = catalog.modes()
adapter = catalog.load(mode)
for _ in range(3):
    boxes, ms = adapter.infer(np.zeros((720, 1280, 3), np.uint8))
print(f"CUDA_SMOKE_PASS mode={mode.value} model_id={adapter.model_id} runtime={adapter.runtime} last_ms={ms:.1f}")
PY
}

# MediaMTX: reuse a running instance, otherwise own one for this process (killed on exit).
ensure_mediamtx() {  # gate
  MEDIA_PID=""
  if ! port_open 127.0.0.1 "$RTSP_PORT"; then
    port_open 127.0.0.1 "$WEBRTC_PORT" && fail "$1" "WebRTC $WEBRTC_PORT busy while RTSP $RTSP_PORT absent; inspect MediaMTX"
    [ -x "$MEDIAMTX_BIN" ] || fail "$1" "MediaMTX missing: $MEDIAMTX_BIN"
    MTX_WEBRTCADDITIONALHOSTS="${MEDIA_PUBLIC_HOST:-$CLOUD_HOST}" "$MEDIAMTX_BIN" media/mediamtx.yml >"$ROOT/mediamtx.log" 2>&1 &
    MEDIA_PID=$!
    trap '[ -n "$MEDIA_PID" ] && kill "$MEDIA_PID" 2>/dev/null' EXIT
    for _ in $(seq 40); do
      port_open 127.0.0.1 "$RTSP_PORT" && port_open 127.0.0.1 "$WEBRTC_PORT" && break
      kill -0 "$MEDIA_PID" 2>/dev/null || fail "$1" "MediaMTX exited; see mediamtx.log"
      sleep 0.25
    done
  fi
  port_open 127.0.0.1 "$RTSP_PORT" && port_open 127.0.0.1 "$WEBRTC_PORT" || fail "$1" "MediaMTX not ready"
}
