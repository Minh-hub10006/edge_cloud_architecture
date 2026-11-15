#!/usr/bin/env bash
# v4 multi-stream run on one Linux Edge (7/11): N Edge Agents, each stream with its own video and mode, then a table.
#   bash scripts/run_edge_v4.sh HOST "video:MODE[:START_FRAME] ..." [SECONDS=120] [START_AT=HH:MM:SS]
#   START_FRAME (12/11): this camera begins that many frames into its video (and main stream), so many cameras on the
#   same file are not in step, e.g. "vn6:C:0 vn6:C:700 vn6:C:1400".
#   HOST 246 | 252 | 247 (Windows, Git Bash) MODE A = YOLO11s on the Edge, tracked on the Edge
#                                                  N = YOLOv8n 320 on the Edge, tracked on the Edge (v3 option kept)
#                                                  B = RT-DETR split (Edge NPU head + Cloud tail)
#                                                  C = RT-DETR on the Cloud (JPEG)   D = D-FINE-X on the Cloud (JPEG)
#   e.g. bash scripts/run_edge_v4.sh 246 "vn9:B vn6:B" 120          START_AT lets 3 Edges begin together (NTP clocks).
# Needs: ~/test_videos/<video>.mp4 and <video>_main.mp4 (scripts/make_main_video.py), v4 Cloud workers + Kafka on 249.
# Analytics decode on the CPU (OpenCV); V4_DECODER=v4l2h264dec|qtic2vdec switches to hardware (open loop 7/11).
# V4_HW_SIZE=640 (with V4_DECODER) also resizes on the GPU to 640x640 (decision 8/11).
# V4_SLOW_TASKS=check,... turns the slow lane on (edge_agent/slow_lane.py); results in each cam<i>.log "slow_lane".
set -uo pipefail
host="${1:?HOST 246|252}"; spec="${2:?\"video:MODE ...\"}"; secs="${3:-120}"; start_at="${4:-}"
here="$(cd "$(dirname "$0")/.." && pwd)"
conf="$here/scripts/hosts/edge-$host.sh"
[[ -f "$conf" ]] || { echo "no host config $conf" >&2; exit 2; }
# shellcheck disable=SC1090
source "$conf"
case "$host" in
  246) head_default="$EDGE_HOME/edge_b_20261106/head_w8a8_pc.dlc" ;;
  # 12/11: head_w8a8_pc.dlc already holds the QCS6490 HTP cache (quantized with --htp_socs qcs6490 on 5/11;
  # snpe-dlc-graph-prepare finds the record HTP_V68_QCS6490_2MB and skips), so 252 uses the same file as 246.
  252) head_default="$EDGE_HOME/edge_b_20261106/head_w8a8_pc.dlc" ;;
  # 247 (8/11): GPU Edge, head.onnx on ONNX Runtime CUDA (head id rtl2-onnx-fp32: the Cloud workers must accept it,
  # RTDETR_SPLIT_HEAD_ID=rtl2-w8a8pc,rtl2-onnx-fp32); YOLO11s float ONNX; OpenCV decode on the CPU.
  247) head_default="$EDGE_HOME/edge_b_20261106/head.onnx" ;;
  *) echo "host $host not supported" >&2; exit 2 ;;
esac
head="${V4_HEAD:-$head_default}"
cloud="${V4_CLOUD:-192.168.180.249}"
ids='{"rtdetr_split_cloud":"rtdetrv2_r18vd_sp2_120e_coco+layer2_split","rtdetr_cloud":"rtdetrv2_s_candidate_6b5b2ca4fb32","dfine_x_cloud":"dfine_x_candidate_9461641566be"}'
py="$EDGE_PYTHON_DEFAULT"
# 247's embeddable Python ignores PYTHONPATH: the code directory must be a line of python312._pth
pth="$(dirname "$py")/python312._pth"
if [[ -f "$pth" ]] && ! grep -qiF "$(cygpath -w "$here" 2>/dev/null || echo "$here")" "$pth"; then
  echo "add this line to $pth (before 'import site'): $(cygpath -w "$here" 2>/dev/null || echo "$here")" >&2; exit 2
fi
out="$EDGE_HOME/v4_runs_edge/run_${host}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$out"
echo "$spec" > "$out/spec.txt"

# 1. inputs exist before anything starts
for item in $spec; do
  IFS=: read -r v m start <<< "$item"
  [[ -z "${start:-}" || "$start" =~ ^[0-9]+$ ]] || { echo "start frame must be a number in $item" >&2; exit 2; }
  for f in "$EDGE_HOME/test_videos/$v.mp4" "$EDGE_HOME/test_videos/${v}_main.mp4"; do
    [[ -f "$f" ]] || { echo "missing $f" >&2; exit 2; }
  done
  case "$m" in A|C|D) ;; B) [[ -f "$head" ]] || { echo "mode B needs the head DLC $head" >&2; exit 2; } ;;
    N) [[ -f "${YOLOV8N_DLC:-}" ]] || { echo "mode N needs the YOLOv8n model ${YOLOV8N_DLC:-}" >&2; exit 2; } ;;
    *) echo "unknown mode $m in $item (A N B C D)" >&2; exit 2 ;; esac
done

cma() { awk '/CmaFree/ {print int($2/1024)}' /proc/meminfo 2>/dev/null | grep . || echo "n/a"; }  # n/a: no CMA (247)
# 7/11: CMA (shared by the NPU and the video decoder) went from 68 to 9 MB over a few runs and only a reboot freed it.
echo "CmaFree before: $(cma) MB" | tee "$out/cma.txt"

# 2. clean leftovers (only ours: our stream id, our DLCs, our main-stream publishers); SIGINT first so they can close
# their sessions. Run before and after every run: 14/11 on 252 a camera that timed out left its ffmpeg main-stream
# publisher looping and the plate NPU daemon running into the next run (no pgrep on 247: nothing outlives its run).
# MediaMTX stays on purpose (reused by the next run).
PUBLISHER="_main.mp4 -c copy -an"   # media/copy_publisher.py's ffmpeg
clean_leftovers() {
  command -v pgrep >/dev/null || return 0
  local found=0 dlc
  pgrep -f -- "--stream-id v4-run" >/dev/null && found=1
  pgrep -f -- "$PUBLISHER" >/dev/null && found=1
  for dlc in "$head" "$YOLO11S_DLC" "${YOLOV8N_DLC:-none}" "${PLATE_DLC:-none}"; do pgrep -f -- "--dlc $dlc" >/dev/null && found=1; done
  (( found )) || return 0
  echo "leftovers of a run ($1): stopping them"
  pkill -INT -f -- "--stream-id v4-run" 2>/dev/null; sleep 5
  for dlc in "$head" "$YOLO11S_DLC" "${YOLOV8N_DLC:-none}" "${PLATE_DLC:-none}"; do pkill -f -- "--dlc $dlc" 2>/dev/null; done
  pkill -f -- "$PUBLISHER" 2>/dev/null   # plain ffmpeg stream copy, no decoder session
  sleep 2
  echo "CmaFree after cleanup: $(cma) MB" | tee -a "$out/cma.txt"
}
clean_leftovers "before"

# 3. local MediaMTX for the main streams
if ! (exec 3<>/dev/tcp/127.0.0.1/8554) 2>/dev/null; then
  # 15/11 on 247: the redirect must cover the backgrounded "cd && nohup" subshell too - in Git Bash that subshell lives
  # as long as MediaMTX and held the caller's stdout, so "run_edge_v4.sh | tee" (scenario.sh) never ended
  ( (cd "$here" && MTX_WEBRTCADDITIONALHOSTS="$EDGE_IP" exec nohup "$EDGE_MEDIAMTX_BIN" media/mediamtx.yml)       > "$out/mediamtx.log" 2>&1 < /dev/null & )
  sleep 2
fi

# 4. same start on every Edge
if [[ -n "$start_at" ]]; then
  wait_s=$(( $(date -d "$start_at" +%s) - $(date +%s) ))
  (( wait_s > 0 )) && { echo "waiting ${wait_s}s for $start_at"; sleep "$wait_s"; }
fi

# UC1 (8/11): the plates slow task runs the host's plate detector (PLATE_DLC: NPU DLC or ONNX) on vehicle crops
plate_args=()
if [[ ",${V4_SLOW_TASKS:-}," == *,plates,* ]]; then
  [[ -f "${PLATE_DLC:-}" ]] || { echo "plates task: plate detector missing (PLATE_DLC=${PLATE_DLC:-})" >&2; exit 2; }
  plate_args=(--plate-detector "$PLATE_DLC")
  [[ -n "${PLATE_SNPE_GRAPH:-}" ]] && export PLATE_SNPE_GRAPH   # 252: graph-scoped v3 daemon
fi
# 12/11: mode N on 252 needs the graph-scoped v3 daemon too (edge-252.sh: the legacy daemon silently runs YOLOv8n on the
# CPU). It was never exported: 252 ran N on the legacy snpe-daemon, CPU 76 %, 96.6 C, two agents hung.
if [[ -n "${YOLOV8N_SNPE_GRAPH:-}" ]]; then
  export YOLOV8N_SNPE_GRAPH
fi
export PYTHONPATH="$here:$EDGE_HOME/v4_pydeps" QAIRT_ROOT="${QAIRT_ROOT_DEFAULT:-}" QAIRT_ABI="${QAIRT_ABI:-}" SNPE_PERSISTENT=1 \
  SNPE_PERSISTENT_MODE=daemon YOLO11S_SNPE_DAEMON_BIN="${YOLO11S_DAEMON_BIN:-}"  # NPU settings: empty on 247
i=0
for item in $spec; do
  i=$((i+1)); IFS=: read -r v m start <<< "$item"
  case "$m" in
    A) args=(--mode yolo11s_edge --yolo11s-dlc "$YOLO11S_DLC" --edge-tracking) ;;
    N) args=(--mode yolo_edge --yolo-dlc "$YOLOV8N_DLC" --edge-tracking) ;;
    B) args=(--mode rtdetr_split_cloud --rtdetr-head-dlc "$head") ;;
    C) args=(--mode rtdetr_cloud) ;;
    D) args=(--mode dfine_x_cloud) ;;
  esac
  "$py" -B -u -m edge_agent.control_v1 --video "$EDGE_HOME/test_videos/$v.mp4" --main-video "$EDGE_HOME/test_videos/${v}_main.mp4" \
    ${V4_DECODER:+--decoder "$V4_DECODER"} ${V4_HW_SIZE:+--hw-size "$V4_HW_SIZE"} ${V4_SLOW_TASKS:+--slow-tasks "$V4_SLOW_TASKS"} ${V4_SLOW_CONFIG:+--slow-config "$V4_SLOW_CONFIG"} ${V4_PROFILES:+--camera-profile "$V4_PROFILES"} ${plate_args[@]+"${plate_args[@]}"} --loop-source --frames $((secs * 15)) --start-frame "${start:-0}" "${args[@]}" --policy observe --cloud-inflight 2 \
    --target-fps 15 --fps-levels 15 --preview-fps 15 --transport kafka --kafka-bootstrap "$cloud:29092" \
    --rtsp-base rtsp://127.0.0.1:8554 --cloud-frame-source jpeg --node-id "$NODE_ID" --camera-id "cam-$i" --stream-id v4-run \
    --cloud-model-ids "$ids" --capability "$EDGE_CAPABILITY" --metrics-port 0 --run-timeout-s $((secs + 120)) \
    --state-jsonl "$out/cam$i.jsonl" > "$out/cam$i.log" 2>&1 &
  echo "started cam-$i: $v mode $m"
  sleep 3  # each agent starts its own NPU daemon; do not open them all at once
done
wait
sleep 2
clean_leftovers "after"
"$py" -B "$here/scripts/v4_run_summary.py" "$out"
echo "NPU daemons left: $(if command -v pgrep >/dev/null; then pgrep -fc snpe-daemon; else echo n/a; fi) | CmaFree after the run: $(cma) MB ($(head -1 "$out/cma.txt"))"
