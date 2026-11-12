#!/usr/bin/env bash
# Unpaced pipeline capacity; every result completes at Cloud ByteTrack.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
mode="${1:-yolo_edge}"
transport="${2:-grpc}"
frames="${3:-900}"
repetitions="${4:-3}"
inflight="${5:-4}"
video="${BENCHMARK_VIDEO:-/home/ubuntu/edge_multimodel/videos/clips_1min/traffic_1min.mp4}"
if [[ "$frames" -gt 1700 && -z "${BENCHMARK_VIDEO:-}" ]]; then
  video=/home/ubuntu/edge_multimodel/videos/clips_1min/traffic_benchmark_loop.mp4
fi
[[ -f "$video" ]] || { echo "Video not found: $video; set BENCHMARK_VIDEO" >&2; exit 2; }
case "$mode" in
  yolo_edge)
    export SNPE_PERSISTENT=1 SNPE_PERSISTENT_MODE=daemon
    model_args=(--mode=yolo_edge --model-key=yolov8n_320_int8_split_traffic100
      --artifact=/home/ubuntu/edge_artifacts/npu243/yolov8n_320_int8_split_traffic100.dlc
      --media-mode=none --priming-frames=0 --padding-frames=0)
    ;;
  yolo11s_edge)
    export SNPE_PERSISTENT=1 SNPE_PERSISTENT_MODE=daemon
    model_args=(--mode=yolo11s_edge --model-key=yolo11s320_w8a8_qcs6490
      --artifact="${YOLO11S_DLC:?Set pinned split-v8 YOLO11S_DLC}"
      --imgsz=320 --layout=nhwc --confidence=0.10 --nms-iou=0.5 --allow-diagnostic
      --media-mode=none --priming-frames=0 --padding-frames=0)
    ;;
  rtmdet_edge)
    export SNPE_PERSISTENT=1 SNPE_PERSISTENT_MODE=daemon
    model_args=(--mode=rtmdet_edge --model-key=rtmdet_s_8xb32_300e_coco
      --artifact="${RTMDET_SMALL_DLC:?Set RTMDET_SMALL_DLC and its .dlc.json manifest}"
      --imgsz=640 --confidence=0.10 --media-mode=none --priming-frames=0 --padding-frames=0)
    ;;
  dfine_cloud|rtdetr_cloud|rtdetrv2_small_cloud|lwdetr_large_cloud)
    case "$mode" in
      dfine_cloud) model_id=dfine_full ;;
      rtdetr_cloud|rtdetrv2_small_cloud) model_id="${RTDETRV2_MODEL_ID:-rtdetrv2_r18vd_sp2_120e_coco}" ;;
      lwdetr_large_cloud) model_id="${LWDETR_MODEL_ID:-lwdetr_large_coco}" ;;
    esac
    # H.264/RTSP carries input pixels; media-fps is a timestamp timebase, not pacing.
    model_args=(--mode="$mode" --model-key="$model_id" --media-mode=rtsp
      --media-fps=30 --width=1280 --height=720 --priming-frames=30 --padding-frames=30)
    ;;
  *) echo "Usage: bash $0 {yolo_edge|yolo11s_edge|rtmdet_edge|dfine_cloud|rtdetrv2_small_cloud|lwdetr_large_cloud} {grpc|kafka} [frames] [repetitions] [max-inflight]" >&2; exit 2 ;;
esac
python -u -m benchmark pipeline "${model_args[@]}" \
  --profile=pipeline_capacity --transport="$transport" \
  --grpc-target=192.168.180.249:50071 \
  --kafka-bootstrap=192.168.180.95:9092 --kafka-namespace=benchmark \
  --rtsp-base=rtsp://192.168.180.249:8554 \
  --video="$video" \
  --node-id=edge-246 --camera-id=traffic-01 \
  --frames="$frames" --warmup-frames=30 --repetitions="$repetitions" \
  --queue-depth=16 --max-inflight="$inflight" --run-timeout-s=600
