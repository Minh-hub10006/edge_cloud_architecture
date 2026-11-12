#!/usr/bin/env bash
# Per-frame detections of one deployed model over every video in a folder (offline quality table).
#
#   bash scripts/quality_dump.sh <model> <video_dir> <out_dir>      # model: yolov8n|yolo11s|rtdetrv2|lwdetr
#
# Same artifact, thresholds and adapter as live/benchmark (models.sh). Edge models run on the Edge
# (NPU INT8 / GPU ONNX), Cloud models on Cloud249. Every decoded frame is kept (--stride 1).
# Resumable: existing <out_dir>/<model>/<video>.json files are skipped.
# QUALITY_SHARD=i/n (e.g. 0/2 on Edge246, 1/2 on Edge252) takes every n-th video starting at i.
# QUALITY_CROP_BOTTOM=64 for videos made by scripts/live_transcode.py (drops the identity band).
set -euo pipefail

MODEL="${1:-}"; VIDEO_DIR="${2:-}"; OUT_DIR="${3:-}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source "$ROOT/scripts/models.sh"
[ -n "$OUT_DIR" ] && load_model "$MODEL" || { echo "Usage: bash scripts/quality_dump.sh yolov8n|yolo11s|rtdetrv2|lwdetr VIDEO_DIR OUT_DIR" >&2; exit 2; }
[ -d "$VIDEO_DIR" ] || fail Q0 "missing video folder $VIDEO_DIR"
shard="${QUALITY_SHARD:-0/1}"; index="${shard%/*}"; count="${shard#*/}"
[[ "$index" =~ ^[0-9]+$ && "$count" =~ ^[1-9][0-9]*$ && "$index" -lt "$count" ]] || fail Q0 "QUALITY_SHARD must be i/n with 0 <= i < n"

PY="${QUALITY_PYTHON:-$([ "$PLACEMENT" = edge ] && echo "$EDGE_PYTHON_DEFAULT" || echo "$CLOUD_PYTHON_DEFAULT")}"
export PYTHONPATH="$ROOT"
check_artifact Q0
export_model_runtime "$MODEL" || fail Q0 "model runtime"
[ "$PLACEMENT" = edge ] && apply_edge_sched Q0 "$OUT_DIR/$MODEL"
mkdir -p "$OUT_DIR/$MODEL"
extra=()
[ "$MODEL" = yolo11s ] && extra+=(--allow-diagnostic)

mapfile -t videos < <(find "$VIDEO_DIR" -maxdepth 1 -name '*.mp4' | sort)
[ "${#videos[@]}" -gt 0 ] || fail Q0 "no .mp4 in $VIDEO_DIR"
done_n=0; skipped=0; failed=0
# Before each video wait until the hottest thermal zone is below QUALITY_COOL_C (70): Edge252 has no heatsink
# and reaches 95 C within ~75 s of sustained load. No thermal zones (Windows 247) = no wait.
cool_down() {
  local t
  # Some 252 zones fail to read (Invalid argument); under pipefail that cat status must not end the wait.
  while t=$( { cat /sys/class/thermal/thermal_zone*/temp 2>/dev/null || true; } | sort -n | tail -1) && [ -n "$t" ] \
      && [ $(( t / 1000 )) -ge "${QUALITY_COOL_C:-75}" ]; do
    sleep 10
  done
}
for i in "${!videos[@]}"; do
  (( i % count == index )) || continue
  video="${videos[$i]}"; out="$OUT_DIR/$MODEL/$(basename "${video%.mp4}").json"
  if [ -s "$out" ]; then skipped=$((skipped + 1)); continue; fi
  cool_down
  # Write to a temporary name so an interrupted dump is never mistaken for a finished one.
  if "$PY" -B scripts/edge_parity.py dump --adapter "$BENCH_ADAPTER" --artifact "$ARTIFACT" --video "$video" \
       --out "$out.part" --frames 1000000000 --stride 1 --imgsz "$IMGSZ" --layout "$LAYOUT" \
       --confidence "$CONFIDENCE" --nms-iou "$NMS_IOU" --crop-bottom "${QUALITY_CROP_BOTTOM:-0}"        --hot-c "${QUALITY_HOT_C:-85}" --cool-c "${QUALITY_COOL_C:-75}" "${extra[@]}" >"$out.log" 2>&1; then
    mv "$out.part" "$out"; done_n=$((done_n + 1))
    gate Q1 "$(basename "$video") $(tail -1 "$out.log")"
  else
    failed=$((failed + 1)); gate Q1 "FAIL $(basename "$video"); see $out.log"
  fi
done
gate Q2 "$MODEL shard $shard: dumped=$done_n skipped_existing=$skipped failed=$failed -> $OUT_DIR/$MODEL"
[ "$failed" = 0 ]
