#!/usr/bin/env bash
# Cloud249 (v50 step 12 + PLAN_CLOUD_RES R3/R4): F1 of the Cloud choices on the JPEG a v50 Edge sends, per JPEG
# quality and model input size (RT-DETRv2-S at 480 / 640, D-FINE-X at 640), then model-only GPU capacity.
#   nohup bash scripts/run_jpeg_quality_cloud.sh > ~/jpeg_q.log 2>&1 &
# Inputs: ~/DETRAC-Live/*.mp4 (live-transcoded, 64 px identity band); ONLY = the 40 UA-DETRAC test videos.
# Quality 0 = the Edge's square resize without JPEG (baseline). Outputs ~/jpeg_q/<model>_q<Q>/ and ~/jpeg_q.tgz
# (score on the laptop with model_compare.py --fixed-threshold), ~/res_capacity.jsonl.
# Resumable (finished videos are skipped). The GPU fell off the bus twice on Oct 22 under ~300 W: dumps pace
# themselves under POWER_W (250 W) / 80 C and stop if nvidia-smi loses the GPU.
set -euo pipefail
PY="${PY:-$HOME/miniconda3/envs/qnn/bin/python}"
VIDEOS="${VIDEOS:-$HOME/DETRAC-Live}"
OUT="$HOME/jpeg_q"
POWER_W="${POWER_W:-250}"
DX="$HOME/cloud_artifacts/dfine_x_candidate_20261021/dfine_x_obj2coco.onnx"
RT="$HOME/cloud_artifacts/rtdetrv2_s_candidate_20261005/rtdetrv2_s.onnx"  # deployed export: batch 1 only
RT480="$HOME/cloud_artifacts/rtdetrv2_s_res/rtdetrv2_s_480.onnx"        # same weights, dynamic batch
cd "$(dirname "$0")/.."
ONLY=MVI_39031,MVI_39051,MVI_39211,MVI_39271,MVI_39311,MVI_39361,MVI_39371,MVI_39401,MVI_39501,MVI_39511,MVI_40701,MVI_40711,MVI_40712,MVI_40714,MVI_40742,MVI_40743,MVI_40761,MVI_40762,MVI_40763,MVI_40771,MVI_40772,MVI_40773,MVI_40774,MVI_40775,MVI_40792,MVI_40793,MVI_40851,MVI_40852,MVI_40853,MVI_40854,MVI_40855,MVI_40863,MVI_40864,MVI_40891,MVI_40892,MVI_40901,MVI_40902,MVI_40903,MVI_40904,MVI_40905
cool() {
  while [ "$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | head -1)" -ge 75 ]; do
    echo "GPU hot, waiting"; sleep 60
  done
}
run() {  # name onnx size batch quality [all]
  cool
  echo "== $1 q$5 $(date +%T)"
  local only=(--only "$ONLY")
  [ "${6:-}" = all ] && only=()
  # 0.3: F1 is scored at 0.6 / 0.65; keeps the dumps small. JPEG side = model input side.
  "$PY" -B scripts/detr_candidate_dump.py "$2" "$VIDEOS" "$OUT/${1}_q$5" --crop-bottom 64 --batch "$4" \
    --confidence 0.3 --labels contiguous "${only[@]}" --jpeg-quality "$5" --jpeg-size "$3" --input-size "$3" \
    --power-limit-w "$POWER_W" --max-temp-c 80 | tail -n 1
}
(cd "$(dirname "$RT480")" && sha256sum -c SHA256SUMS --status) || { echo "RT 480 ONNX checksum mismatch"; exit 1; }
run rtdetrv2_480 "$RT480" 480 4 0 all   # the resolution oracle needs train + test
for q in 0 95 85 70 50; do
  run dfine_x "$DX" 640 4 "$q"
  run rtdetrv2 "$RT" 640 1 "$q"
done
run rtdetrv2_480 "$RT480" 480 4 85
tar -C "$HOME" -czf "$HOME/jpeg_q.tgz" jpeg_q
echo "DUMPS DONE $(date +%T) $(du -h "$HOME/jpeg_q.tgz" | cut -f1)"
# Model-only capacity: short full-load bursts, small batches.
rm -f "$HOME/res_capacity.jsonl"
for spec in "dfine_x_640|$DX|640|1 4" "rtdetrv2_640|$RT|640|1" "rtdetrv2_480|$RT480|480|1 4"; do
  IFS='|' read -r name onnx size batches <<< "$spec"
  cool
  # shellcheck disable=SC2086  # batches is a space-separated list
  "$PY" -B scripts/onnx_capacity.py "$onnx" --name "$name" --input-size "$size" --batches $batches \
    --seconds 15 >> "$HOME/res_capacity.jsonl"
done
echo "ALL DONE $(date +%T)"
cat "$HOME/res_capacity.jsonl"
