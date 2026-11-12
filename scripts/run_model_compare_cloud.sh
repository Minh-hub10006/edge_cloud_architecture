#!/usr/bin/env bash
# Cloud249: dump every candidate detector over the live-transcoded UA-DETRAC videos, then measure GPU capacity.
#   nohup bash scripts/run_model_compare_cloud.sh > ~/cand_run.log 2>&1 &
# Inputs: ~/cand_models/*.onnx (uploaded), ~/DETRAC-Live/*.mp4. Outputs: ~/cand_dumps/<name>/, ~/cand_capacity.jsonl.
# Resumable (finished videos are skipped). Stops if the GPU passes 80 C (checked between models).
set -euo pipefail
PY="${PY:-$HOME/miniconda3/envs/qnn/bin/python}"
VIDEOS="${VIDEOS:-$HOME/DETRAC-Live}"
M="$HOME/cand_models"
OUT="$HOME/cand_dumps"
cd "$(dirname "$0")/.."
hot() { t=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | head -1); [ "$t" -ge 80 ]; }
run() {  # name onnx normalize batch
  hot && { echo "GPU hot, stop"; exit 1; }
  echo "== $1 $(date +%T)"
  "$PY" -B scripts/detr_candidate_dump.py "$2" "$VIDEOS" "$OUT/$1" --crop-bottom 64 --batch "$4" --normalize "$3" \
    --confidence 0.05 | tail -n 2
}
run dfine_x   "$M/dfine_x_obj2coco.onnx"      none 8
run dfine_l   "$M/dfine_l_obj2coco.onnx"      none 8
run deim_x    "$M/deim_dfine_x_coco.onnx"     none 8
run deimv2_x  "$M/deimv2_dinov3_x_coco.onnx"  imagenet 8
run deimv2_l  "$M/deimv2_dinov3_l_coco.onnx"  imagenet 8
run rtdetrv2  "$HOME/cloud_artifacts/rtdetrv2_s_candidate_20261005/rtdetrv2_s.onnx" none 1
run lwdetr    "$HOME/cloud_artifacts/lwdetr_large_candidate_20261006/lwdetr_large_coco.onnx" none 1
rm -f "$HOME/cand_capacity.jsonl"
for n in dfine_x:dfine_x_obj2coco dfine_l:dfine_l_obj2coco deim_x:deim_dfine_x_coco deimv2_x:deimv2_dinov3_x_coco deimv2_l:deimv2_dinov3_l_coco; do
  hot && sleep 60
  "$PY" -B scripts/onnx_capacity.py "$M/${n#*:}.onnx" --name "${n%%:*}" --seconds 15 >> "$HOME/cand_capacity.jsonl"
done
"$PY" -B scripts/onnx_capacity.py "$HOME/cloud_artifacts/rtdetrv2_s_candidate_20261005/rtdetrv2_s.onnx" --name rtdetrv2 \
  --batches 1 --seconds 15 >> "$HOME/cand_capacity.jsonl"
echo "DONE $(date +%T)"; cat "$HOME/cand_capacity.jsonl"
