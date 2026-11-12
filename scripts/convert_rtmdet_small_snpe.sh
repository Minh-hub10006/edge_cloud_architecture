#!/usr/bin/env bash
# Run on SDK conversion host after sourcing the matching SNPE/QAIRT environment.
set -euo pipefail
onnx="${1:?Usage: $0 float.onnx calibration_dir output_dir [convert|quantize]}"
calibration="${2:?Calibration directory required}"
output="${3:?Output directory required}"
phase="${4:-convert}"
[[ -f "$onnx" && -f "$onnx.json" && -f "$calibration/input_list.txt" ]] || { echo "Missing source/manifest/calibration" >&2; exit 2; }
[[ "$phase" == convert || "$phase" == quantize ]] || { echo "Unknown phase: $phase" >&2; exit 2; }
for tool in snpe-onnx-to-dlc snpe-dlc-quantize snpe-dlc-info; do
  command -v "$tool" >/dev/null || { echo "Missing SDK tool: $tool" >&2; exit 2; }
done
python -c 'import json,sys; c=json.load(open(sys.argv[1])); assert c["input_layout"]=="nhwc", "expected NHWC calibration for converted SNPE input"' "$calibration/calibration.json"
if [[ "$phase" == convert ]]; then
  [[ ! -e "$output" ]] || { echo "Refusing to overwrite: $output" >&2; exit 2; }
  mkdir -p -- "$output"
  snpe-onnx-to-dlc --input_network "$onnx" --output_path "$output/rtmdet_small_fp32.dlc"
  snpe-dlc-info -i "$output/rtmdet_small_fp32.dlc" > "$output/fp32_info.txt"
  echo "CHECKPOINT: inspect fp32_info.txt: input NHWC [1,640,640,3], boxes [1,8400,4], scores [1,8400,80]."
  echo "STOP before quantization. Only after confirming IO, rerun phase quantize with RTMDET_INPUT_LAYOUT_CONFIRMED=nhwc."
  exit 0
fi
[[ "${RTMDET_INPUT_LAYOUT_CONFIRMED:-}" == nhwc ]] || { echo "IO/layout gate not confirmed" >&2; exit 2; }
[[ "${RTMDET_HTP_SOC:-}" == qcs6490 ]] || { echo "Target gate: set RTMDET_HTP_SOC=qcs6490; do not use SDK default SoC" >&2; exit 2; }
[[ -f "$output/rtmdet_small_fp32.dlc" && -f "$output/fp32_info.txt" ]] || { echo "Run convert and inspect IO first" >&2; exit 2; }
[[ ! -e "$output/rtmdet_small_int8.dlc" && ! -e "$output/int8_info.txt" ]] || { echo "Refusing to overwrite INT8 outputs" >&2; exit 2; }
/bin/bash "$(command -v snpe-dlc-quantize)" --input_dlc "$output/rtmdet_small_fp32.dlc" \
  --input_list "$calibration/input_list.txt" --output_dlc "$output/rtmdet_small_int8.dlc" \
  --enable_htp --htp_socs "$RTMDET_HTP_SOC"
snpe-dlc-info -i "$output/rtmdet_small_int8.dlc" > "$output/int8_info.txt"
echo "Conversion complete, NOT admitted: inspect input NHWC640 + output boxes/scores, then seal and test DSP parity."
