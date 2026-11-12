#!/usr/bin/env bash
# Edge half of a cut Cloud detector -> SNPE DLC (plan 4/11, phase 1). Run on 249 after sourcing SNPE/QAIRT 2.43.
#   convert_cut_head_snpe.sh head.onnx CALIB_DIR OUT_DIR convert   # FP32 DLC + IO info, then STOP and check IO
#   CUT_IO_CONFIRMED=1 convert_cut_head_snpe.sh head.onnx CALIB_DIR OUT_DIR quantize   # W8A8 + W8A16 for qcs6490
set -euo pipefail
onnx="${1:?Usage: $0 head.onnx calib_dir out_dir convert|quantize}"
calib="${2:?calibration dir}"
out="${3:?output dir}"
phase="${4:-convert}"
[[ -f "$onnx" && -f "$calib/input_list.txt" && -f "$calib/calibration.json" ]] || { echo "Missing head.onnx or calibration" >&2; exit 2; }
for tool in snpe-onnx-to-dlc snpe-dlc-quantize snpe-dlc-info; do
  command -v "$tool" >/dev/null || { echo "Missing SDK tool: $tool (source the SNPE 2.43 env first)" >&2; exit 2; }
done
python3 -c 'import json,sys; c=json.load(open(sys.argv[1])); assert c["input_layout"]=="nhwc"' "$calib/calibration.json"
if [[ "$phase" == convert ]]; then
  [[ ! -e "$out" ]] || { echo "Refusing to overwrite: $out" >&2; exit 2; }
  mkdir -p -- "$out"
  sha256sum "$onnx" > "$out/head_onnx.sha256"
  snpe-onnx-to-dlc --input_network "$onnx" --input_dim images 1,3,640,640 --output_path "$out/head_fp32.dlc"
  snpe-dlc-info -i "$out/head_fp32.dlc" > "$out/fp32_info.txt"
  echo "CHECKPOINT: in fp32_info.txt the input must be [1,640,640,3] and the output [1,80,80,128] (layer2, NHWC)."
  echo "Then rerun with CUT_IO_CONFIRMED=1 ... quantize"
  exit 0
fi
if [[ "$phase" == quantize_pc ]]; then
  # Round 2 (4/11): G2 failed with the defaults (corr .98, A16 barely better -> weight error). Per-channel weights,
  # 32-bit bias, and the same with cross-layer equalization (CLE), activations 8 and 16 bit.
  [[ "${CUT_IO_CONFIRMED:-}" == 1 && -f "$out/head_fp32.dlc" ]] || { echo "Convert + IO check first" >&2; exit 2; }
  for v in "pc 8" "pc 16" "pccle 8"; do
    set -- $v
    dlc="$out/head_w8a${2}_${1}.dlc"
    [[ ! -e "$dlc" ]] || { echo "Refusing to overwrite: $dlc" >&2; exit 2; }
    extra=(--use_per_channel_quantization --bias_bitwidth 32)
    [[ "$1" == pccle ]] && extra+=(--algorithms cle)
    /bin/bash "$(command -v snpe-dlc-quantize)" --input_dlc "$out/head_fp32.dlc" --input_list "$calib/input_list.txt" \
      --output_dlc "$dlc" --act_bitwidth "$2" "${extra[@]}" --enable_htp --htp_socs qcs6490
  done
  sha256sum "$out"/*.dlc > "$out/dlc.sha256"
  echo "Done: head_w8a8_pc.dlc, head_w8a16_pc.dlc, head_w8a8_pccle.dlc (gate G2 still to check)."
  exit 0
fi
[[ "$phase" == quantize ]] || { echo "Unknown phase: $phase" >&2; exit 2; }
[[ "${CUT_IO_CONFIRMED:-}" == 1 ]] || { echo "Check fp32_info.txt IO first, then set CUT_IO_CONFIRMED=1" >&2; exit 2; }
[[ -f "$out/head_fp32.dlc" ]] || { echo "Run the convert phase first" >&2; exit 2; }
for a in 8 16; do
  dlc="$out/head_w8a${a}.dlc"
  [[ ! -e "$dlc" ]] || { echo "Refusing to overwrite: $dlc" >&2; exit 2; }
  /bin/bash "$(command -v snpe-dlc-quantize)" --input_dlc "$out/head_fp32.dlc" --input_list "$calib/input_list.txt" \
    --output_dlc "$dlc" --act_bitwidth "$a" --enable_htp --htp_socs qcs6490
  snpe-dlc-info -i "$dlc" > "$out/w8a${a}_info.txt"
done
sha256sum "$out"/*.dlc > "$out/dlc.sha256"
echo "Done: head_w8a8.dlc, head_w8a16.dlc (NOT admitted until the NPU-vs-float tensor check, gate G2)."
