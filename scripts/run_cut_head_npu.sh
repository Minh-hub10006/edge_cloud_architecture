#!/usr/bin/env bash
# Edge half of the cut detector on the NPU (plan 4/11, phase 1, run alone): outputs for gate G2 + timing.
# Run on 246/252 inside the unpacked bundle (head_w8a8.dlc, head_w8a16.dlc, test_raw/*.raw):
#   bash run_cut_head_npu.sh
set -euo pipefail
Q="${QAIRT_ROOT:-$HOME/sigm/qairt/2.43.0.260128}"
[[ -d "$Q" ]] || Q=/root/sigm/qairt/2.43.0.260128   # 252 keeps the SDK under /root
export LD_LIBRARY_PATH="$Q/lib/aarch64-oe-linux-gcc11.2:${LD_LIBRARY_PATH:-}" ADSP_LIBRARY_PATH="$Q/lib/hexagon-v68/unsigned"
export PATH="$Q/bin/aarch64-oe-linux-gcc11.2:$PATH"
here="$(cd "$(dirname "$0")" && pwd)"
ls "$here"/test_raw/*.raw > "$here/test_list.txt"
temp() { cat /sys/class/thermal/thermal_zone*/temp 2>/dev/null | sort -n | tail -1 | awk '{print $1/1000}'; }
for name in ${CUT_DLCS:-w8a8 w8a16}; do
  dlc="$here/head_${name}.dlc"
  out="$here/npu_out_${name}"
  [[ ! -e "$out" ]] || { echo "Refusing to overwrite $out" >&2; exit 2; }
  echo "== $name: outputs for G2 (temp $(temp) C)"
  snpe-net-run --container "$dlc" --input_list "$here/test_list.txt" --use_dsp --output_dir "$out" >/dev/null
  echo "== $name: throughput 30 s, burst (temp $(temp) C)"
  snpe-throughput-net-run --container "$dlc" --duration 30 --use_dsp --perf_profile burst \
    --input_raw "$(head -1 "$here/test_list.txt")" 2>&1 | grep -iE "fps|throughput|infer|total" | tail -5
  echo "   temp after $(temp) C"
done
tar czhf "$here/npu_out_$(hostname).tgz" -C "$here" $(for n in ${CUT_DLCS:-w8a8 w8a16}; do echo npu_out_$n; done) test_list.txt
echo "Done: $here/npu_out_$(hostname).tgz (copy back to the laptop for the G2 check)"
