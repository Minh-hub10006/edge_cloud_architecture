#!/usr/bin/env bash
# Measurement scenarios (15/11): one command per Edge per scenario, 4 cameras, resources logged alongside.
#   bash scripts/scenario.sh CODE HOST [SECONDS=180] [START_AT=HH:MM:SS]      HOST 246 | 247 | 252
#   S0  no camera (idle resources only, 60 s)     S1a 4 x N (YOLOv8n Edge)     S1b 4 x A (YOLO11s Edge)
#   S2a 4 x C (RT-DETR Cloud, JPEG)               S2b 4 x D (D-FINE-X Cloud)   S3  4 x B (RT-DETR split, 2-bit)
#   S4  mixed: 246 N B C D | 247 A B C D | 252 N A B D (the 15/11 runs)
# Same START_AT on the 3 Edges = they begin together. Run on 249 at the same time:
#   python3 -B scripts/resource_log.py --role cloud --seconds 300 --out ~/v4_res/CODE
# Resources go to ~/v4_res/CODE (Linux: resource_log.py --role edge; 247: resource_log_win.sh).
set -uo pipefail
code="${1:?S0|S1a|S1b|S2a|S2b|S3|S4}"; host="${2:?246|247|252}"; secs="${3:-180}"; start_at="${4:-}"
here="$(cd "$(dirname "$0")/.." && pwd)"
out="$HOME/v4_res/$code"; mkdir -p "$out"
case "$code" in
  S0) m="";; S1a) m="N N N N";; S1b) m="A A A A";; S2a) m="C C C C";; S2b) m="D D D D";; S3) m="B B B B";;
  S4) case "$host" in 246) m="N B C D";; 247) m="A B C D";; 252) m="N A B D";; esac;;
  *) echo "unknown scenario $code" >&2; exit 2;;
esac
case "$host" in
  246) : "${V4_DECODER:=v4l2h264dec}"; v=(vn9 vn6 vn9 vn6);;
  252) : "${V4_DECODER:=qtic2vdec}"; v=(vn6 vn9 vn6 vn9);;
  247) v=(vn9 vn6 vn9 vn6);;
  *) echo "host $host not supported" >&2; exit 2;;
esac
: "${V4_HW_SIZE:=640}" "${V4_SLOW_TASKS:=plates,count,congestion,crowd,gathering,fight,crash,loiter}"
export V4_HW_SIZE V4_SLOW_TASKS ${V4_DECODER:+V4_DECODER}
[[ "$host" == 247 ]] && unset V4_HW_SIZE   # OpenCV decode on 247: no hardware resize
res_secs=60
if [[ -n "$m" ]]; then
  res_secs=$(( secs + 300 ))   # upper bound only: stopped when the run ends
  # 247's logger starts only once the agents run (at START_AT): only the Linux loggers must also cover the wait
  [[ -n "$start_at" && "$host" != 247 ]] && res_secs=$(( res_secs + $(date -d "$start_at" +%s) - $(date +%s) ))
fi
if [[ "$host" == 247 ]]; then
  bash "$here/scripts/resource_log_win.sh" "$res_secs" "$out" "${m:+8}" & res=$!
else
  python3 -B "$here/scripts/resource_log.py" --role edge --seconds "$res_secs" --out "$out" & res=$!
fi
if [[ -n "$m" ]]; then
  read -r a b c d <<< "$m"
  spec="${v[0]}:$a ${v[1]}:$b ${v[2]}:$c ${v[3]}:$d"
  echo "== $code on $host: $spec, $secs s, results and resources in $out"
  bash "$here/scripts/run_edge_v4.sh" "$host" "$spec" "$secs" "$start_at" 2>&1 | tee "$out/run_$host.log"
  sleep 5; kill -TERM "$res" 2>/dev/null   # 15/11: stop the logger when the cameras end, not after a fixed time
fi
wait "$res"
echo "== $code on $host done: $(ls "$out" | tr '\n' ' ')"
