#!/usr/bin/env bash
# 247 (Windows, Git Bash) resources once per second (15/11): the Windows side of resource_log.py --role edge,
# built-in tools only (typeperf, nvidia-smi). Stops by itself after SECONDS.
#   bash scripts/resource_log_win.sh 260 ~/v4_res [WAIT_PYTHON=0]
# typeperf finds processes once, when it starts: WAIT_PYTHON=4 first waits (up to 5 min) for 4 python processes (the
# camera agents) so their columns exist.
# Writes res_edge_<host>_sys_<time>.csv (CPU % of all cores, free RAM MB, network bytes/s per interface, CPU % and
# working set of every python process; process CPU % is of one core: 100 = one core) and res_edge_<host>_gpu_<time>.csv.
set -euo pipefail
secs="${1:-260}"; out="${2:-$HOME/v4_res}"; want="${3:-0}"; mkdir -p "$out"
for _ in $(seq 300); do
  (( $(tasklist //FI "IMAGENAME eq python.exe" //NH | grep -ci python) >= want )) && break
  sleep 1
done
tag="res_edge_$(hostname)"; stamp="$(date +%Y%m%d_%H%M%S)"
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,power.draw,temperature.gpu \
  --format=csv,noheader,nounits -l 1 > "$out/${tag}_gpu_$stamp.csv" &
gpu=$!
typeperf -si 1 -sc "$secs" -f CSV -y -o "$(cygpath -w "$out/${tag}_sys_$stamp.csv")" \
  '\Processor(_Total)\% Processor Time' '\Memory\Available MBytes' \
  '\Network Interface(*)\Bytes Received/sec' '\Network Interface(*)\Bytes Sent/sec' \
  '\Process(python*)\% Processor Time' '\Process(python*)\Working Set' \
  '\Process(ffmpeg*)\% Processor Time' '\Process(mediamtx*)\% Processor Time' > /dev/null
kill "$gpu" 2>/dev/null || true
ls -la "$out/${tag}_"*"_$stamp.csv"
