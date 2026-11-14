"""Resources of one run, once per second (14/11). Read only; stops by itself after --seconds.

  python3 -B scripts/resource_log.py --role cloud --seconds 260 --out ~/v4_res     # 249
  python3 -B scripts/resource_log.py --role edge  --seconds 260 --out ~/v4_res     # 246 / 252 (Linux)

Writes <out>/res_<role>_<host>_<time>.csv (one row per second) and prints a summary (mean / p95 / max):
- machine: CPU % (of all cores), RAM used, network in / out (Mbit/s, all interfaces but lo), hottest thermal zone,
  CmaFree (Edge);
- per process group (CPU in cores, RAM MB): Edge = camera agents, GStreamer decoders, NPU daemons, MediaMTX, ffmpeg;
  Cloud = fast-lane workers, slow-lane worker, OCR, DB writer, Triton;
- Cloud: GPU % / memory / power / temperature (nvidia-smi), and per Triton model between start and end: requests,
  executions, mean batch (= inferences / executions), mean queue and compute time per request (metrics :18002).
NPU load on QCS6490 has no counter we can read: not measured.
"""
import argparse
import csv
import os
import re
import socket
import subprocess
import time
import urllib.request

GROUPS = {
    "edge": {"agents": r"control_v1", "decoders": r"gst-launch", "npu_daemons": r"snpe-daemon",
             "mediamtx": r"mediamtx", "ffmpeg": r"ffmpeg"},
    "cloud": {"fast_workers": r"v4-cloud|cloud_core\.worker", "slow_worker": r"v4-slow-worker",
              "ocr": r"v4-plate-ocr", "writer": r"v4-results-writer", "triton": r"tritonserver"},
}
TICK = os.sysconf("SC_CLK_TCK")


def cpu_total():
    f = [int(x) for x in open("/proc/stat").readline().split()[1:]]
    return sum(f), f[3] + f[4]   # all, idle + iowait


def mem_used_mb():
    m = {k: int(v.split()[0]) for k, v in (l.split(":", 1) for l in open("/proc/meminfo"))}
    return (m["MemTotal"] - m["MemAvailable"]) / 1024, m.get("CmaFree", 0) / 1024


def net_bytes():
    rx = tx = 0
    for line in open("/proc/net/dev").readlines()[2:]:
        name, data = line.split(":", 1)
        if name.strip() != "lo":
            f = data.split()
            rx, tx = rx + int(f[0]), tx + int(f[8])
    return rx, tx


def max_temp():
    best = None
    for z in os.listdir("/sys/class/thermal"):
        try:
            t = int(open(f"/sys/class/thermal/{z}/temp").read()) / 1000
            best = t if best is None or t > best else best
        except (OSError, ValueError):
            pass
    return best


def procs(groups):
    """pid -> (group, cpu ticks, rss MB) for every matching process."""
    out = {}
    for pid in filter(str.isdigit, os.listdir("/proc")):
        try:
            cmd = open(f"/proc/{pid}/cmdline", "rb").read().replace(b"\0", b" ").decode(errors="replace")
            stat = open(f"/proc/{pid}/stat").read().rsplit(")", 1)[1].split()
        except OSError:
            continue
        for g, pat in groups.items():
            if re.search(pat, cmd):
                out[pid] = (g, int(stat[11]) + int(stat[12]), int(stat[21]) * os.sysconf("SC_PAGE_SIZE") / 2**20)
                break
    return out


def gpu():
    try:
        line = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,power.draw,temperature.gpu",
                               "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5).stdout
        return [float(x) for x in line.split("\n")[0].split(",")]
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return [None] * 4


def triton_metrics(url):
    """model -> {counter: value} for the counters we need."""
    want = ("nv_inference_request_success", "nv_inference_exec_count", "nv_inference_count",
            "nv_inference_queue_duration_us", "nv_inference_compute_infer_duration_us")
    out = {}
    try:
        text = urllib.request.urlopen(url, timeout=5).read().decode()
    except OSError:
        return out
    for line in text.splitlines():
        m = re.match(r'(\w+)\{[^}]*model="([^"]+)"[^}]*\} ([0-9.e+]+)', line)
        if m and m.group(1) in want:
            out.setdefault(m.group(2), {})[m.group(1)] = out.get(m.group(2), {}).get(m.group(1), 0) + float(m.group(3))
    return out


def stats(xs):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return "n/a"
    return f"mean {sum(xs) / len(xs):7.1f}  p95 {xs[min(len(xs) - 1, int(.95 * len(xs)))]:7.1f}  max {xs[-1]:7.1f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", choices=("edge", "cloud"), required=True)
    ap.add_argument("--seconds", type=float, default=260)
    ap.add_argument("--out", default=os.path.expanduser("~/v4_res"))
    ap.add_argument("--triton-metrics", default="http://localhost:18002/metrics")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, f"res_{a.role}_{socket.gethostname()}_{time.strftime('%Y%m%d_%H%M%S')}.csv")
    groups = GROUPS[a.role]
    ncpu = os.cpu_count()
    t0 = time.monotonic()
    tri0 = triton_metrics(a.triton_metrics) if a.role == "cloud" else {}
    prev = (cpu_total(), net_bytes(), procs(groups), time.monotonic())
    cols = (["t_s", "cpu_pct", "ram_mb", "cma_free_mb", "net_in_mbit", "net_out_mbit", "temp_c"]
            + [f"{g}_cores" for g in groups] + [f"{g}_mb" for g in groups]
            + (["gpu_pct", "gpu_mem_mb", "gpu_w", "gpu_c"] if a.role == "cloud" else []))
    rows = []
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        while time.monotonic() - t0 < a.seconds:
            time.sleep(max(0.0, 1 - (time.monotonic() - prev[3])))
            cur = (cpu_total(), net_bytes(), procs(groups), time.monotonic())
            dt = cur[3] - prev[3]
            busy = 1 - (cur[0][1] - prev[0][1]) / max(1, cur[0][0] - prev[0][0])
            ram, cma = mem_used_mb()
            row = [round(cur[3] - t0, 1), round(100 * busy, 1), round(ram), round(cma),
                   round((cur[1][0] - prev[1][0]) * 8 / dt / 1e6, 2), round((cur[1][1] - prev[1][1]) * 8 / dt / 1e6, 2),
                   max_temp()]
            # per process, only those alive in both samples (14/11: a process ending made a group's sum go negative)
            cores, mb = dict.fromkeys(groups, 0.0), dict.fromkeys(groups, 0.0)
            for pid, (g, ticks, rss) in cur[2].items():
                mb[g] += rss
                if pid in prev[2]:
                    cores[g] += (ticks - prev[2][pid][1]) / TICK / dt
            row += [round(cores[g], 2) for g in groups] + [round(mb[g]) for g in groups]
            if a.role == "cloud":
                row += gpu()
            w.writerow(row)
            f.flush()
            rows.append(dict(zip(cols, row)))
            prev = cur
    print(f"== {a.role} {socket.gethostname()}: {len(rows)} s, {ncpu} cores -> {path}")
    for c in cols[1:]:
        print(f"{c:18s} {stats([r[c] for r in rows])}")
    if a.role == "cloud":
        tri1 = triton_metrics(a.triton_metrics)
        print(f"{'Triton model':16s} {'requests':>9s} {'execs':>7s} {'mean batch':>10s} {'queue ms':>9s} {'compute ms':>10s}")
        for m in sorted(tri1):
            d = {k: tri1[m].get(k, 0) - tri0.get(m, {}).get(k, 0) for k in tri1[m]}
            req, ex = d.get("nv_inference_request_success", 0), d.get("nv_inference_exec_count", 0)
            if req:
                print(f"{m:16s} {req:9.0f} {ex:7.0f} {d.get('nv_inference_count', 0) / max(ex, 1):10.2f} "
                      f"{d.get('nv_inference_queue_duration_us', 0) / req / 1e3:9.1f} "
                      f"{d.get('nv_inference_compute_infer_duration_us', 0) / req / 1e3:10.1f}")


if __name__ == "__main__":
    main()
