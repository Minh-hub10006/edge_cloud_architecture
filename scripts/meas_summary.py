"""One table per scenario from the files pulled to the laptop (15/11): run tables + Edge / Cloud resources.

  python -B scripts/meas_summary.py C:/Users/ADMIN/v50_deploy/meas [S1a S1b ...]

Layout per scenario: <S>/e246|e252|e247/run_<host>.log + res_edge_*.csv (Linux, resource_log.py) or res_edge_*_sys_ /
_gpu_ (247, resource_log_win.sh), <S>/cloud/res_cloud_*.csv. Only the seconds while the cameras ran are used: Linux
Edge rows with agents_cores >= 0.3, 247 rows with python CPU >= 50 % of a core, Cloud rows inside the union of the
Edges' active windows. File stamps: local time (UTC+7) except 252, whose clock is UTC. When a folder holds several
resource files (a run repeated), the newest one is used. Cloud net_in counts Docker bridges too (overstated).
"""
import csv
import glob
import os
import re
import statistics as st
import sys
from datetime import datetime, timedelta, timezone

VN = timezone(timedelta(hours=7))


def stamp(path, utc=False):
    m = re.search(r"(\d{8})_(\d{6})", os.path.basename(path))
    return datetime.strptime("".join(m.groups()), "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc if utc else VN).timestamp()


def run_table(path):
    """run_<host>.log -> {cam: (mode, inferred, of, drops, errors, p50, p95, cpu, temp)} (the last table in the file)."""
    rows = {}
    for line in open(path, encoding="utf-8", errors="replace"):
        m = re.match(r"(cam-\d+)\s+\S+:(\w)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+) /\s+(\d+)\s+(\d+) /\s+(\d+)\s+(\S+)\s+(\S+)", line)
        if m:
            g = m.groups()
            rows[g[0]] = (g[1], int(g[2]), int(g[3]), int(g[4]), int(g[5]), int(g[8]), int(g[9]), g[10], g[11])
    return rows


def linux_edge(path, utc):
    r = [x for x in csv.DictReader(open(path)) if float(x["agents_cores"]) >= .3]
    if not r:
        return None, None
    t0 = stamp(path, utc)
    win = (t0 + float(r[0]["t_s"]), t0 + float(r[-1]["t_s"]))
    m = lambda k: st.mean(float(x[k]) for x in r)  # noqa: E731
    return {"s": len(r), "cpu%": m("cpu_pct"), "agents": m("agents_cores"), "decoders": m("decoders_cores"),
            "npu_d": m("npu_daemons_cores"), "ram": m("ram_mb"), "temp_max": max(float(x["temp_c"]) for x in r),
            "out": m("net_out_mbit"), "cma_min": min(float(x["cma_free_mb"]) for x in r)}, win


def win_edge(sys_path, gpu_path):
    rows = list(csv.reader(open(sys_path)))
    head, data = rows[0], rows[1:]
    py = [i for i, h in enumerate(head) if "Process(python" in h and "Processor Time" in h]
    nic = [i for i, h in enumerate(head) if "Bytes Sent" in h]
    val = lambda r, i: float(r[i]) if r[i].strip() else 0.0  # noqa: E731
    act = [r for r in data if sum(val(r, i) for i in py) >= 50]
    if not act:
        return None, None
    t = lambda r: datetime.strptime(r[0], "%m/%d/%Y %H:%M:%S.%f").replace(tzinfo=VN).timestamp()  # noqa: E731
    win = (t(act[0]), t(act[-1]))
    g = [ln.split(",") for ln in open(gpu_path) if ln.strip()]
    gt = lambda x: datetime.strptime(x[0].strip(), "%Y/%m/%d %H:%M:%S.%f").replace(tzinfo=VN).timestamp()  # noqa: E731
    g = [x for x in g if win[0] <= gt(x) <= win[1]]
    return {"s": len(act), "cpu%": st.mean(val(r, 1) for r in act),
            "agents": st.mean(sum(val(r, i) for i in py) for r in act) / 100,
            "out": st.mean(sum(val(r, i) for i in nic) for r in act) * 8 / 1e6,
            "gpu%": st.mean(float(x[1]) for x in g) if g else float("nan"),
            "gpu_mem": max(float(x[2]) for x in g) if g else float("nan"),
            # Windows has no thermal zones, GPU temperature only
            "temp_max": max(float(x[4]) for x in g) if g else float("nan")}, win


def cloud(path, win):
    t0 = stamp(path)
    r = [x for x in csv.DictReader(open(path)) if win[0] <= t0 + float(x["t_s"]) <= win[1]]
    if not r:
        return None
    m = lambda k: st.mean(float(x[k]) for x in r)  # noqa: E731
    return {"s": len(r), "gpu%": m("gpu_pct"), "gpu_max": max(float(x["gpu_pct"]) for x in r), "gpu_w": m("gpu_w"),
            "fast": m("fast_workers_cores"), "slow": m("slow_worker_cores"), "ocr": m("ocr_cores"),
            "triton": m("triton_cores"), "cpu%": m("cpu_pct"), "in": m("net_in_mbit")}


def newest(pattern):
    files = sorted(glob.glob(pattern))
    return files[-1] if files else None


def scenario(root, name):
    print(f"\n## {name}")
    wins = []
    print("| Edge | cam | mode | frames | of | % | drops | err | p50 ms | p95 ms |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    res = {}
    for host in ("246", "252", "247"):
        d = os.path.join(root, name, "e" + host)
        log = newest(os.path.join(d, "run_*.log"))
        for cam, v in sorted((run_table(log) if log else {}).items()):
            print(f"| {host} | {cam} | {v[0]} | {v[1]} | {v[2]} | {100 * v[1] / max(v[2], 1):.1f} | {v[3]} | {v[4]} | "
                  f"{v[5]} | {v[6]} |")
        if host == "247":
            s, g = newest(os.path.join(d, "*_sys_*.csv")), newest(os.path.join(d, "*_gpu_*.csv"))
            res[host], w = win_edge(s, g) if s and g else (None, None)
        else:
            f = newest(os.path.join(d, "res_edge_*.csv"))
            res[host], w = linux_edge(f, utc=host == "252") if f else (None, None)
        if w:
            wins.append(w)
    print("\n| Edge | s | CPU % | camera agents (cores) | decoders | NPU daemons | RAM MB | max C (247: GPU) | out Mbit/s | min CMA MB | GPU % |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for host, r in res.items():
        if r is None:
            print(f"| {host} | no active seconds found |"); continue
        f = lambda k, p=1: f"{r[k]:.{p}f}" if k in r else "-"  # noqa: E731
        print(f"| {host} | {r['s']} | {f('cpu%')} | {f('agents', 2)} | {f('decoders', 2)} | {f('npu_d', 2)} | "
              f"{f('ram', 0)} | {f('temp_max', 0)} | {f('out')} | {f('cma_min', 0)} | {f('gpu%')} |")
    c = newest(os.path.join(root, name, "cloud", "res_cloud_*.csv"))
    if c and wins:
        r = cloud(c, (min(w[0] for w in wins), max(w[1] for w in wins)))
        if r:
            print(f"\nCloud 249 ({r['s']} s): GPU {r['gpu%']:.1f} % (max {r['gpu_max']:.0f}), {r['gpu_w']:.0f} W, "
                  f"CPU {r['cpu%']:.1f} %, fast workers {r['fast']:.2f} cores, slow {r['slow']:.2f}, OCR {r['ocr']:.2f}, "
                  f"Triton {r['triton']:.2f}, net in {r['in']:.1f} Mbit/s (Docker counted twice)")


def main():
    root = sys.argv[1]
    names = sys.argv[2:] or sorted(n for n in os.listdir(root) if re.match(r"S\d", n))
    for n in names:
        scenario(root, n)


if __name__ == "__main__":
    main()
