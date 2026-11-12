"""Mobile uplink traces for drl/sim_v3 (docs/PLAN_DRL_V3_VI.md section 13, option A of 26/10).

Shape from public traces, level from a cited median (no own measurement):
  * time variation: UCC traces (Raca et al., MMSys 2018 4G / MMSys 2020 5G, 1 sample/s), static phones only (fixed
    cameras); rows while the phone was downloading (State D); 4G: Dataset/static, 5G: Download/Static (file download,
    saturating). The UCC uplink column is ACK traffic, so the DOWNLINK series is used as the shape: each trace is
    divided by its median and clipped to [.1, 4] x median.
  * level: uplink median 4G 18 Mbit/s (Vienna 4G/5G Drive-Test Dataset, arXiv 2603.02638, phone uplink tests:
    medians 16-20 Mbit/s over three operators, checked 26/10); 5G 50 Mbit/s ASSUMED (shortcut: replace by a cited or
    measured uplink median before the paper).
  * extra RTT: the UCC 5G ping samples (PINGAVG, median ~75 ms) for both 4G and 5G (UCC 4G has no ping).
Traces are split by file: the first 70% (sorted names) train, the rest test.
"""
from __future__ import annotations

import csv
import glob
import re
from pathlib import Path

import numpy as np

ROOT = Path("C:/Users/ADMIN/datasets_net")
SOURCES = {"4g": "4g/Dataset/static/*.csv", "5g": "5g/5G-production-dataset/Download/Static/*.csv"}
UL_MEDIAN_MBPS = {"4g": 18.0, "5g": 50.0}
PING_GLOB = "5g/5G-production-dataset/Download/*/*.csv"


def _rows(path):
    with open(path, encoding="utf-8", errors="ignore") as f:
        return list(csv.DictReader(f))


def load(split, root=ROOT, ul_median=None):
    """-> {"4g": [(ul_mbps array, rtt_ms array), ...], "5g": [...]} for SimV3Config.net_traces."""
    ul_median = {**UL_MEDIAN_MBPS, **(ul_median or {})}
    pings = np.array([float(r["PINGAVG"]) for f in sorted(glob.glob(str(root / PING_GLOB)))
                      for r in _rows(f) if r.get("PINGAVG", "-") not in ("-", "")])
    if not len(pings):
        raise ValueError(f"no ping samples under {root}")
    out = {}
    for kind, pattern in SOURCES.items():
        files = sorted(glob.glob(str(root / pattern)))
        cut = max(1, round(.7 * len(files)))
        chosen = files[:cut] if split == "train" else files[cut:]
        traces = []
        for k, f in enumerate(chosen):
            dl = np.array([float(r["DL_bitrate"]) for r in _rows(f) if r["State"] == "D"]) / 1000
            if len(dl) < 60 or np.median(dl) <= 0:
                continue
            shape = np.clip(dl / np.median(dl), .1, 4.0)
            start = (k * 997) % len(pings)  # each trace its own stretch of the ping samples
            rtt = np.resize(np.roll(pings, -start), len(shape))
            traces.append((shape * ul_median[kind], rtt))
        out[kind] = traces
    return out


if __name__ == "__main__":
    for split in ("train", "test"):
        t = load(split)
        for kind, traces in t.items():
            ul = np.concatenate([u for u, _ in traces])
            rtt = np.concatenate([r for _, r in traces])
            print(f"{split} {kind}: {len(traces)} traces, {len(ul)} s, UL Mbit/s p10/p50/p90 "
                  f"{np.percentile(ul, [10, 50, 90]).round(1)}, RTT ms p50/p90 {np.percentile(rtt, [50, 90]).round(0)}")


# Mahimahi cellular traces (Netravali et al., USENIX ATC 2015; recorded with Saturator, Winstein et al. NSDI 2013):
# saturated LTE uplink, one line per 1500-byte delivery opportunity (ms). Used UNMODIFIED, binned to Mbit/s per s.
# ATT-LTE-driving.up is excluded (flat ~1 Mbit/s, operator-capped). Split by file (each split has a strong and a
# weak operator). Extra RTT: fixed, as mahimahi's mm-delay (sensitivity: mahimahi_rtt_ms).
MAHIMAHI_ROOT = Path("C:/Users/ADMIN/datasets_net/mahimahi")
MAHIMAHI_SPLIT = {"train": ("Verizon-LTE-driving.up", "TMobile-LTE-short.up"),
                  "test": ("TMobile-LTE-driving.up", "Verizon-LTE-short.up")}


# 5G uplink measured by the user (28/10, docs: iperf3 -t 600 -i 1 -P 8 -w 4M --get-server-output over a direct
# Tailscale path, VN operators): per-second [SUM] at the receiver ("Server output"), first 2 s (buffer burst) cut.
# Dropped: ul_5g_vt1_sang (window-limited), ul_5g_mang2 (both ends on one network). RTT: idle ping ~60 ms, fixed.
IPERF5G_ROOT = Path("C:/iperf3")
IPERF5G_SPLIT = {"train": ("ul_5g_vt1_toi_w4m.txt", "ul_5g_vnpt5g_20261028_0937.txt"),
                 "test": ("ul_5g_vnpt5g_20261028_1119.txt",)}
_SUM = re.compile(r"^\[SUM\]\s+([\d.]+)-([\d.]+)\s+sec\s+\S+\s+\S+\s+([\d.]+)\s+([KMG])bits/sec\s*$")


def parse_iperf_sum(text):
    """Per-second receiver Mbit/s from an iperf3 -P log (server section when present)."""
    if "Server output:" in text:
        text = text.split("Server output:", 1)[1]
    out = []
    for line in text.splitlines():
        m = _SUM.match(line)
        if m and abs(float(m[2]) - float(m[1]) - 1.0) < .05:
            out.append(float(m[3]) * {"K": 1e-3, "M": 1.0, "G": 1e3}[m[4]])
    return np.array(out[2:])


def load_5g_iperf(split, root=IPERF5G_ROOT, rtt_ms=60.0):
    """-> {"5g": [(ul_mbps per second, rtt_ms per second), ...]} for SimV3Config.net_traces."""
    out = []
    for name in IPERF5G_SPLIT[split]:
        ul = parse_iperf_sum((root / name).read_text(encoding="utf-8", errors="replace"))
        out.append((ul, np.full(len(ul), rtt_ms)))
    return {"5g": out}


def load_mahimahi(split, root=MAHIMAHI_ROOT, rtt_ms=50.0):
    """-> {"4g": [(ul_mbps per second, rtt_ms per second), ...]} for SimV3Config.net_traces."""
    out = []
    for name in MAHIMAHI_SPLIT[split]:
        t = np.loadtxt(root / name, dtype=np.int64)
        ul = np.bincount(t // 1000) * 1500 * 8 / 1e6
        out.append((ul, np.full(len(ul), rtt_ms)))
    return {"4g": out}
