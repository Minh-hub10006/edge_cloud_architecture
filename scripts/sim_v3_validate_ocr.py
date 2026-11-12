"""Sim vs testbed for the live OCR model: N=14 (10 VN streams), the Cloud `rules` scheduler (4 streams D-FINE-X@15,
the rest YOLO11s@15), 4 or 8 OCR workers, test table. Testbed 30/10 (sweep4/KET_QUA_N14_OCR.md): OCR ~41 reads/s,
busy workers 2.5 / 3.9, read 62 / 95 ms, D-FINE-X processing mean 33 / 65-77 ms, path latency p50 47 / 85-125 ms.

  PYTHONPATH=. python -B scripts/sim_v3_validate_ocr.py
"""
import dataclasses
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from drl.sim_v3 import SimV3, TableV3, ocr_busy  # noqa: E402
from drl.train_v3 import config  # noqa: E402
from scripts.sim_v3_baselines import DX, PROBES, TABLE, rules  # noqa: E402


def run(table, cfg, workers, episodes=10, n=14):
    lat, reads, busy, read_ms, served = [], [], [], [], []
    for seed in range(episodes):
        sim = SimV3(table, dataclasses.replace(cfg, ocr_workers=workers), seed)
        sim.reset(n)
        for _ in range(cfg.episode_decisions):
            for sec in sim.step(rules(4, DX, 15)(sim)):
                lat += [r["latency"] for r in sec if r["mode"] == DX]
                r = sim.ocr_crops * sum(s.cell().get("new", 0) for s in sim.streams)
                b, ms, sv = ocr_busy(r, workers, sim.cfg)
                reads.append(r), busy.append(b), read_ms.append(ms), served.append(sv)
    return {"reads/s": np.mean(reads), "busy": np.mean(busy), "read ms": np.mean(read_ms), "served": np.mean(served),
            "D-FINE p50": np.percentile(lat, 50), "D-FINE p95": np.percentile(lat, 95)}


def main():
    table = TableV3.load(TABLE, "test", probe_paths=PROBES)
    base = dataclasses.replace(config("probe", "test", False), vn_share=10 / 14, ocr_read_ms=50.0, ocr_ms_per_crop=0.0)
    for kappa in (0.0, .25, .5):
        for workers in (4, 8):  # .45 reads/vehicle -> ~41 reads/s as on the testbed
            m = run(table, dataclasses.replace(base, ocr_crops_per_vehicle=.45, ocr_cap_exponent=kappa), workers)
            print(f"exponent {kappa} workers {workers}: " + " ".join(f"{k} {v:.2f}" for k, v in m.items()), flush=True)
    m = run(table, dataclasses.replace(base, ocr_read_ms=0.0, ocr_crops_per_vehicle=0.0), 4)
    print("no OCR: " + " ".join(f"{k} {v:.2f}" for k, v in m.items()))


if __name__ == "__main__":
    main()
