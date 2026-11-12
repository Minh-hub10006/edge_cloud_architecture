"""P2 gate (docs/PLAN_DRL_V3B_VI.md): replay the v50.5 testbed stages in drl/sim_v3 and compare TA@200 and the
per-Edge, per-model path latency with results/test_v50 (state.jsonl medians).

  python -B scripts/sim_v3_check.py
"""
import collections
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from drl.sim_v3 import EDGES_V3, SimV3, SimV3Config, TableV3, metrics  # noqa: E402

NAMES = {"yolov8n": "yolo_edge", "yolo11s": "yolo11s_edge", "rtdetrv2": "rtdetr_cloud", "dfine_x": "dfine_x_cloud"}
E246, E247, E252 = EDGES_V3
DX, RT, Y = "dfine_x_cloud", "rtdetr_cloud", "yolo11s_edge"
STAGES = {  # (edge, mode) per stream; measured TA@200 (results/test_v50/TABLE_V50.md)
    "2 fixed models": ([(E246, DX), (E246, RT), (E246, Y), (E247, DX), (E247, RT), (E247, Y), (E252, DX), (E252, RT), (E252, Y)], .858),
    "3 sched fixed (5 valid)": ([(E247, DX)] * 3 + [(E246, RT), (E252, Y)], .859),
    "4 rules": ([(E247, DX)] * 3 + [(E246, DX), (E246, Y), (E246, Y)] + [(E252, Y)] * 3, .869),
    "5 fixed + probe": ([(E247, DX)] * 3 + [(E246, RT)] * 3 + [(E252, Y)] * 3, .859),
}
MEASURED_MS = {("edge-246", DX): 32.7, ("edge-246", RT): 26.1, ("edge-246", Y): 25.4, ("edge-247", DX): 23.0,
               ("edge-247", RT): 21.9, ("edge-247", Y): 16.6, ("edge-252", DX): 49.3, ("edge-252", RT): 44.4,
               ("edge-252", Y): 41.7}


def main():
    table = TableV3.load("results/detrac/table_fps_v50.jsonl", "test", NAMES)  # the table n_streams_eval used
    cfg = SimV3Config(vn_share=0)
    lat = collections.defaultdict(list)
    worst = 0.0
    for name, (plan, measured) in STAGES.items():
        ta = []
        for seed in range(20):
            sim = SimV3(table, cfg, seed)
            sim.reset(len(plan), [e for e, _ in plan])
            secs = []
            for _ in range(cfg.episode_decisions):
                secs += sim.step([(m, 15) for _, m in plan])
            ta.append(metrics(secs, 0)["ta200"])
            for s in secs:
                for r in s:
                    lat[(r["edge"], r["mode"])].append(r["latency"])
        worst = max(worst, abs(np.mean(ta) - measured))
        print(f"{name:26} sim TA@200 {np.mean(ta):.3f} +- {np.std(ta):.3f}  measured {measured:.3f}  diff {np.mean(ta) - measured:+.3f}")
    for k in sorted(lat):
        print(f"  latency {k[0]} {k[1]:14} sim {np.median(lat[k]):5.1f} ms  measured {MEASURED_MS[k]:5.1f} ms")
    print("GATE", "PASS" if worst <= .03 else "FAIL", f"(worst |diff| {worst:.3f} <= .03)")


if __name__ == "__main__":
    main()
