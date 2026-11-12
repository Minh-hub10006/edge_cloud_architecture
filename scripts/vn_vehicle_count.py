"""Vehicles per frame, averaged per second, for VN-Hard (auto labels, 15 frames/s; DETRAC rows already carry
vehicle_count). Feeds the downstream-load model of drl/sim_v3 (Cloud GPU share taken by per-vehicle tasks).

  python -B scripts/vn_vehicle_count.py  ->  results/vn_hard/vehicle_count_vn.jsonl
"""
import glob
import json
import os

out = []
for f in sorted(glob.glob("results/vn_hard/labels_auto/*.jsonl")):
    split, seq = os.path.basename(f)[:-6].split("_", 1)
    per = {}
    for line in open(f, encoding="utf-8"):
        r = json.loads(line)
        per.setdefault(r["frame"] // 15, []).append(len(r["boxes"]))
    out += [{"split": split, "sequence": seq, "dataset": "vn_hard", "second": s, "vehicle_count": sum(v) / len(v)}
            for s, v in sorted(per.items())]
assert out, "no VN labels found"
with open("results/vn_hard/vehicle_count_vn.jsonl", "w", encoding="utf-8") as fh:
    fh.writelines(json.dumps(r) + "\n" for r in out)
print(len(out), "seconds; mean vehicles/frame", round(sum(r["vehicle_count"] for r in out) / len(out), 1))
