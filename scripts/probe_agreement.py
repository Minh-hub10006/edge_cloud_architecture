"""Probe as box agreement (the probe as designed on 23/10, docs/PLAN_PROBE_VI.md): on the same two frames a live probe
sends per second, F1 of the Edge YOLO11s boxes against the D-FINE-X boxes (IoU >= .5, greedy), instead of the count
of vehicles the Edge missed. Stored as a disagreement on the old 0-20 scale, 20 x (1 - F1), so the simulator,
the heuristics and the planners read it exactly like probe_missed (larger = harder).

  python -B scripts/probe_agreement.py
writes results/detrac/probe_agree.jsonl and results/vn_hard/probe_agree_vn.jsonl (rows of the existing probe tables).
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SETS = [  # probe table, Edge dump dir, reference dump dir, fps, frame offsets in the second, vehicle classes, output
    (ROOT / "results/detrac/probe_table.jsonl", ROOT / "results/detrac/quality/yolo11s",
     ROOT / "results/detrac/quality_live/dfine_x", 25, (6, 18), {2, 5, 7}, ROOT / "results/detrac/probe_agree.jsonl"),
    (ROOT / "results/vn_hard/probe_table_vn.jsonl", ROOT.parent / "v50_deploy/vn_cloud/vn_dump/yolo11s",
     ROOT.parent / "v50_deploy/vn_cloud/vn_dump/dfine_x", 15, (3, 10), {2, 3, 5, 7},  # VN labels include motorbikes
     ROOT / "results/vn_hard/probe_agree_vn.jsonl"),
]
EDGE_CONF, REF_CONF = .30, .65  # as scripts/hardness_signal.py probe_table


def agreement_f1(edge, ref):
    """F1 of edge boxes against ref boxes, greedy one-to-one matching at IoU >= .5; two empty frames agree fully."""
    if not edge and not ref:
        return 1.0
    used, matched = set(), 0
    for r in ref:
        best, best_k = 0.0, None
        for k, e in enumerate(edge):
            if k in used:
                continue
            w = max(0.0, min(r[2], e[2]) - max(r[0], e[0]))
            h = max(0.0, min(r[3], e[3]) - max(r[1], e[1]))
            inter = w * h
            union = (r[2] - r[0]) * (r[3] - r[1]) + (e[2] - e[0]) * (e[3] - e[1]) - inter
            if union > 0 and inter / union > best:
                best, best_k = inter / union, k
        if best >= .5:
            used.add(best_k)
            matched += 1
    p = matched / len(edge) if edge else 0.0
    r = matched / len(ref) if ref else 0.0
    return 2 * p * r / (p + r) if p + r else 0.0


def boxes(folder, seq, thr, classes):
    frames = json.loads((Path(folder) / f"{seq}.json").read_text(encoding="utf-8"))["frames"]
    return {f["frame"]: [b[2:6] for b in f["boxes"] if b[1] >= thr and int(b[0]) in classes] for f in frames}


def build(table, edge_dir, ref_dir, fps, offsets, classes, out):
    rows, cache, skipped = [], {}, 0
    for line in table.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        seq = r["sequence"]
        if seq not in cache:
            try:
                cache[seq] = boxes(edge_dir, seq, EDGE_CONF, classes), boxes(ref_dir, seq, REF_CONF, classes)
            except FileNotFoundError:
                cache[seq] = None
        if cache[seq] is None:
            skipped += 1
            continue
        edge, ref = cache[seq]
        frames = [r["second"] * fps + o for o in offsets]
        f1 = [agreement_f1(edge.get(f, []), ref.get(f, [])) for f in frames if f in ref]
        if f1:
            rows.append({k: r[k] for k in r if k != "probe_missed"} | {"probe_agree_f1": f1,
                                                                        "probe_disagree": [20 * (1 - x) for x in f1]})
    out.write_text("".join(json.dumps(x) + "\n" for x in rows), encoding="utf-8")
    return len(rows), skipped


def main():
    for args in SETS:
        n, skipped = build(*args)
        print(f"{args[-1].relative_to(ROOT)}: {n} seconds ({skipped} rows without dumps)")


if __name__ == "__main__":
    main()
