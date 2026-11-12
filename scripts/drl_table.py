"""Print `drl.mappo eval` JSON lines as a compact table (one row per scenario x policy).

  python -B scripts/drl_table.py results/drl/eval.jsonl [--per-edge]
"""
import json
import sys

import numpy as np


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    # utf-8-sig: PowerShell redirection adds a BOM.
    rows = [json.loads(x) for x in open(argv[0], encoding="utf-8-sig") if x.strip()]
    ta = [k for k in rows[0] if k.startswith("ta_")]
    print("| scenario | policy | " + " | ".join(f"TA@{k[3:]}" for k in ta)
          + " | timely FPS | F1 | below A_min | p95 ms | Cloud share |")
    print("|---|---|" + "---|" * (len(ta) + 5))
    for r in rows:
        print(f"| {r['scenario']} | {r['policy']} | " + " | ".join(f"{r[k]:.3f}" for k in ta)
              + f" | {15 * r['timely']:.1f} | {r['f1']:.3f} | {r['below_a_min']:.3f} | {r['latency_p95_ms']:.0f} | "
              f"{r['cloud_share']:.2f} |")
    seeds = {}
    for r in rows:
        if r["policy"].startswith("actor:"):
            seeds.setdefault(r["scenario"], []).append(r)
    if seeds:
        print("\n| scenario | DRL seeds | return | timely | F1 | below A_min | F1 deficit | p95 ms | Cloud share |")
        print("|---|---|---|---|---|---|---|---|---|")
        for scenario, rs in seeds.items():
            cell = lambda k, f: f"{f(np.mean([r[k] for r in rs]))} ± {f(np.std([r[k] for r in rs]))}"  # noqa: E731
            print(f"| {scenario} | {len(rs)} | {cell('team_return', '{:.1f}'.format)} | {cell('timely', '{:.3f}'.format)} | "
                  f"{cell('f1', '{:.3f}'.format)} | {cell('below_a_min', '{:.3f}'.format)} | "
                  f"{cell('f1_deficit', '{:.4f}'.format)} | {cell('latency_p95_ms', '{:.0f}'.format)} | "
                  f"{cell('cloud_share', '{:.2f}'.format)} |")
    if "--per-edge" in argv:
        print("\n| scenario | policy | edge | timely | F1 | below A_min | Cloud share |\n|---|---|---|---|---|---|---|")
        for r in rows:
            for edge, m in r["per_edge"].items():
                if r["scenario"] == "real3":
                    print(f"| {r['scenario']} | {r['policy']} | {edge} | {m['timely']:.3f} | {m['f1']:.3f} | "
                          f"{m['below_a_min']:.3f} | {m['cloud_share']:.2f} |")


if __name__ == "__main__":
    main()
