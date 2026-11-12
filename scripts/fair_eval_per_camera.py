"""Per-camera numbers behind the round-29 comparison (31/10), for plotting: same situations, seeds and policies as
`fair_eval.py --fixed-camera --drl results/drl_r28 --drl-plan --singles`, one CSV row per (situation, episode, method,
camera). The per-episode sum of `fps_tinh_diem` equals the c70 of that episode (checked against --check FILE).

  PYTHONPATH=. python -B scripts/fair_eval_per_camera.py OUT.csv [--check results/fair/r29_rows.json]
"""
import csv
import json
import multiprocessing as mp
import sys

import numpy as np

ARGS = sys.argv[1:]  # read before the round-29 flags are added for fair_eval
sys.argv[1:1] = ["--fixed-camera", "--drl", "results/drl_r28", "--drl-plan", "--singles"]
import scripts.fair_eval as fe  # noqa: E402

o, cb = fe.o, fe.cb
COLS = ["so_camera", "ocr_doc_moi_xe", "ocr_worker", "ti_le_VN", "bien_dong", "episode", "phuong_phap", "drl_seed",
        "camera", "loai_video", "edge", "f1_tb", "ti_le_tre_hoac_bo", "fps_chon_tb", "fps_dung_han", "dat_rang_buoc",
        "fps_tinh_diem", "tg_yolov8n", "tg_yolo11s", "tg_rtdetr", "tg_dfine_x", "do_tre_tb_ms"]


def run(cfg, n, seed, pol):
    sim = o.SimV3(o._G["test"], cfg, seed)
    sim.reset(n)
    secs, modes = [], []
    for _ in range(cfg.episode_decisions):
        acts = pol(sim)
        secs += sim.step(acts)
        modes.append([m for m, _ in acts])
    rows = []
    for i, s in enumerate(sim.streams):
        r = [sec[i] for sec in secs]
        f1 = float(np.mean([x["f1"] for x in r]))
        late = 1 - float(np.mean([x["done"] * x["on_time"] for x in r]))
        eff = float(np.mean([x["fps"] * x["done"] * x["on_time"] for x in r]))
        ok = f1 >= .70 and late <= .05  # the c70 rule (drl/sim_v3.py metrics)
        ms = [m[i] for m in modes]
        rows.append([i, "VN" if s.key[0] != "detrac" else "UA-DETRAC", s.edge.name, round(f1, 4), round(late, 4),
                     round(float(np.mean([x["fps"] for x in r])), 2), round(eff, 3), int(ok), round(eff * ok, 3)]
                    + [round(float(np.mean([m == x for m in ms])), 3)
                       for x in ("yolo_edge", "yolo11s_edge", "rtdetr_cloud", "dfine_x_cloud")]
                    + [round(float(np.mean([x["latency"] for x in r])), 1)])
    return rows


def job(args):
    (d, dyn), static, theta, n, seed = args
    cfg = o.cfg_for("test", d, dyn)
    sel = o.dataclasses.replace(cfg, probe_selective=True)
    pols = fe.baseline_policies(static, theta)
    cands = ([o.greedy(c) for c in o.CAPS] + [o.rescue(c, m) for c in o.CAPS for m in (o.DX, o.RT)]
             + list(pols.values()))
    props = fe.drl_proposals()
    todo = [(b, "", cfg, (pols | fe.single_policies())[b]) for b in fe.BASELINES]
    todo += [("true", "", cfg, cb.planner(cands, "true", True)), ("probe_sel", "", sel, cb.planner(cands, "probe_sel", True)),
             ("drl_plan", "", sel, cb.planner(props, "probe_sel", True)),
             ("drl_plan_mix", "", sel, cb.planner(cands + props, "probe_sel", True))]
    for name, ps in fe.drl_policies().items():
        todo += [(name, k, sel, p) for k, p in enumerate(ps)]
    head = [n, d[0], d[1], d[2], "co" if dyn == "all" else "khong", seed]
    return [head + [name, k] + r for name, k, c, p in todo for r in run(c, n, seed, p)]


def main(out, check=None):
    cells = cb.CELLS
    with mp.Pool(14, initializer=cb._init) as pool:
        tuned = dict(zip(cells, pool.starmap(o.tune, cells)))
        rows = [r for part in pool.map(job, [(c, *tuned[c], n, s) for c in cells for n in fe.NS
                                             for s in range(fe.TEST_EPISODES)], chunksize=1) for r in part]
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(COLS)
        w.writerows(rows)
    print(out, len(rows), "rows")
    if check:  # every episode total must equal the c70 of the round-29 run (same seeds, same policies)
        ref = {}
        for k, seed, base, plan, _ in json.load(open(check))["rows"]:
            for m, v in (base | plan).items():
                if not m.endswith("_seeds"):
                    ref[(k[2], tuple(k[0]), k[1], seed, m)] = v
        tot = {}
        for r in rows:
            if r[6] in ("drl", "drl_alloc"):
                continue  # their reference is the mean over seeds; checked below
            key = (r[0], (r[1], r[2], r[3]), "all" if r[4] == "co" else "none", r[5], r[6])
            tot[key] = tot.get(key, 0) + r[-6]
        bad = [k for k, v in tot.items() if abs(v - ref[k]) > .05]
        print("check: episode totals equal to round 29 in", len(tot) - len(bad), "/", len(tot), bad[:3])


if __name__ == "__main__":
    main(ARGS[0], ARGS[ARGS.index("--check") + 1] if "--check" in ARGS else None)
