"""Report figure (1/11): FPS c70 vs the number of streams N = 4..12 in the main situation (mostly VN, OCR as on
the testbed, 8 OCR workers, steady traffic), every single model, the hand rules, the heuristics and DRL round 28.
10 test episodes per N (same seeds for every method); DRL = mean over its 5 seeds per episode.

  PYTHONPATH=. python -B scripts/plot_by_n.py OUT_DIR
"""
import csv
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np

OUT = Path(sys.argv[1])
sys.argv[1:] = ["--fixed-camera", "--drl", "results/drl_r28"]
import scripts.fair_eval as fe  # noqa: E402

o, cb = fe.o, fe.cb
CELL = ((.45, 8, .7), "none")
NS, EPISODES = (4, 6, 8, 10, 12), 10
GLOBAL_RULE = (.8, o.RT, 15)  # one hand rule for all situations, chosen on train (round 32)
METHODS = [  # key, label, color (reference palette order; D-FINE-X is always 0 -> neutral gray)
    ("drl", "DRL", "#2a78d6"),
    ("all_v8n", "Tất cả YOLOv8n (Edge)", "#eb6834"), ("all_y11s", "Tất cả YOLO11s (Edge)", "#1baf7a"),
    ("allrt", "Tất cả RT-DETRv2 (Cloud)", "#eda100"), ("all_dx", "Tất cả D-FINE-X (Cloud)", "#8a8984"),
    ("rule_global", "Luật tay chung (80 % lên RT-DETR)", "#008300"),
    ("static", "Luật tay chỉnh riêng tình huống", "#6250d6"),
    ("thresh", "Heuristic ngưỡng probe", "#e34948"), ("net", "Heuristic theo mạng", "#e87ba4"),
]


def job(args):
    static, theta, n, seed = args
    cfg = o.cfg_for("test", *CELL)
    sel = o.dataclasses.replace(cfg, probe_selective=True)
    pols = fe.baseline_policies(static, theta) | fe.single_policies() | {"rule_global": o.rules(*GLOBAL_RULE)}
    out = {k: o.episode(o._G["test"], cfg, n, seed, p)[fe.KEY] for k, p in pols.items()}
    seeds = [o.episode(o._G["test"], sel, n, seed, p)[fe.KEY] for p in fe.drl_policies()["drl"]]
    out["drl"] = float(np.mean(seeds))
    return n, seed, out, seeds


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    OUT.mkdir(parents=True, exist_ok=True)
    with mp.Pool(14, initializer=cb._init) as pool:
        static, theta = pool.starmap(o.tune, [CELL])[0]
        rows = pool.map(job, [(static, theta, n, s) for n in NS for s in range(EPISODES)], chunksize=1)
    print("luat tay chinh rieng (chon tren train):", static, "| heuristic nguong:", theta)
    mean = {m: [np.mean([r[2][m] for r in rows if r[0] == n]) for n in NS] for m, _, _ in METHODS}
    std = {m: [np.std([r[2][m] for r in rows if r[0] == n], ddof=1) for n in NS] for m, _, _ in METHODS}

    with open(OUT / "so_lieu_theo_N.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["N", "episode"] + [lab for _, lab, _ in METHODS] + [f"DRL seed {k}" for k in range(5)])
        for n, seed, out, seeds in sorted(rows, key=lambda r: (r[0], r[1])):
            w.writerow([n, seed] + [round(out[m], 2) for m, _, _ in METHODS] + [round(x, 2) for x in seeds])
    with open(OUT / "bang_trung_binh_theo_N.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["phuong_phap"] + [f"N={n} (TB +- do lech)" for n in NS])
        for m, lab, _ in METHODS:
            w.writerow([lab] + [f"{a:.1f} +- {s:.1f}" for a, s in zip(mean[m], std[m])])
        for ref in ("rule_global", "static"):
            line = []
            for n in NS:
                d = np.array([r[2]["drl"] - r[2][ref] for r in rows if r[0] == n])
                base = np.mean([r[2][ref] for r in rows if r[0] == n])
                line.append(f"{d.mean() / max(base, 1e-9) * 100:+.1f}% (p={fe.sign_flip_p(d):.3f})")
            w.writerow([f"DRL so voi {dict((k, l) for k, l, _ in METHODS)[ref]}"] + line)
    sys.stdout.reconfigure(encoding="utf-8")
    print(open(OUT / "bang_trung_binh_theo_N.csv", encoding="utf-8-sig").read())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6), sharey=True, facecolor="#fcfcfb")
    panels = (("DRL so với từng model lẻ", ["all_v8n", "all_y11s", "allrt", "all_dx", "drl"]),
              ("DRL so với luật tay và heuristic", ["rule_global", "static", "thresh", "net", "drl"]))
    info = {m: (lab, c) for m, lab, c in METHODS}
    for ax, (title, keys) in zip(axes, panels):
        ax.set_facecolor("#fcfcfb")
        for m in keys:
            lab, c = info[m]
            lw = 3 if m == "drl" else 2
            ax.plot(NS, mean[m], color=c, lw=lw, marker="o", ms=8, ls="--" if m == "all_dx" else "-",
                    markeredgecolor="#fcfcfb", markeredgewidth=2, label=lab, zorder=3 if m == "drl" else 2)
        ax.set_title(title, loc="left", fontsize=11, color="#0b0b0b")
        ax.set_xticks(NS, [f"N={n}" for n in NS])
        ax.set_xlabel("Số luồng camera", color="#52514e")
        ax.grid(axis="y", color="#e4e3df", lw=.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    axes[0].set_ylabel("FPS c70 (tổng FPS của camera đạt F1 ≥ 0,7 và trễ ≤ 5 %)", color="#52514e")
    fig.suptitle("Tình huống chính: phần lớn video VN, OCR như testbed, 8 worker OCR (10 episode mỗi N)",
                 x=.01, ha="left", fontsize=12, color="#0b0b0b")
    fig.tight_layout(rect=(0, 0, 1, .95))
    fig.savefig(OUT / "hinh_FPS_c70_theo_N.png", dpi=200)
    print("saved", OUT)


if __name__ == "__main__":
    main()
