"""Figures for the fair comparison (round 29, 31/10) from `fair_eval.py --dump FILE`.

  python scripts/plot_fair_eval.py results/fair/r29_rows.json OUT_DIR
fig1: median change vs the train-chosen baseline over the 16 situations, with significant wins/losses;
fig2: mean constrained FPS (c70) per situation group, every method; fig3: change per situation (heatmap)."""
import itertools
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
import numpy as np  # noqa: E402

FAMILY = {"Model lẻ": "#2a78d6", "Luật tay / heuristic": "#eb6834", "DRL": "#eda100"}
METHODS = [  # (key, label, family); "base" keys come from the baseline dict, the rest from the planner dict
    ("all_v8n", "Tất cả YOLOv8n (Edge)", "Model lẻ"), ("all_y11s", "Tất cả YOLO11s (Edge)", "Model lẻ"),
    ("allrt", "Tất cả RT-DETRv2 (Cloud)", "Model lẻ"), ("all_dx", "Tất cả D-FINE-X (Cloud)", "Model lẻ"),
    ("static", "Luật tay: k camera lên Cloud", "Luật tay / heuristic"),
    ("thresh", "Heuristic ngưỡng probe", "Luật tay / heuristic"),
    ("net", "Heuristic theo mạng", "Luật tay / heuristic"),
    ("drl", "DRL", "DRL"),
]  # 1/11: the planner and DRL + lookahead are left out of the report figures (user decision)
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def sign_flip_p(d):
    d = np.asarray(d, float)
    signs = np.array(list(itertools.product((1, -1), repeat=len(d))))
    return float(np.mean(np.abs((signs * d).mean(axis=1)) >= abs(d.mean()) - 1e-12))


def load(path):
    data = json.load(open(path))
    chosen = {tuple(map(str, k)): b for k, b, _ in data["chosen"]}
    sit = {}
    for k, seed, base, plan, orc in data["rows"]:
        key = tuple(map(str, k))
        vals = {m: (base[m] if m in base else plan.get(m)) for m, _, _ in METHODS}
        vals["_base"], vals["_true"] = base[chosen[key]], plan.get("true")
        sit.setdefault(key, []).append(vals)
    return sit, {key: chosen[key] for key in sit}


def describe(key):
    d, dyn, n = key
    reads, workers, vn = json.loads(d.replace("(", "[").replace(")", "]")) if d.startswith(("[", "(")) else (0, 0, 0)
    return (f"{n} cam · {'phần lớn' if vn > .5 else 'nửa'} VN · OCR {'nặng' if reads > .5 else 'thường'} "
            f"{workers}w · {'biến động' if dyn == 'all' else 'ổn định'}"), n, vn > .5


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)


def main(path, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    sit, chosen = load(path)
    methods = [m for m in METHODS if all(r[m[0]] is not None for v in sit.values() for r in v)]
    keys = sorted(sit, key=lambda k: (describe(k)[2], int(k[2]), k[0], k[1]))
    rel = {m[0]: [] for m in methods}
    win = {m[0]: [0, 0] for m in methods}
    for k in keys:
        base = np.array([r["_base"] for r in sit[k]])
        for m, _, _ in methods:
            v = np.array([r[m] for r in sit[k]])
            r_ = (v.mean() / max(base.mean(), 1e-9) - 1) * 100
            rel[m].append(r_)
            if sign_flip_p(v - base) < .05 and r_ != 0:
                win[m][0 if r_ > 0 else 1] += 1

    # fig1: median change, one bar per method, families colored, wins/losses written beside
    fig, ax = plt.subplots(figsize=(9, 6.2), facecolor=SURFACE)
    style(ax)
    y = np.arange(len(methods))[::-1]
    med = [np.median(rel[m]) for m, _, _ in methods]
    ax.barh(y, med, color=[FAMILY[f] for _, _, f in methods], height=.62, edgecolor=SURFACE, linewidth=2)
    ax.axvline(0, color=MUTED, lw=1)
    for yi, mv, (m, _, _) in zip(y, med, methods):
        ax.text(max(mv, 0) + 1.5 if mv > -60 else 1.5, yi, f"{mv:+.1f} %   thắng rõ {win[m][0]} · thua rõ {win[m][1]}",
                va="center", fontsize=8.5, color=INK)
    ax.set_yticks(y, [lab for _, lab, _ in methods], fontsize=9, color=INK)
    ax.set_xlabel("Trung vị thay đổi so với baseline chọn trên train (16 tình huống, FPS c70)", color=MUTED, fontsize=9)
    ax.grid(axis="x", color=GRID, lw=.6)
    ax.set_axisbelow(True)
    ax.set_xlim(min(min(med) - 5, -10), max(med) + 75)
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c) for c in FAMILY.values()], labels=list(FAMILY),
              loc="upper center", bbox_to_anchor=(.5, -.09), ncol=3, frameon=False, fontsize=8.5)
    ax.set_title("So sánh công bằng: mỗi phương pháp so với baseline tốt nhất chọn trên train", loc="left",
                 fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(out / "fig1_tong_quan.png", dpi=200)

    # fig2: absolute c70 per group (10/14 cameras x half/mostly VN), small multiples, shared x
    groups = [(n, vn) for vn in (False, True) for n in ("10", "14")]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), sharex=True, facecolor=SURFACE)
    xmax = 0
    for ax, (n, vn) in zip(axes.flat, groups):
        style(ax)
        ks = [k for k in keys if k[2] == n and describe(k)[2] == vn]
        vals = [np.mean([r[m] for k in ks for r in sit[k]]) for m, _, _ in methods]
        xmax = max(xmax, max(vals))
        ax.barh(y, vals, color=[FAMILY[f] for _, _, f in methods], height=.62, edgecolor=SURFACE, linewidth=2)
        for yi, v in zip(y, vals):
            ax.text(v + 1, yi, f"{v:.0f}", va="center", fontsize=8, color=INK)
        ax.set_yticks(y, [lab for _, lab, _ in methods], fontsize=8.5, color=INK)
        ax.set_title(f"{n} camera · {'phần lớn' if vn else 'khoảng nửa'} VN ({len(ks)} tình huống × 10 ep)",
                     loc="left", fontsize=10, color=INK)
        ax.grid(axis="x", color=GRID, lw=.6)
        ax.set_axisbelow(True)
    fig.supxlabel("FPS c70 trung bình (tổng FPS của các camera đạt F1 ≥ 0,7 và trễ ≤ 5 %)", color=MUTED, fontsize=9)
    axes[0, 0].set_xlim(0, xmax * 1.12)
    fig.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c) for c in FAMILY.values()], labels=list(FAMILY),
               loc="upper right", ncol=3, frameon=False, fontsize=9)
    fig.suptitle("FPS c70 theo nhóm tình huống", x=.01, ha="left", fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0, 1, .96))
    fig.savefig(out / "fig2_theo_nhom.png", dpi=200)

    # fig3: change per situation, diverging blue (worse) - neutral - orange (better), values written in each cell
    m_ = np.array([rel[m] for m, _, _ in methods])
    lim = max(20, np.percentile(np.abs(m_[np.isfinite(m_)]), 95))
    cmap = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#f0efec", "#eb6834"])
    fig, ax = plt.subplots(figsize=(15, 6.5), facecolor=SURFACE)
    ax.imshow(np.clip(m_, -lim, lim), cmap=cmap, vmin=-lim, vmax=lim, aspect="auto")
    for i in range(m_.shape[0]):
        for j in range(m_.shape[1]):
            ax.text(j, i, f"{m_[i, j]:+.0f}", ha="center", va="center", fontsize=7.5, color=INK)
    ax.set_yticks(range(len(methods)), [lab for _, lab, _ in methods], fontsize=8.5)
    ax.set_xticks(range(len(keys)), [describe(k)[0] + f"\n(base: {chosen[k]})" for k in keys], rotation=60,
                  ha="right", fontsize=7.5)
    ax.set_title("Thay đổi (%) so với baseline chọn trên train, từng tình huống (cam = tốt hơn, xanh = kém hơn)",
                 loc="left", fontsize=11, color=INK)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "fig3_tung_tinh_huong.png", dpi=200)

    with open(out / "bang_so_lieu.csv", "w", encoding="utf-8-sig") as f:  # table view of every number in the figures
        f.write("tinh_huong,baseline_train," + ",".join(lab for _, lab, _ in methods) + "\n")
        for j, k in enumerate(keys):
            f.write(f"\"{describe(k)[0]}\",{chosen[k]}," + ",".join(f"{rel[m][j]:+.1f}%" for m, _, _ in methods) + "\n")
    print("saved", *sorted(p.name for p in out.iterdir()))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
