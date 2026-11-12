"""Report figure from scripts/plot_by_n.py's so_lieu_theo_N.csv (no rerun): FPS c70 vs N, 7 methods (1/11 user
decision: 4 single models, the one hand rule for all situations, the probe-threshold heuristic, DRL).

  python -B scripts/plot_by_n_report.py DIR   (reads DIR/so_lieu_theo_N.csv, writes DIR/hinh_bao_cao_theo_N.png + CSV)
"""
import csv
import itertools
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

LINES = [  # column in the CSV, label, color (reference palette), dashed
    ("Tất cả YOLOv8n (Edge)", "Tất cả YOLOv8n (Edge)", "#eb6834", False),
    ("Tất cả YOLO11s (Edge)", "Tất cả YOLO11s (Edge)", "#1baf7a", False),
    ("Tất cả RT-DETRv2 (Cloud)", "Tất cả RT-DETRv2 (Cloud)", "#eda100", False),
    ("Tất cả D-FINE-X (Cloud)", "Tất cả D-FINE-X (Cloud)", "#8a8984", True),
    ("Luật tay chung (80 % lên RT-DETR)", "Luật tay (80 % camera lên RT-DETR)", "#008300", False),
    ("Heuristic ngưỡng probe", "Heuristic ngưỡng probe", "#e34948", False),
    ("DRL", "DRL", "#2a78d6", False),
]


def sign_flip_p(d):
    d = np.asarray(d, float)
    signs = np.array(list(itertools.product((1, -1), repeat=len(d))))
    return float(np.mean(np.abs((signs * d).mean(axis=1)) >= abs(d.mean()) - 1e-12))


def main(folder):
    folder = Path(folder)
    rows = list(csv.DictReader(open(folder / "so_lieu_theo_N.csv", encoding="utf-8-sig")))
    ns = sorted({int(r["N"]) for r in rows})
    val = {(col, n): np.array([float(r[col]) for r in rows if int(r["N"]) == n]) for col, *_ in LINES for n in ns}

    fig, ax = plt.subplots(figsize=(10, 6), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    for col, lab, color, dashed in LINES:
        y = [val[(col, n)].mean() for n in ns]
        ax.plot(ns, y, color=color, lw=3.2 if col == "DRL" else 2, ls="--" if dashed else "-", marker="o", ms=8,
                markeredgecolor="#fcfcfb", markeredgewidth=2, label=lab, zorder=3 if col == "DRL" else 2)
    ax.set_xticks(ns, [f"N={n}" for n in ns])
    ax.set_xlabel("Số luồng camera", color="#52514e")
    ax.set_ylabel("FPS c70 (tổng FPS của camera đạt F1 ≥ 0,7 và trễ ≤ 5 %)", color="#52514e")
    ax.grid(axis="y", color="#e4e3df", lw=.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=9, loc="upper left", bbox_to_anchor=(1.01, 1))
    ax.set_title("Tình huống chính: phần lớn video VN, OCR như testbed, 8 worker OCR (10 episode mỗi N)",
                 loc="left", fontsize=11, color="#0b0b0b")
    fig.tight_layout()
    fig.savefig(folder / "hinh_bao_cao_theo_N.png", dpi=200)

    with open(folder / "bang_bao_cao_theo_N.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["phuong_phap"] + [f"N={n}" for n in ns])
        for col, lab, *_ in LINES:
            w.writerow([lab] + [f"{val[(col, n)].mean():.1f} +- {val[(col, n)].std(ddof=1):.1f}" for n in ns])
        for col, lab in (("Luật tay chung (80 % lên RT-DETR)", "luật tay"), ("Heuristic ngưỡng probe", "heuristic")):
            w.writerow([f"DRL so với {lab}"] + [
                f"{(val[('DRL', n)].mean() / max(val[(col, n)].mean(), 1e-9) - 1) * 100:+.1f}% "
                f"(p={sign_flip_p(val[('DRL', n)] - val[(col, n)]):.3f})" for n in ns])
    sys.stdout.reconfigure(encoding="utf-8")
    print(open(folder / "bang_bao_cao_theo_N.csv", encoding="utf-8-sig").read())


if __name__ == "__main__":
    main(sys.argv[1])
