"""UA-DETRAC sequences -> camera locations (30/10), so a simulated camera stays on one road like a real fixed camera.
The dataset does not label locations; sequences shot by the same fixed camera share their background, so: background
per sequence = per-pixel median of 15 frames (moving vehicles vanish), 96x54 grayscale, normalized; sequences join a
location by average linkage: two groups merge while their mean background correlation >= THRESHOLD (single linkage
chained different roads together; by eye, pairs >= .6 are the same place and .55 already mixes day/night scenes). Writes results/detrac/locations.json
{sequence: location id} and a contact sheet results/detrac/locations.jpg (one row per location) to check by eye.

  python -B scripts/detrac_locations.py [IMAGE_ROOT]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
IMAGES = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / "Downloads/DETRAC-Images/DETRAC-Images"
THRESHOLD = .60
W, H = 96, 54


def background(seq_dir):
    files = sorted(seq_dir.glob("img*.jpg"))
    pick = [files[int(i)] for i in np.linspace(0, len(files) - 1, 15)]
    stack = np.stack([cv2.resize(cv2.imread(str(f), cv2.IMREAD_GRAYSCALE), (W, H), interpolation=cv2.INTER_AREA)
                      for f in pick]).astype(np.float32)
    return np.median(stack, axis=0)


def groups(sim, threshold):
    """Average-linkage clusters: repeatedly merge the two groups with the highest mean pairwise similarity while it is
    >= threshold -> list of index lists."""
    cl = [[i] for i in range(len(sim))]
    while len(cl) > 1:
        best, pair = -1.0, None
        for a in range(len(cl)):
            for b in range(a + 1, len(cl)):
                m = float(sim[np.ix_(cl[a], cl[b])].mean())
                if m > best:
                    best, pair = m, (a, b)
        if best < threshold:
            break
        a, b = pair
        cl[a] = sorted(cl[a] + cl[b])
        del cl[b]
    return sorted(cl, key=lambda g: (-len(g), g))


def main():
    seqs = sorted(p.name for p in IMAGES.iterdir() if p.is_dir())
    bg = [background(IMAGES / s) for s in seqs]
    z = np.stack([(b - b.mean()) / (b.std() + 1e-6) for b in bg]).reshape(len(seqs), -1)
    sim = z @ z.T / z.shape[1]
    loc = groups(sim, THRESHOLD)
    out = {seqs[i]: k for k, g in enumerate(loc) for i in g}
    (ROOT / "results/detrac/locations.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    width = max(len(g) for g in loc)
    sheet = np.full((len(loc) * H, width * W), 255, np.uint8)
    for r, g in enumerate(loc):
        for c, i in enumerate(g):
            sheet[r * H:(r + 1) * H, c * W:(c + 1) * W] = np.clip(bg[i], 0, 255).astype(np.uint8)
    cv2.imwrite(str(ROOT / "results/detrac/locations.jpg"), sheet)
    sizes = sorted((len(g) for g in loc), reverse=True)
    print(f"{len(seqs)} sequences -> {len(loc)} locations; sizes {sizes}")
    off = sim[~np.eye(len(seqs), dtype=bool)]
    print(f"background correlation: within locations min pair {min(sim[i, j] for g in loc for i in g for j in g):.2f}, "
          f"all pairs 90th pct {np.percentile(off, 90):.2f}")


if __name__ == "__main__":
    main()
