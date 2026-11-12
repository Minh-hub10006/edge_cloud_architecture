"""Pseudo-labels for VN-Hard: DEIMv2-X (outside the system's models) on every frame, full frame + overlapping tiles.

  python -B scripts/vn_pseudo_label.py VIDEO.mp4 OUT.jsonl [--deimv2 C:/Users/ADMIN/DEIMv2] [--tiles 2x3]
      [--every 1] [--hot-c 85 --cool-c 75]

One JSONL row per decoded frame {"frame", "boxes": [[coco_class, conf, x1, y1, x2, y2] normalized]}, COCO contiguous
classes car 2, motorcycle 3, bus 5, truck 7, conf >= 0.3 (the label threshold is chosen later). Resumable: frames
already in OUT are skipped. Tile boxes touching an inner tile edge are dropped (cut vehicles; the neighbour tile or the
full frame sees them whole), then per-class NMS at IoU 0.5. Pauses while the GPU is at or above --hot-c until --cool-c.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision.ops import batched_nms

VEHICLES = (2, 3, 5, 7)
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def load_model(repo, device):
    sys.path.insert(0, str(repo))
    from huggingface_hub import PyTorchModelHubMixin
    from engine.backbone import DINOv3STAs
    from engine.deim import HybridEncoder, DEIMTransformer
    from engine.deim.postprocessor import PostProcessor

    class DEIMv2(nn.Module, PyTorchModelHubMixin):  # same wrapper as the repo's hf_models.ipynb
        def __init__(self, config):
            super().__init__()
            self.backbone = DINOv3STAs(**config["DINOv3STAs"])
            self.encoder = HybridEncoder(**config["HybridEncoder"])
            self.decoder = DEIMTransformer(**config["DEIMTransformer"])
            self.postprocessor = PostProcessor(**config["PostProcessor"])

        def forward(self, x, sizes):
            return self.postprocessor(self.decoder(self.encoder(self.backbone(x))), sizes)

    return DEIMv2.from_pretrained(str(Path(repo) / "weight" / "DEIMv2_DINOv3_X_COCO")).eval().to(device)


def windows(w, h, cols, rows, overlap=.2):
    """Full frame first, then a cols x rows grid of tiles overlapping by `overlap` of a tile."""
    tw, th = int(w / (cols - (cols - 1) * overlap)), int(h / (rows - (rows - 1) * overlap))
    out = [(0, 0, w, h)]
    for r in range(rows):
        for c in range(cols):
            x0 = min(int(c * tw * (1 - overlap)), w - tw)
            y0 = min(int(r * th * (1 - overlap)), h - th)
            out.append((x0, y0, x0 + tw, y0 + th))
    return out


@torch.no_grad()
def detect(model, bgr, wins, device, conf=.3):
    h, w = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    crops = [cv2.resize(rgb[y0:y1, x0:x1], (640, 640), interpolation=cv2.INTER_LINEAR) for x0, y0, x1, y1 in wins]
    x = torch.from_numpy(np.stack(crops)).permute(0, 3, 1, 2).float().div(255)
    x = ((x - MEAN) / STD).to(device)
    sizes = torch.tensor([[x1 - x0, y1 - y0] for x0, y0, x1, y1 in wins], device=device)
    with torch.autocast("cuda", dtype=torch.float16, enabled=device.startswith("cuda")):
        outs = model(x, sizes)
    boxes, scores, labels = [], [], []
    for k, ((x0, y0, x1, y1), o) in enumerate(zip(wins, outs)):
        keep = o["scores"] >= conf
        keep &= torch.isin(o["labels"], torch.tensor(VEHICLES, device=o["labels"].device))
        b, s, lab = o["boxes"][keep].float(), o["scores"][keep].float(), o["labels"][keep]
        if k:  # tile: drop boxes cut by an inner tile edge (2 px margin)
            m = 2
            inner = ((b[:, 0] <= m) & (x0 > 0)) | ((b[:, 1] <= m) & (y0 > 0)) \
                | ((b[:, 2] >= (x1 - x0) - m) & (x1 < w)) | ((b[:, 3] >= (y1 - y0) - m) & (y1 < h))
            b, s, lab = b[~inner], s[~inner], lab[~inner]
        boxes.append(b + torch.tensor([x0, y0, x0, y0], device=b.device, dtype=b.dtype))
        scores.append(s)
        labels.append(lab)
    b, s, lab = torch.cat(boxes), torch.cat(scores), torch.cat(labels)
    keep = batched_nms(b, s, lab, .5)
    b = (b[keep] / torch.tensor([w, h, w, h], device=b.device)).clamp(0, 1).cpu().numpy()
    return [[int(c), round(float(p), 4), *[round(float(v), 5) for v in bb]]
            for c, p, bb in zip(lab[keep].cpu().numpy(), s[keep].cpu().numpy(), b)]


def gpu_temp():
    try:
        return int(subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader"],
                                  capture_output=True, text=True, timeout=10).stdout.split()[0])
    except Exception:
        return None  # no reading: keep going (nvidia-smi missing is not a heat signal)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("video")
    p.add_argument("out")
    p.add_argument("--deimv2", default="C:/Users/ADMIN/DEIMv2")
    p.add_argument("--tiles", default="2x3", help="COLSxROWS; 1x1 = full frame only")
    p.add_argument("--every", type=int, default=1, help="label every N-th frame")
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--hot-c", type=int, default=85)
    p.add_argument("--cool-c", type=int, default=75)
    a = p.parse_args(argv)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(a.deimv2, device)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(x)["frame"] for x in out.read_text(encoding="utf-8").splitlines()} if out.exists() else set()
    cols, rows = map(int, a.tiles.lower().split("x"))
    cap = cv2.VideoCapture(a.video)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    wins, i, t0, new = None, -1, time.time(), 0
    with out.open("a", encoding="utf-8") as sink:
        while a.max_frames is None or i + 1 < a.max_frames:
            ok, bgr = cap.read()
            if not ok:
                break
            i += 1
            if i % a.every or i in done:
                continue
            if wins is None:
                wins = windows(bgr.shape[1], bgr.shape[0], cols, rows) if cols * rows > 1 else [(0, 0, bgr.shape[1], bgr.shape[0])]
            sink.write(json.dumps({"frame": i, "boxes": detect(model, bgr, wins, device)}) + "\n")
            new += 1
            if new % 50 == 0:
                sink.flush()
                t = gpu_temp()
                print(f"{Path(a.video).name} {i + 1}/{n} {new / (time.time() - t0):.2f} frame/s GPU {t} C", flush=True)
                if t is not None and t >= a.hot_c:
                    while (t := gpu_temp()) is not None and t > a.cool_c:
                        print(f"GPU {t} C >= {a.hot_c}: pause until {a.cool_c}", flush=True)
                        time.sleep(30)
    print(f"DONE {a.video} -> {out} ({new} new frames)")


if __name__ == "__main__":
    main()
