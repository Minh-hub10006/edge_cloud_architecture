"""Feature compression for the cut detector (plan 4/11, phase 1): send the cut tensor as an image instead of zstd.

The 128 x 80 x 80 tensor after RT-DETRv2 layer2 is quantized to 8 bit (one range for the whole tensor, or one per
channel = per map), tiled into one 1280 x 640 grayscale mosaic (16 x 8 maps) and coded with PNG / JPEG / WebP.
Reported per codec: KB per frame and box agreement of the tail's output with the float split (= full model).

  python -B scripts/feature_codec.py HEAD.onnx TAIL.onnx SOURCE [--n 20]
"""
import argparse

import cv2
import numpy as np
import onnxruntime as ort
import zstandard

from cut_points import agreement, detections, frames

CODECS = [("png", ".png", []), ("jpeg95", ".jpg", [cv2.IMWRITE_JPEG_QUALITY, 95]),
          ("jpeg85", ".jpg", [cv2.IMWRITE_JPEG_QUALITY, 85]), ("jpeg70", ".jpg", [cv2.IMWRITE_JPEG_QUALITY, 70]),
          ("jpeg50", ".jpg", [cv2.IMWRITE_JPEG_QUALITY, 50]), ("webp90", ".webp", [cv2.IMWRITE_WEBP_QUALITY, 90]),
          ("webp70", ".webp", [cv2.IMWRITE_WEBP_QUALITY, 70])]
COLS = 16


def to_u8(t, per_channel):
    """t [C,H,W] -> uint8 [C,H,W] + (lo, hi) per channel (or one pair broadcast)."""
    axes = (1, 2) if per_channel else None
    lo = t.min(axis=axes, keepdims=True)
    hi = t.max(axis=axes, keepdims=True)
    q = np.round((t - lo) / np.maximum(hi - lo, 1e-12) * 255).astype(np.uint8)
    return q, lo, hi


def mosaic(q):
    c, h, w = q.shape
    rows = c // COLS
    return q.reshape(rows, COLS, h, w).transpose(0, 2, 1, 3).reshape(rows * h, COLS * w)


def unmosaic(m, c, h, w):
    rows = c // COLS
    return m.reshape(rows, h, COLS, w).transpose(0, 2, 1, 3).reshape(c, h, w)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("head")
    ap.add_argument("tail")
    ap.add_argument("source")
    ap.add_argument("--n", type=int, default=20)
    a = ap.parse_args()
    so = {"providers": ["CPUExecutionProvider"]}
    head, tail = ort.InferenceSession(a.head, **so), ort.InferenceSession(a.tail, **so)
    cut = head.get_outputs()[0].name
    outs = [o.name for o in tail.get_outputs()]
    others = [i.name for i in tail.get_inputs() if i.name != cut]
    zc = zstandard.ZstdCompressor(level=3)
    res = {}
    for x, sizes in frames(a.source, a.n, 640):
        mid = head.run(None, {"images": x})[0]
        feeds = {o: sizes for o in others}
        ref = detections(tail.run(None, {cut: mid} | feeds), outs)
        t = mid[0]
        c, h, w = t.shape
        for per in (False, True):
            q, lo, hi = to_u8(t, per)
            side = 0 if not per else 2 * 4 * c  # per-channel ranges travel too (float32 pairs)
            cases = [("zstd", len(zc.compress(q.tobytes())), q)]
            m = mosaic(q)
            for name, ext, params in CODECS:
                ok, buf = cv2.imencode(ext, m, params)
                assert ok
                dec = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
                cases.append((name, len(buf), unmosaic(dec, c, h, w)))
            for name, nbytes, qq in cases:
                deq = (qq.astype(np.float32) / 255 * (hi - lo) + lo)[None]
                got = detections(tail.run(None, {cut: deq.astype(np.float32)} | feeds), outs)
                k = f"{'kenh' if per else 'chung'}-{name}"
                res.setdefault(k, []).append(((nbytes + side) / 1024, agreement(ref, got)))
    print(f"{a.source} | {a.n} frames | KB per frame (mean) | box agreement (mean / min)")
    for k, v in res.items():
        kb, ag = np.array(v).T
        print(f"  {k:16s} {kb.mean():7.1f} KB   {ag.mean():.3f} / {ag.min():.3f}")


if __name__ == "__main__":
    main()
