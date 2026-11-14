"""Path B without training (14/11): which compression of the cut tensor gets to JPEG's bandwidth at an acceptable F1?

  python -B scripts/compress_survey.py MANIFEST.json FULL_640.onnx [--out DIR]
  (MANIFEST = the G3 frames: 200 UA-DETRAC test frames, 5 per sequence, with image + annotation paths)

RT-DETRv2-S is cut after ResNet layer2 (128 x 80 x 80, float head on the laptop). For every variant: bytes per frame,
encode time on this CPU, and micro F1 against the UA-DETRAC labels exactly as gate G3 scores (vehicles, score >= .6,
IoU .5, ignored regions dropped). References: the full model on the raw 640 frame, and path C (JPEG of the frame).
Gate for B as its own level: <= ~76 KB per frame (path C's JPEG) and F1 >= ~0.83 (above A 0.78, at most ~0.04 under C).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import zstandard

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cut_points import CUTS, split  # noqa: E402
from detrac import match_frame, parse_annotation  # noqa: E402

VEHICLES, THR, S, COLS = {2, 5, 7}, .6, 640, 16


# ---------------------------------------------------------------- codecs: t [C,H,W] float -> bytes -> t' [C,H,W]
def quant(t, bits, per_channel):
    axes = (1, 2) if per_channel else None
    lo, hi = t.min(axis=axes, keepdims=True), t.max(axis=axes, keepdims=True)
    q = np.round((t - lo) / np.maximum(hi - lo, 1e-12) * (2 ** bits - 1)).astype(np.uint8)
    return q, lo.astype(np.float32), hi.astype(np.float32)


def dequant(q, lo, hi, bits):
    return q.astype(np.float32) / (2 ** bits - 1) * (hi - lo) + lo


def pack_bits(q, bits):
    """uint8 values < 2**bits -> dense bit stream (the wire form)."""
    return np.packbits(np.unpackbits(q.reshape(-1, 1), axis=1)[:, 8 - bits:].reshape(-1)).tobytes()


def zstd_codec(bits, per_channel, level=3):
    def run(t):
        q, lo, hi = quant(t, bits, per_channel)
        blob = zstandard.ZstdCompressor(level=level).compress(pack_bits(q, bits)) + lo.tobytes() + hi.tobytes()
        return blob, dequant(q, lo, hi, bits)
    return run


def mosaic(q):
    c, h, w = q.shape
    return q.reshape(c // COLS, COLS, h, w).transpose(0, 2, 1, 3).reshape(c // COLS * h, COLS * w)


def unmosaic(m, c, h, w):
    return m.reshape(c // COLS, h, COLS, w).transpose(0, 2, 1, 3).reshape(c, h, w)


def image_codec(ext, params, per_channel):
    def run(t):
        q, lo, hi = quant(t, 8, per_channel)
        ok, buf = cv2.imencode(ext, mosaic(q), params)
        back = unmosaic(cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE), *t.shape)
        return buf.tobytes() + lo.tobytes() + hi.tobytes(), dequant(back, lo, hi, 8)
    return run


def pooled(inner, k=2):
    """Average k x k on the Edge, nearest upsampling on the Cloud (the tail expects 80 x 80)."""
    def run(t):
        c, h, w = t.shape
        small = t.reshape(c, h // k, k, w // k, k).mean(axis=(2, 4))
        blob, back = inner(small)
        return blob, np.repeat(np.repeat(back, k, axis=1), k, axis=2)
    return run


def top_channels(inner, keep):
    """Send only the `keep` channels with the most energy in this frame (+ their indexes); the rest arrive as 0."""
    def run(t):
        idx = np.sort(np.argsort(-np.abs(t).mean(axis=(1, 2)))[:keep])
        blob, back = inner(t[idx])
        full = np.zeros_like(t)
        full[idx] = back
        return blob + idx.astype(np.uint8).tobytes(), full
    return run


def nonuniform2(level=3):
    """2 bits per value, per channel: code 0 = exactly 0 (ReLU: ~40 % zeros), codes 1-3 = the means of the positive
    values in that channel's lower / middle / upper thirds (3 floats per channel on the wire)."""
    def run(t):
        c = t.shape[0]
        flat = t.reshape(c, -1)
        cent = np.zeros((c, 4), np.float32)
        code = np.zeros(flat.shape, np.uint8)
        for i in range(c):
            pos = flat[i][flat[i] > 0]
            if pos.size:
                e = np.quantile(pos, [1 / 3, 2 / 3])
                parts = [pos[pos <= e[0]], pos[(pos > e[0]) & (pos <= e[1])], pos[pos > e[1]]]
                cent[i, 1:] = [x.mean() if x.size else 0 for x in parts]
                nz = flat[i] > 0
                code[i, nz] = 1 + np.searchsorted(e, flat[i][nz])
        blob = zstandard.ZstdCompressor(level=level).compress(pack_bits(code, 2)) + cent[:, 1:].tobytes()
        return blob, np.take_along_axis(cent, code.astype(np.int64), axis=1).reshape(t.shape)
    return run


def pooled_bilinear(inner, kh=2, kw=2):
    """Average kh x kw on the Edge, bilinear upsampling on the Cloud."""
    def run(t):
        c, h, w = t.shape
        small = t.reshape(c, h // kh, kh, w // kw, kw).mean(axis=(2, 4))
        blob, back = inner(small)
        up = np.stack([cv2.resize(ch, (w, h), interpolation=cv2.INTER_LINEAR) for ch in back])
        return blob, up
    return run


def variants_round2():
    v = {}
    v["u2 per-channel zstd19"] = zstd_codec(2, True, level=19)
    v["nonuniform2 per-channel zstd"] = nonuniform2()
    v["nonuniform2 per-channel zstd19"] = nonuniform2(19)
    for bits in (4, 3, 2):
        v[f"pool2 bilinear u{bits} per-channel zstd"] = pooled_bilinear(zstd_codec(bits, True))
    v["pool2 bilinear nonuniform2"] = pooled_bilinear(nonuniform2())
    for bits in (3, 2):
        v[f"pool 1x2 bilinear u{bits} per-channel zstd"] = pooled_bilinear(zstd_codec(bits, True), 1, 2)
    v["pool 1x2 bilinear nonuniform2"] = pooled_bilinear(nonuniform2(), 1, 2)
    return v


def variants():
    v = {"u8 zstd (now)": zstd_codec(8, False)}
    for bits in (6, 5, 4, 3, 2):
        v[f"u{bits} zstd"] = zstd_codec(bits, False)
        v[f"u{bits} per-channel zstd"] = zstd_codec(bits, True)
    for bits in (8, 6, 4, 3):
        v[f"pool2 u{bits} per-channel zstd"] = pooled(zstd_codec(bits, True))
    for keep in (64, 32):
        for bits in (4, 3):
            v[f"top{keep}ch u{bits} per-channel zstd"] = top_channels(zstd_codec(bits, True), keep)
    for q in (90, 70, 50, 30):
        v[f"jpeg{q} mosaic per-channel"] = image_codec(".jpg", [cv2.IMWRITE_JPEG_QUALITY, q], True)
        v[f"webp{q} mosaic per-channel"] = image_codec(".webp", [cv2.IMWRITE_WEBP_QUALITY, q], True)
    for q in (90, 70, 50):
        v[f"pool2 webp{q} per-channel"] = pooled(image_codec(".webp", [cv2.IMWRITE_WEBP_QUALITY, q], True))
    try:   # only if this OpenCV was built with AVIF
        cv2.imencode(".avif", np.zeros((16, 16), np.uint8))
        for q in (80, 60, 40):
            v[f"avif{q} mosaic per-channel"] = image_codec(".avif", [cv2.IMWRITE_AVIF_QUALITY, q], True)
    except (cv2.error, AttributeError):
        pass
    return v


# ---------------------------------------------------------------- scoring
def vehicles(out, names):
    o = dict(zip(names, out))
    s, b, lab = o["scores"][0], o["boxes"][0], o["labels"][0]
    return [(float(c), tuple(map(float, bb))) for c, bb, l in zip(s, b, lab) if c >= THR and int(l) in VEHICLES]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest")
    ap.add_argument("full")
    ap.add_argument("--out", default="results/compress_survey")
    ap.add_argument("--round", type=int, default=1, help="2: the second set of variants (14/11)")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if not (out / "rtdetr_layer2" / "tail.onnx").exists():
        split(a.full, CUTS["rtdetr_layer2"], out / "rtdetr_layer2")
    so = {"providers": ["CPUExecutionProvider"]}
    full = ort.InferenceSession(a.full, **so)
    head = ort.InferenceSession(str(out / "rtdetr_layer2" / "head.onnx"), **so)
    tail = ort.InferenceSession(str(out / "rtdetr_layer2" / "tail.onnx"), **so)
    names, cut = [o.name for o in full.get_outputs()], head.get_outputs()[0].name
    rows = json.loads(Path(a.manifest).read_text(encoding="utf-8"))
    codecs = variants() if a.round == 1 else {"u8 zstd (now)": zstd_codec(8, False),
                                                "u2 per-channel zstd": zstd_codec(2, True), **variants_round2()}
    tot = {k: [0, 0, 0] for k in ["full model (raw frame)", "C jpeg85 frame", "C jpeg70 frame", "C jpeg50 frame",
                                   *codecs]}
    size = {k: [] for k in tot}
    enc_ms = {k: [] for k in tot}
    anns = {}
    for i, r in enumerate(rows):
        if r["annotation"] not in anns:
            anns[r["annotation"]] = parse_annotation(r["annotation"])
        ann = anns[r["annotation"]]
        gt = ann["frames"][r["frame"]]
        bgr = cv2.resize(cv2.imread(r["image"]), (S, S), interpolation=cv2.INTER_AREA)
        sizes = np.array([[r["w"], r["h"]]], dtype=np.int64)
        to_x = lambda im: np.ascontiguousarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.float32).transpose(2, 0, 1)[None] / 255)
        preds = {"full model (raw frame)": full.run(None, {"images": to_x(bgr), "orig_target_sizes": sizes})}
        for q in (85, 70, 50):
            t0 = time.perf_counter()
            ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, q])
            enc_ms[f"C jpeg{q} frame"].append((time.perf_counter() - t0) * 1e3)
            size[f"C jpeg{q} frame"].append(len(buf))
            preds[f"C jpeg{q} frame"] = full.run(None, {"images": to_x(cv2.imdecode(buf, cv2.IMREAD_COLOR)),
                                                         "orig_target_sizes": sizes})
        mid = head.run(None, {"images": to_x(bgr)})[0][0]
        for k, codec in codecs.items():
            t0 = time.perf_counter()
            blob, back = codec(mid)
            enc_ms[k].append((time.perf_counter() - t0) * 1e3)
            size[k].append(len(blob))
            preds[k] = tail.run(None, {cut: back[None].astype(np.float32), "orig_target_sizes": sizes})
        for k, p in preds.items():
            tp, fp, fn = match_frame(gt, vehicles(p, names), ann["ignored"])
            t = tot[k]
            t[0] += tp; t[1] += fp; t[2] += fn  # noqa: E702
        if i % 40 == 39:
            print(f"{i + 1}/{len(rows)} frames", flush=True)
    res = []
    for k, (tp, fp, fn) in tot.items():
        kb = np.mean(size[k]) / 1024 if size[k] else 0.0
        res.append({"variant": k, "kb": round(kb, 1), "f1": round(2 * tp / max(2 * tp + fp + fn, 1), 4),
                    "encode_ms": round(float(np.mean(enc_ms[k])), 2) if enc_ms[k] else None})
    res.sort(key=lambda r: -r["kb"])
    (out / f"survey_round{a.round}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"{len(rows)} frames, micro F1 (vehicles, score >= {THR}, IoU .5)")
    print(f"{'variant':36s} {'KB':>7s} {'F1':>7s} {'enc ms':>7s}")
    for r in res:
        print(f"{r['variant']:36s} {r['kb']:7.1f} {r['f1']:7.4f} {r['encode_ms'] if r['encode_ms'] is not None else '':>7}")


if __name__ == "__main__":
    main()
