"""Split a Cloud detector ONNX at a cut point and measure it (plan 4/11, phase 1: cut points before the v4 build).

For one cut (a node whose outputs are the only tensors crossing it): head = images -> cut tensors (Edge), tail =
cut tensors + the other graph inputs -> original outputs (Cloud). Checks head + tail == full model, then measures what
the Edge would send per frame: values, zero share, uint8 / 4-bit size after zstd, and how much the tail's boxes move
when it gets the quantized tensor instead of the float one (agreement F1 vs the full model, IoU >= .5, score >= .5).

  python -B scripts/cut_points.py MODEL.onnx IMAGE_DIR [--cut rtdetr_layer2] [--n 20] [--out DIR]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import onnx
import onnxruntime as ort
import zstandard

# node names (RT-DETRv2 ONNX export) whose outputs are the cut; only layer2 is a single-tensor cut
CUTS = {
    "rtdetr_layer2": ["/model/backbone/res_layers.1/blocks.1/act/Relu"],
    "rtdetr_backbone": ["/model/backbone/res_layers.1/blocks.1/act/Relu", "/model/backbone/res_layers.2/blocks.1/act/Relu",
                        "/model/backbone/res_layers.3/blocks.1/act/Relu"],
    # D-FINE-X (HGNetv2-X): the two single-tensor cuts found by --profile (8 % and 28 % of the MACs on the Edge)
    "dfine_s1_down": ["/model/backbone/stages.1/downsample/conv/Conv"],
    "dfine_s1": ["/model/backbone/stages.1/blocks/blocks.1/Add"],
    # LW-DETR-large (ViT): after the patch embedding (1 %) and after ViT block 3 (19 %, last single-tensor cut)
    "lwdetr_patch": ["/backbone/backbone.0/encoder/patch_embed/proj/Conv"],
    "lwdetr_b2": ["/backbone/backbone.0/encoder/blocks.2/Add_1"],
}
IMAGENET = (np.array([.485, .456, .406], np.float32).reshape(1, 3, 1, 1), np.array([.229, .224, .225], np.float32).reshape(1, 3, 1, 1))


def profile(model_path, top=15):
    """Single-tensor cut candidates: points where only one image-derived tensor is alive, with its value count and
    the share of the model's conv/matmul MACs that would run on the Edge (before the cut)."""
    from onnx import numpy_helper
    m = onnx.shape_inference.infer_shapes(onnx.load(model_path))
    shape = {vi.name: [d.dim_value if d.dim_value > 0 else 1 for d in vi.type.tensor_type.shape.dim]
             for vi in list(m.graph.value_info) + list(m.graph.input) + list(m.graph.output)}
    W = {i.name: numpy_helper.to_array(i).shape for i in m.graph.initializer}
    nodes, image = list(m.graph.node), m.graph.input[0].name
    dep = {image}
    for n in nodes:
        if any(t in dep for t in n.input):
            dep.update(n.output)
    # static inference loses the spatial size behind dynamic Pad/Resize (D-FINE): take the real shapes from one run
    want = [n.output[0] for n in nodes if n.output and n.output[0] in dep and ("/backbone/" in n.name or "/encoder/" in n.name)]
    probe = onnx.load(model_path)
    probe.graph.output.extend(onnx.helper.make_empty_tensor_value_info(t) for t in want)
    sess = ort.InferenceSession(probe.SerializeToString(), providers=["CPUExecutionProvider"])
    feeds = {}
    for i in sess.get_inputs():
        dims = [d if isinstance(d, int) else 1 for d in i.shape]
        feeds[i.name] = (np.random.rand(*dims).astype(np.float32) if "float" in i.type
                         else np.array([[640, 640]], dtype=np.int64))
    names = [o.name for o in sess.get_outputs()]
    for t, v in zip(names, sess.run(names, feeds)):
        if t in want:
            shape[t] = list(v.shape)
    del sess
    last = {t: k for k, n in enumerate(nodes) for t in n.input if t in dep}
    for o in m.graph.output:
        last[o.name] = len(nodes)
    macs = []
    for n in nodes:
        mac = 0
        if n.op_type == "Conv" and n.input[1] in W and n.output[0] in shape:
            co, ci, kh, kw = W[n.input[1]]
            mac = co * ci * kh * kw * int(np.prod(shape[n.output[0]][2:]))
        elif n.op_type in ("MatMul", "Gemm") and n.output[0] in shape and n.input[0] in shape:
            b = W.get(n.input[1]) or shape.get(n.input[1])
            mac = int(np.prod(shape[n.input[0]])) * b[-1] if b else 0
        macs.append(mac)
    total, done, live, rows = sum(macs), 0, {image}, []
    for k, n in enumerate(nodes):
        done += macs[k]
        live = {t for t in live | {t for t in n.output if t in dep} if last.get(t, -1) > k}
        if len(live) == 1:
            (t,) = live
            rows.append((int(np.prod(shape.get(t, [0]))), done / total, n.name, shape.get(t)))
    best = {}
    for v, share, name, shp in rows:  # keep the first node for each distinct (size, share) step
        best.setdefault((v, round(share, 3)), (v, share, name, shp))
    print(f"{model_path}: {total / 1e9:.1f} GMAC, {len(rows)} single-tensor points")
    for v, share, name, shp in sorted(best.values(), key=lambda r: r[1])[:: max(1, len(best) // top) if top else 1][: top or None]:
        print(f"  Edge {share:5.0%}  send {v / 1e6:6.3f} M values  {shp}  after {name}")


def split(model_path, cut_nodes, out_dir):
    m = onnx.shape_inference.infer_shapes(onnx.load(model_path))  # the Extractor needs value_info for the cut tensors
    by_name = {n.name: n for n in m.graph.node}
    cut = [by_name[n].output[0] for n in cut_nodes]
    image = m.graph.input[0].name
    inits = {i.name for i in m.graph.initializer}  # old exports (LW-DETR) also list the weights as graph inputs
    others = [i.name for i in m.graph.input[1:] if i.name not in inits]
    outs = [o.name for o in m.graph.output]
    ex = onnx.utils.Extractor(m)
    head, tail = ex.extract_model([image], cut), ex.extract_model(cut + others, outs)
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx.save(head, out_dir / "head.onnx")
    onnx.save(tail, out_dir / "tail.onnx")
    return cut, image, others, outs


def _sample(src, n):
    """n frames spread over a folder of .jpg or a video file."""
    if Path(src).is_file():
        cap = cv2.VideoCapture(str(src))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for i in range(0, total, max(1, total // n))[:n]:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, bgr = cap.read()
            if ok:
                yield bgr
        return
    paths = sorted(Path(src).rglob("*.jpg"))
    for p in paths[:: max(1, len(paths) // n)][:n]:
        yield cv2.imread(str(p))


def frames(src, n, size):
    for bgr in _sample(src, n):
        h, w = bgr.shape[:2]
        x = cv2.cvtColor(cv2.resize(bgr, (size, size), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB).astype(np.float32) / 255
        yield x.transpose(2, 0, 1)[None], np.array([[w, h]], dtype=np.int64)


def quantize(t, bits):
    """Per-tensor affine quantization as the NPU emits it (uint8 + scale/offset); returns dequantized values and bytes."""
    lo, hi = float(t.min()), float(t.max())
    q = np.round((t - lo) / max(hi - lo, 1e-12) * (2 ** bits - 1)).astype(np.uint8)
    deq = q.astype(np.float32) / (2 ** bits - 1) * (hi - lo) + lo
    raw = q.tobytes() if bits == 8 else (q.ravel()[0::2] << 4 | np.append(q.ravel(), 0)[1::2][: (q.size + 1) // 2]).tobytes()
    return deq, raw


def detections(out, names, thr=.5):
    o = dict(zip(names, out))
    if "dets" in o:  # LW-DETR: logits + cxcywh, decoded as the Cloud does (inference/detection_decode.py)
        from inference.detection_decode import lwdetr_decode
        boxes = lwdetr_decode(o, thr, 300)
        return np.array([[b.left, b.top, b.right, b.bottom] for b in boxes]).reshape(-1, 4), np.array([b.class_id for b in boxes])
    keep = o["scores"][0] >= thr
    return o["boxes"][0][keep], o["labels"][0][keep]


def agreement(a, b):
    """F1 of boxes b against boxes a (same label, IoU >= .5, greedy)."""
    (ba, la), (bb, lb) = a, b
    if len(ba) == 0 and len(bb) == 0:
        return 1.0
    used, tp = set(), 0
    for i in range(len(bb)):
        best, bj = .5, None
        for j in range(len(ba)):
            if j in used or la[j] != lb[i]:
                continue
            x1, y1 = np.maximum(ba[j][:2], bb[i][:2])
            x2, y2 = np.minimum(ba[j][2:], bb[i][2:])
            inter = max(0, x2 - x1) * max(0, y2 - y1)
            iou = inter / ((ba[j][2] - ba[j][0]) * (ba[j][3] - ba[j][1]) + (bb[i][2] - bb[i][0]) * (bb[i][3] - bb[i][1]) - inter)
            if iou >= best:
                best, bj = iou, j
        if bj is not None:
            used.add(bj)
            tp += 1
    return 2 * tp / (len(ba) + len(bb))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("images", nargs="?")
    ap.add_argument("--cut", default="rtdetr_layer2", help=f"preset {sorted(CUTS)} or comma-separated node names")
    ap.add_argument("--profile", action="store_true", help="list single-tensor cut candidates and exit")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--out", default="results/cut_points")
    a = ap.parse_args()
    if a.profile:
        return profile(a.model)
    nodes = CUTS.get(a.cut) or a.cut.split(",")
    out_dir = Path(a.out) / (a.cut if a.cut in CUTS else "custom")
    cut, image, others, outs = split(a.model, nodes, out_dir)
    so = ort.SessionOptions()
    full, head, tail = (ort.InferenceSession(str(p), so, providers=["CPUExecutionProvider"])
                        for p in (a.model, out_dir / "head.onnx", out_dir / "tail.onnx"))
    size = full.get_inputs()[0].shape[-1]
    imagenet = "dets" in outs  # LW-DETR takes ImageNet-normalized input (inference/cloud_detr.py)
    extra = lambda sizes: {others[0]: sizes} if others else {}  # noqa: E731  (RT-DETR / D-FINE: orig_target_sizes)
    stats = {k: [] for k in ("values", "zeros", "f32_kb", "u8_zstd_kb", "u4_zstd_kb", "parity_px", "agree_f32", "agree_u8",
                             "agree_u4")}
    zc = zstandard.ZstdCompressor(level=3)
    for x, sizes in frames(a.images, a.n, size):
        if imagenet:
            x = (x - IMAGENET[0]) / IMAGENET[1]
        ref = full.run(None, {image: x} | extra(sizes))
        mids = head.run(None, {image: x})
        got = tail.run(None, dict(zip(cut, mids)) | extra(sizes))
        # element-wise diff is informational: models that sort queries by score (D-FINE, LW-DETR) reorder the
        # outputs on tiny float differences; the gate is the box set (agree_f32)
        stats["parity_px"].append(max(float(np.abs(r - g).max()) for r, g in zip(ref, got) if r.dtype.kind == "f"))
        stats["agree_f32"].append(agreement(detections(ref, outs), detections(got, outs)))
        t = np.concatenate([m.ravel() for m in mids])
        stats["values"].append(t.size)
        stats["zeros"].append(float((t == 0).mean()))
        stats["f32_kb"].append(t.size * 4 / 1024)
        for bits in (8, 4):
            deq, raw, parts = [], b"", []
            for mm in mids:
                d, r = quantize(mm, bits)
                parts.append(d)
                raw += r
            stats[f"u{bits}_zstd_kb"].append(len(zc.compress(raw)) / 1024)
            q_out = tail.run(None, dict(zip(cut, parts)) | extra(sizes))
            stats[f"agree_u{bits}"].append(agreement(detections(ref, outs), detections(q_out, outs)))
    assert min(stats["agree_f32"]) >= .99, f"head + tail differs from the full model: box agreement {min(stats['agree_f32'])}"
    print(f"cut {a.cut} -> {[c for c in cut]} | input {size}x{size} | {len(stats['values'])} frames")
    for k, v in stats.items():
        print(f"  {k:12s} mean {np.mean(v):10.3f}  min {np.min(v):10.3f}  max {np.max(v):10.3f}")


if __name__ == "__main__":
    main()
