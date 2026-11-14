"""UC1 OCR in batches: export + parity gate (13/11).

  python -B scripts/plate_ocr_parity.py export --model-home ~/plate_ocr_home --out plate_ocr.onnx
  python -B scripts/plate_ocr_parity.py parity --model-home ~/plate_ocr_home --crops DIR [DIR ...]
         --backend torch | onnx:plate_ocr.onnx | triton[:plate_ocr]   [--conc 16]

parity: every crop read by EasyOCR itself (inference.plate_ocr.PlateReader, one line at a time) and by
inference.plate_ocr_batch.BatchPlateReader on the backend; gate: the same text for every crop. Then the throughput of
both (crops/s): EasyOCR one crop after another in one process, the batch reader with --conc crops at a time.
"""
import argparse
import glob
import os
import time
from concurrent.futures import ThreadPoolExecutor


def recognizer(model_home):
    from inference.plate_ocr import PlateReader
    ref = PlateReader(model_home, passes=3, gpu=True)
    net = ref.reader.recognizer
    return ref, getattr(net, "module", net)


def export(a):
    import torch
    _, net = recognizer(a.model_home)
    net = net.float().cpu().eval()

    class Logits(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, image):   # Model.forward, AdaptiveAvgPool2d((None, 1)) written as the mean it is (no ONNX op)
            f = self.m.FeatureExtraction(image).permute(0, 3, 1, 2).mean(dim=3)
            return self.m.Prediction(self.m.SequenceModeling(f).contiguous())

    torch.onnx.export(Logits(net), torch.zeros(2, 1, 64, 256), a.out, input_names=["image"], output_names=["logits"],
                      dynamic_axes={"image": {0: "n", 3: "w"}, "logits": {0: "n", 1: "t"}}, opset_version=17,
                      dynamo=False)
    print(f"{a.out}: {os.path.getsize(a.out) / 1e6:.1f} MB")


def backend(spec, net):
    import numpy as np
    if spec == "torch":
        import torch
        dev = next(net.parameters()).device

        def infer(x):
            with torch.no_grad():
                return net(torch.from_numpy(x).to(dev), None).float().cpu().numpy()
        return infer
    if spec.startswith("onnx:"):
        import onnxruntime as ort
        s = ort.InferenceSession(spec[5:], providers=ort.get_available_providers())
        return lambda x: s.run(["logits"], {"image": np.ascontiguousarray(x)})[0]
    if spec.startswith("triton"):
        from inference.triton_detr import TritonSession
        s = TritonSession(spec.partition(":")[2] or "plate_ocr", shm=False)
        return lambda x: s.run(["logits"], {"image": x})[0]
    raise SystemExit(f"backend? {spec}")


def parity(a):
    import cv2
    from inference.plate_ocr_batch import BatchPlateReader
    ref, net = recognizer(a.model_home)
    files = sorted(f for d in a.crops for f in glob.glob(os.path.join(d, "*.jpg")))
    crops = [(f, cv2.imread(f)) for f in files]
    crops = [(f, c) for f, c in crops if c is not None and min(c.shape[:2]) >= 4]
    batch = BatchPlateReader(backend(a.backend, net))
    t = time.perf_counter()
    want = [ref.read(c)[0] for _, c in crops]
    t_ref = time.perf_counter() - t
    batch.read(crops[0][1])   # warm-up (first Triton / ORT shapes)
    t = time.perf_counter()
    with ThreadPoolExecutor(a.conc) as pool:
        got = [r[0] for r in pool.map(lambda c: batch.read(c[1]), crops)]
    t_batch = time.perf_counter() - t
    diff = [(os.path.basename(f), w, g) for (f, _), w, g in zip(crops, want, got) if w != g]
    same = (len(crops) - len(diff)) / max(1, len(crops))
    # gate 99.4 %: EasyOCR itself on CPU vs GPU agrees on 521/524 crops (device rounding, 13/11)
    print(f"{len(crops)} crops: same text {len(crops) - len(diff)}/{len(crops)} ({same:.1%})"
          f" -> {'PASS' if same >= .994 else 'FAIL'} (gate 99.4 %)")
    for d in diff[:20]:
        print(f"   {d[0]}: EasyOCR {d[1]!r} batch {d[2]!r}")
    print(f"throughput: EasyOCR one by one {len(crops) / t_ref:.0f} crops/s; batch ({a.backend}, {a.conc} at a time) "
          f"{len(crops) / t_batch:.0f} crops/s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["export", "parity"])
    ap.add_argument("--model-home", required=True)
    ap.add_argument("--out", default="plate_ocr.onnx")
    ap.add_argument("--crops", nargs="+")
    ap.add_argument("--backend", default="torch")
    ap.add_argument("--conc", type=int, default=16)
    a = ap.parse_args()
    export(a) if a.cmd == "export" else parity(a)


if __name__ == "__main__":
    main()
