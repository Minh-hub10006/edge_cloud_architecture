"""UC3 model choice (P2, 10/11): DM-Count (MIT) density maps, several trained weights x several input sizes.

  python -B scripts/crowd_eval.py eval  WEIGHTS_DIR SHB_TEST_DIR [--limit 316]   (images/, ground_truth/)
  python -B scripts/crowd_eval.py show  WEIGHT.pth IMAGE [IMAGE ...] --out DIR [--long 1280]
  python -B scripts/crowd_eval.py onnx  WEIGHT.pth OUT.onnx [--size 1280]

eval: ShanghaiTech part B test (images/IMG_n.jpg + ground-truth/GT_IMG_n.mat, one point per head) -> MAE / RMSE of the
      count for every *.pth in WEIGHTS_DIR and every input variant:
        native  the image as is (the paper's protocol; our numbers must be close to the published ones)
        long N  resized so its long side is N (aspect kept): what the Edge would send
        sq N    long side N, padded to N x N (fixed size = what Triton batches)
show: overlays the density map and the count on images (ShanghaiTech or our camera frames) -> PNGs to look at.
onnx: exports the network with a fixed N x N input, checks ONNX Runtime against PyTorch on random input.
Preprocessing as DM-Count's datasets/crowd.py: RGB, /255, ImageNet mean/std; the output map is 1/8 of the input.
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

def _device():
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


DEVICE = _device()
MEAN, STD = np.array([0.485, 0.456, 0.406], np.float32), np.array([0.229, 0.224, 0.225], np.float32)


def network():
    import torch.nn as nn
    from torch.nn import functional as F
    cfg = [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512]
    layers, c = [], 3
    for v in cfg:
        if v == "M":
            layers.append(nn.MaxPool2d(2, 2))
        else:
            layers += [nn.Conv2d(c, v, 3, padding=1), nn.ReLU(inplace=True)]
            c = v

    class VGG(nn.Module):   # DM-Count models.py, forward returns the density map only
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(*layers)
            self.reg_layer = nn.Sequential(nn.Conv2d(512, 256, 3, padding=1), nn.ReLU(inplace=True),
                                           nn.Conv2d(256, 128, 3, padding=1), nn.ReLU(inplace=True))
            self.density_layer = nn.Sequential(nn.Conv2d(128, 1, 1), nn.ReLU())

        def forward(self, x):
            x = F.interpolate(self.features(x), scale_factor=2, mode="bilinear", align_corners=True)
            return self.density_layer(self.reg_layer(x))
    return VGG()


def load(path):
    import torch
    net = network()
    state = torch.load(path, map_location="cpu")
    net.load_state_dict(state.get("model_state_dict", state) if isinstance(state, dict) else state)
    return net.eval().to(DEVICE)


def prepare(bgr, variant):
    """-> (NCHW float32 tensor, scale of the kept image). variant: native | long N | sq N."""
    h, w = bgr.shape[:2]
    kind, n = (variant.split() + [None])[:2]
    img = bgr
    if kind in ("long", "sq"):
        n = int(n)
        s = n / max(h, w)
        img = cv2.resize(bgr, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    x = (cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255 - MEAN) / STD
    if kind == "sq":
        canvas = np.zeros((n, n, 3), np.float32)   # 0 after normalization = the mean colour
        canvas[:x.shape[0], :x.shape[1]] = x
        x = canvas
    else:   # the network divides by 16: pad to a multiple (zeros), as above
        ph, pw = -x.shape[0] % 16, -x.shape[1] % 16
        x = np.pad(x, ((0, ph), (0, pw), (0, 0)))
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None])


def density(net, x):
    import torch
    with torch.no_grad():
        return net(torch.from_numpy(x).to(DEVICE))[0, 0].cpu().numpy()


def gt_count(mat_path):
    from scipy.io import loadmat
    info = loadmat(mat_path)["image_info"]
    return len(info[0, 0][0, 0][0])


def evaluate(weights_dir, test_dir, limit, variants):
    images = sorted(Path(test_dir, "images").glob("*.jpg"))[:limit]
    print(f"{len(images)} ShanghaiTech B test images")
    for w in sorted(Path(weights_dir).rglob("*.pth")):
        net = load(w)
        errs = {v: [] for v in variants}
        for img_path in images:
            gt = gt_count(Path(test_dir, "ground_truth", f"GT_{img_path.stem}.mat"))
            bgr = cv2.imread(str(img_path))
            for v in variants:
                errs[v].append(density(net, prepare(bgr, v)).sum() - gt)
        for v in variants:
            e = np.array(errs[v])
            print(f"  {w.parent.name}/{w.name:28s} {v:10s} MAE {np.abs(e).mean():6.2f}  RMSE {np.sqrt((e ** 2).mean()):6.2f}"
                  f"  bias {e.mean():+6.2f}", flush=True)


def show(weight, images, out, variant):
    net = load(weight)
    Path(out).mkdir(parents=True, exist_ok=True)
    for p in images:
        bgr = cv2.imread(str(p))
        d = density(net, prepare(bgr, variant))
        h, w = bgr.shape[:2]
        kind, n = (variant.split() + [None])[:2]
        s = int(n) / max(h, w) if kind in ("long", "sq") else 1.0
        d_img = cv2.resize(d[:max(1, round(h * s / 8)), :max(1, round(w * s / 8))], (w, h), interpolation=cv2.INTER_LINEAR)
        heat = cv2.applyColorMap(np.uint8(255 * d_img / max(d_img.max(), 1e-6)), cv2.COLORMAP_JET)
        vis = cv2.addWeighted(bgr, .55, heat, .45, 0)
        cv2.putText(vis, f"{d.sum():.0f} people ({variant})", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 4)
        cv2.imwrite(str(Path(out, f"{Path(p).stem}_{variant.replace(' ', '')}.png")), vis)
        print(f"{p}: {d.sum():.1f}")


def export(weight, out, size):
    import onnxruntime as ort
    import torch
    net = load(weight).cpu()
    x = torch.randn(2, 3, size, size)
    torch.onnx.export(net, x, out, input_names=["image"], output_names=["density"], opset_version=17,
                      dynamic_axes={"image": {0: "batch"}, "density": {0: "batch"}})
    ref = net(x).detach().numpy()
    got = ort.InferenceSession(out, providers=["CPUExecutionProvider"]).run(None, {"image": x.numpy()})[0]
    print(f"onnx {out}: {Path(out).stat().st_size / 1e6:.1f} MB, output {got.shape}, max diff vs torch "
          f"{np.abs(got - ref).max():.2e}, count diff {abs(got.sum() - ref.sum()):.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["eval", "show", "onnx"])
    ap.add_argument("args", nargs="+")
    ap.add_argument("--limit", type=int, default=316)
    ap.add_argument("--variants", default="native,long 640,long 1280,sq 640,sq 1280")
    ap.add_argument("--out", default="crowd_show")
    ap.add_argument("--long", default="long 1280")
    ap.add_argument("--size", type=int, default=1280)
    a = ap.parse_args()
    if a.cmd == "eval":
        evaluate(a.args[0], a.args[1], a.limit, a.variants.split(","))
    elif a.cmd == "show":
        show(a.args[0], a.args[1:], a.out, a.long)
    else:
        export(a.args[0], a.args[1], a.size)


if __name__ == "__main__":
    sys.exit(main())
