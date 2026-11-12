"""Static RTMDet-Small raw boxes+scores ONNX, without NMS/custom deploy ops.

Head.forward already scales regression distances by stride. Priors have offset=0.
This is the float reference for SNPE conversion; parity is a mandatory gate.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import shutil
import subprocess
import tempfile

from candidate_artifact import seal


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--image", required=True, type=Path, help="representative non-square BGR image")
    args = parser.parse_args()
    repo, checkpoint = args.repo.expanduser().resolve(strict=True), args.checkpoint.expanduser().resolve(strict=True)
    output = args.output.expanduser().resolve()
    if output.exists() or Path(str(output) + ".json").exists() or output.suffix != ".onnx":
        parser.error("choose a new .onnx output; overwrite is forbidden")
    if subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=repo):
        parser.error("upstream tracked checkout must be clean")
    sys.path.insert(0, str(repo))
    import torch
    import numpy as np
    import cv2
    import onnxruntime as ort
    from mmdet.apis import init_detector
    from inference.rtmdet_snpe import RtmdetSmallSnpeAdapter
    model = init_detector(str(repo / "configs/rtmdet/rtmdet_s_8xb32-300e_coco.py"), str(checkpoint), device="cpu").eval()
    # MMEngine's generic checkpoint loader is permissive. Architecture admission is not.
    state = torch.load(str(checkpoint), map_location="cpu", weights_only=False)["state_dict"]
    model.load_state_dict(state, strict=True)
    image = cv2.imread(str(args.image))
    if image is None or image.shape[0] == image.shape[1]:
        raise ValueError("provide a non-square representative image")
    # Pure preprocessing only, never construct a DSP session on exporter.
    adapter = object.__new__(RtmdetSmallSnpeAdapter)
    adapter.manifest = {"input_layout": "nchw"}
    tensor = adapter.prepare_frame(image).tensor
    dummy = torch.from_numpy(tensor)
    with torch.no_grad():
        cls_scores, bbox_preds = model.bbox_head(model.extract_feat(dummy))
        priors = model.bbox_head.prior_generator.grid_priors([x.shape[-2:] for x in cls_scores], device="cpu")

    class Export(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = model
            for index, prior in enumerate(priors):
                self.register_buffer(f"prior_{index}", prior[:, :2][None])

        def forward(self, value):
            cls, distances = self.model.bbox_head(self.model.extract_feat(value))
            boxes, scores = [], []
            for index, (c, d) in enumerate(zip(cls, distances)):
                score = c.permute(0, 2, 3, 1).reshape(1, -1, 80).sigmoid()
                distance = d.permute(0, 2, 3, 1).reshape(1, -1, 4)
                prior = getattr(self, f"prior_{index}")
                boxes.append(torch.cat((prior - distance[..., :2], prior + distance[..., 2:]), -1))
                scores.append(score)
            return torch.cat(boxes, 1), torch.cat(scores, 1)

    exporter = Export().eval()
    output.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        reference = tuple(x.numpy() for x in exporter(dummy))
        with tempfile.TemporaryDirectory(prefix="rtmdet-candidate-export-") as tmp:
            temporary = Path(tmp) / "model.onnx"
            torch.onnx.export(exporter, dummy, str(temporary), input_names=["input"], output_names=["boxes", "scores"],
                              opset_version=13, do_constant_folding=True)
            with temporary.open("rb") as source, output.open("xb") as target:
                shutil.copyfileobj(source, target)
    session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
    actual = session.run(["boxes", "scores"], {"input": tensor})
    for wanted, found in zip(reference, actual):
        np.testing.assert_allclose(found, wanted, rtol=1e-3, atol=1e-3)
    seal(output, "rtmdet_small", repo=repo, checkpoint=checkpoint,
         export_evidence={"torch_onnx_raw_parity": "PASS", "rtol": 1e-3, "atol": 1e-3,
                          "image_sha256": __import__("candidate_artifact").sha256(args.image)})
    print("Float ONNX parity PASS; DLC conversion/NPU quality still pending")


if __name__ == "__main__":
    main()
