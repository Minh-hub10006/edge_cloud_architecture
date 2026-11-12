"""Build one pinned YOLO11s320 Float reference and traffic calibration packet."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from yolo11s_contract import SIZE, SHAPES, CONFIDENCE, NMS_IOU, prepare, decode, digest, overlay, normalize, COCO_NAMES
from scripts.rebuild_rtmdet_small import compare_detections

ULTRALYTICS_VERSION = "8.3.221"
CHECKPOINT_URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.pt"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.output_dir.exists() or not 1 <= args.threads <= 16 or args.checkpoint.name != "yolo11s.pt":
        parser.error("new output directory, official yolo11s.pt and threads1..16 required")
    import copy
    import types
    import cv2
    import torch
    import onnx
    import onnxruntime as ort
    import ultralytics
    from ultralytics import YOLO
    from ultralytics.data.augment import LetterBox
    if ultralytics.__version__ != ULTRALYTICS_VERSION:
        raise ValueError("export version mismatch")
    cv2.setNumThreads(1)
    torch.set_num_threads(args.threads)
    root = args.output_dir.resolve()
    root.mkdir(parents=True)
    model = YOLO(str(args.checkpoint.resolve(strict=True)))
    if tuple(model.names[i] for i in range(80)) != COCO_NAMES:
        raise ValueError("not COCO80 detection checkpoint")
    export_model = copy.deepcopy(model.model).cpu().float().eval()
    parameter_count = sum(p.numel() for p in export_model.parameters())
    if not 9_000_000 <= parameter_count <= 10_000_000:
        raise ValueError("not YOLO11s parameter count")
    export_model.fuse(verbose=False)
    for module in export_model.modules():
        if hasattr(module, "export"):
            module.export, module.format, module.dynamic = True, "onnx", False

    example = torch.zeros((1, 3, SIZE, SIZE), dtype=torch.float32)
    with torch.no_grad():
        export_model(example)  # Initialize fixed anchors using the pinned upstream head.
    head = export_model.model[-1]
    if head.end2end or head.nc != 80 or head.reg_max != 16:
        raise ValueError("unsupported detection head for independent-branch export")

    def independent_forward(self, features):
        # Never concatenate regression/logits or pixel coordinates/probabilities.
        # Shared backbone features are unchanged; only head tensor routing differs.
        regression = torch.cat([self.cv2[i](features[i]).reshape(1, 4 * self.reg_max, -1)
                                for i in range(self.nl)], dim=2)
        logits = torch.cat([self.cv3[i](features[i]).reshape(1, self.nc, -1)
                            for i in range(self.nl)], dim=2)
        boxes = self.decode_bboxes(self.dfl(regression), self.anchors.unsqueeze(0), xywh=False) * self.strides
        return boxes.transpose(1, 2), logits.sigmoid().transpose(1, 2)

    head.forward = types.MethodType(independent_forward, head)

    class Wrapper(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.detector = export_model

        def forward(self, images):
            return self.detector(images)

    wrapper = Wrapper().eval()
    with torch.no_grad():
        wrapper(example)  # Freeze anchors before export.
    onnx_path = root / "yolo11s320_float.onnx"
    torch.onnx.export(wrapper, example, str(onnx_path), input_names=["images"], output_names=list(SHAPES),
                      opset_version=13, do_constant_folding=True, dynamic_axes=None)
    graph = onnx.load(str(onnx_path))
    onnx.checker.check_model(graph)
    producers = {output: node for node in graph.graph.node for output in node.output}
    score_output = producers["scores"]
    if score_output.op_type != "Transpose" or producers[score_output.input[0]].op_type != "Sigmoid":
        raise ValueError("scores must come directly from independent Sigmoid -> Transpose")
    if len(graph.graph.input) != 1 or graph.graph.input[0].name != "images":
        raise ValueError("unexpected initializer inputs")
    options = ort.SessionOptions()
    options.intra_op_num_threads, options.inter_op_num_threads = args.threads, 1
    session = ort.InferenceSession(str(onnx_path), sess_options=options, providers=["CPUExecutionProvider"])
    if {v.name: tuple(v.shape) for v in session.get_outputs()} != SHAPES:
        raise ValueError("wrong exported shape")
    cap = cv2.VideoCapture(str(args.video.resolve(strict=True)))
    try:
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not cap.isOpened() or frame_count <= 1200:
            raise ValueError("traffic video must contain frames120/600/1200")
        references = []
        for index in (120, 600, 1200):
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, image = cap.read()
            if not ok:
                raise ValueError(f"cannot read frame {index}")
            prepared = prepare(image)
            oracle_input = LetterBox((SIZE, SIZE), auto=False)(image=image)
            oracle_input = np.ascontiguousarray(oracle_input[..., ::-1].transpose(2, 0, 1)[None], dtype=np.float32) / 255
            np.testing.assert_array_equal(prepared.tensor, oracle_input)
            with torch.no_grad():
                torch_values = dict(zip(SHAPES, [v.numpy() for v in wrapper(torch.from_numpy(prepared.tensor))]))
            values = dict(zip(SHAPES, session.run(list(SHAPES), {"images": prepared.tensor})))
            for name in SHAPES:
                np.testing.assert_allclose(values[name], torch_values[name], atol=.002 if name == "boxes" else .0001, rtol=.0001)
            boxes = decode(values, prepared)
            predicted = model.predict(image, imgsz=SIZE, rect=False, conf=CONFIDENCE, iou=NMS_IOU,
                                      max_det=300, device="cpu", verbose=False)[0].boxes
            oracle = normalize(predicted.xyxy.numpy(), predicted.conf.numpy(), predicted.cls.numpy(),
                               width=prepared.width, height=prepared.height)
            parity = compare_detections(oracle, boxes)
            folder = root / f"frame_{index:06d}"
            folder.mkdir()
            if not cv2.imwrite(str(folder / "source.png"), image):
                raise OSError("source image write failed")
            prepared.tensor.transpose(0, 2, 3, 1).astype("<f4").tofile(folder / "input_nhwc.raw")
            np.savez_compressed(folder / "reference_outputs.npz", **values)
            entry = {"frame": index, "folder": folder.name, "image_shape": list(image.shape),
                     "parity": parity, "detections": [asdict(v) for v in boxes]}
            (folder / "reference.json").write_text(json.dumps(entry, indent=2), encoding="utf-8")
            overlay(image, boxes, folder / "float_boxes.png")
            references.append(entry)
            print(json.dumps({"frame": index, "gate": parity["gate"], "detections": len(boxes)}), flush=True)
        calibration = root / "calibration_nhwc64"
        calibration.mkdir()
        indices = list(dict.fromkeys(int(i) for i in np.linspace(0, frame_count - 1, 70)
                                     if int(i) not in (120, 600, 1200)))[:64]
        for index in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, image = cap.read()
            if not ok:
                raise ValueError("calibration video decode failed")
            prepare(image).tensor.transpose(0, 2, 3, 1).astype("<f4").tofile(calibration / f"frame_{index:06d}.raw")
    finally:
        cap.release()
    manifest = {"schema_version": 1, "kind": "yolo11s320_candidate", "input_size": SIZE,
                "checkpoint_url": CHECKPOINT_URL, "checkpoint_sha256": digest(args.checkpoint),
                "onnx_sha256": digest(onnx_path), "ultralytics_version": ULTRALYTICS_VERSION,
                "torch_version": torch.__version__, "parameter_count": parameter_count,
                "source_video_sha256": digest(args.video), "opset": 13,
                "input": {"name": "images", "shape": [1, 3, SIZE, SIZE], "dtype": "float32", "color": "RGB"},
                "snpe_input": {"shape": [1, SIZE, SIZE, 3], "dtype": "float32", "layout": "nhwc"},
                "preprocessing": "centered letterbox114, RGB /255, no mean/std; auto=False",
                "outputs": {name: list(shape) for name, shape in SHAPES.items()},
                "decoder": "xyxy in input pixels; probabilities; best class per query; class-aware host NMS",
                "head_routing": "independent regression and classification; no mixed box-score concatenation",
                "score_output_path": "Sigmoid -> Transpose -> scores",
                "confidence": CONFIDENCE, "nms_iou": NMS_IOU, "top_k": 300,
                "calibration_frames": indices, "reference_frames": [v["frame"] for v in references],
                "float_gate": "PASS", "quantization_gate": "NOT_RUN", "npu_gate": "NOT_RUN",
                "quality_status": "NOT_EVALUATED", "production_ready": False,
                "license": "AGPL-3.0 (Ultralytics); review obligations before distribution"}
    (root / "candidate.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    inventory = {path.relative_to(root).as_posix(): digest(path) for path in sorted(root.rglob("*"))
                 if path.is_file() and path.name != "BUNDLE_SHA256.json"}
    (root / "BUNDLE_SHA256.json").write_text(json.dumps(inventory, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(root), "float_gate": "PASS", "input_size": SIZE,
                      "checkpoint_sha256": manifest["checkpoint_sha256"], "onnx_sha256": manifest["onnx_sha256"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
