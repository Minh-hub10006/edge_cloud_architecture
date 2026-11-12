"""
FDRaaS edge inference: model → pred_corners / pred_logits → pack format 0x02.

References:
  Parallelism_pipeline_AI_detection/scripts/fdraas_edge_send_distributions.py
  Parallelism_pipeline_AI_detection/scripts/tensor_send.py
"""

from __future__ import annotations

import os
import sys
import time
from typing import List, Optional, Tuple

import numpy as np

from comm.fdraas.tensor_protocol import pack_distributions_fdraas
from detectors.demo_boxes import MOCK_REF_POINTS
from real_data import require_real
from detectors.dfine.edge_backbone import pipeline_root
from detectors.dfine.preprocess import frame_to_images_tensor


def _setup_pipeline_imports() -> str:
    root = pipeline_root()
    for p in (root, os.path.join(root, "scripts")):
        if p not in sys.path:
            sys.path.insert(0, p)
    return root


def _pick_output(out_map: dict, names: tuple) -> Optional[np.ndarray]:
    for n in names:
        if n in out_map:
            return out_map[n]
    for k, v in out_map.items():
        lk = k.lower()
        if any(s in lk for s in names):
            return v
    return None


class FDRaaSEdgeInfer:
    """Edge: D-FINE forward (return_corners) → FDRaaS 0x02 bytes."""

    def __init__(
        self,
        config_rel: str = "configs/dfine/dfine_hgnetv2_l_coco.yml",
        checkpoint_rel: str = "models/dfine_l_obj2coco_e25.pth",
        onnx_path: Optional[str] = None,
        top_k: int = 40,
        conf_thr: float = 0.05,
        reg_max: int = 32,
        device: str = "cpu",
    ):
        self.top_k = top_k
        self.conf_thr = conf_thr
        self.reg_max = reg_max
        self.bins_per_edge = reg_max + 1
        self._device = device
        self._torch_model = None
        self._ort_session = None
        self._providers: list = []
        self.runtime = "mock"
        self._pred_corners_to_objects = None

        root = _setup_pipeline_imports()
        self._config_path = os.path.join(root, config_rel)
        ckpt_edge = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            checkpoint_rel,
        )
        self._ckpt_path = ckpt_edge if os.path.isfile(ckpt_edge) else os.path.join(
            root, "checkpoints", os.path.basename(checkpoint_rel)
        )

        self._load_helpers()
        onnx = onnx_path or os.environ.get("DFINE_FULL_ONNX")
        if not onnx or not os.path.isfile(onnx):
            edge_root = os.environ.get("PIPELINE_AI_DETECTION_ROOT", "")
            if edge_root:
                candidate = os.path.join(edge_root, "onnx", "dfine_full.onnx")
                if os.path.isfile(candidate):
                    onnx = candidate
        if not onnx or not os.path.isfile(onnx):
            onnx = os.path.join(root, "onnx_export", "dfine_full.onnx")
        if os.path.isfile(onnx):
            self._init_ort(onnx)
        elif os.path.isfile(self._config_path) and os.path.isfile(self._ckpt_path):
            try:
                import torch  # noqa: F401
                self._init_torch()
            except ImportError:
                print("[FDRaaSEdgeInfer] torch not installed — using mock FDRaaS payloads")
        else:
            print("[FDRaaSEdgeInfer] mock mode — place checkpoint or export dfine_full.onnx")

    def _load_helpers(self):
        try:
            from detectors.dfine.pred_corners import (
                pred_boxes_to_objects,
                pred_corners_to_objects,
            )

            self._pred_corners_to_objects = pred_corners_to_objects
            self._pred_boxes_to_objects = pred_boxes_to_objects
            return
        except ImportError:
            self._pred_boxes_to_objects = None
        try:
            from fdraas_edge_send_distributions import pred_corners_to_objects

            self._pred_corners_to_objects = pred_corners_to_objects
        except ImportError as e:
            print(f"[FDRaaSEdgeInfer] postprocess helpers unavailable: {e}")

    def _init_ort(self, path: str):
        from detectors.dfine.ort_providers import create_ort_session

        self._ort_session, self.runtime, self._providers = create_ort_session(
            path, label="FDRaaSEdgeInfer"
        )

    def _init_torch(self):
        from fdraas_edge_send_distributions import build_model_and_get_distributions

        self._torch_model, self.reg_max = build_model_and_get_distributions(
            self._config_path, self._ckpt_path, self._device
        )
        self.bins_per_edge = int(self.reg_max) + 1
        print(f"[FDRaaSEdgeInfer] PyTorch: {self._ckpt_path}")

    def encode_fdraas(self, frame_bgr: np.ndarray) -> Tuple[float, bytes, int]:
        t0 = time.time()
        objects, ref_points = self._infer_objects(frame_bgr)
        payload = pack_distributions_fdraas(
            objects,
            scene_entropy=0.0,
            ref_points=ref_points,
            bins_per_edge=self.bins_per_edge,
        )
        return time.time() - t0, payload, len(objects)

    def _infer_objects(self, frame_bgr: np.ndarray):
        if self._pred_corners_to_objects is None:
            require_real("FDRaaSEdgeInfer", "pred_corners helpers unavailable")
            return self._mock_objects()
        if self._ort_session is None and self._torch_model is None:
            require_real(
                "FDRaaSEdgeInfer",
                "no ORT session and no torch checkpoint — set DFINE_FULL_ONNX",
            )
            return self._mock_objects()

        x = frame_to_images_tensor(frame_bgr)

        if self._ort_session is not None:
            in_name = self._ort_session.get_inputs()[0].name
            outs = self._ort_session.run(None, {in_name: x.astype(np.float32)})
            out_map = {o.name: v for o, v in zip(self._ort_session.get_outputs(), outs)}
            pred_corners = _pick_output(out_map, ("pred_corners", "corners"))
            pred_logits = _pick_output(out_map, ("pred_logits", "logits"))
            pred_boxes = _pick_output(out_map, ("pred_boxes", "boxes"))
            ref_points = _pick_output(out_map, ("ref_points", "anchors"))
            if pred_logits is None:
                print("[FDRaaSEdgeInfer] missing pred_logits in ORT outputs:", list(out_map))
                return self._mock_objects()
            if pred_corners is not None:
                return self._pred_corners_to_objects(
                    pred_corners,
                    pred_logits,
                    self.reg_max,
                    top_k=self.top_k,
                    conf_thr=self.conf_thr,
                    ref_points=ref_points,
                )
            if pred_boxes is not None and getattr(self, "_pred_boxes_to_objects", None):
                objs, refs = self._pred_boxes_to_objects(
                    pred_boxes,
                    pred_logits,
                    self.reg_max,
                    top_k=self.top_k,
                    conf_thr=self.conf_thr,
                )
                if objs:
                    return objs, refs
            print(
                "[FDRaaSEdgeInfer] ORT outputs lack pred_corners/pred_boxes:",
                list(out_map.keys()),
            )
            return self._mock_objects()

        if self._torch_model is not None:
            import torch

            xt = torch.from_numpy(x).to(self._device)
            with torch.no_grad():
                out = self._torch_model(xt)
            pc = out.get("pred_corners")
            pl = out.get("pred_logits")
            rp = out.get("ref_points")
            if pc is None or pl is None:
                return self._mock_objects()
            return self._pred_corners_to_objects(
                pc,
                pl,
                self.reg_max,
                top_k=self.top_k,
                conf_thr=self.conf_thr,
                ref_points=rp,
            )

        return self._mock_objects()

    def _mock_objects(self):
        objects = []
        ref_pts = []
        n = min(len(MOCK_REF_POINTS), self.top_k)
        for i in range(n):
            conf = 0.55 - i * 0.04
            objects.append(
                {
                    "class_id": 2,
                    "confidence": conf,
                    "left_dist": [0.05] * self.bins_per_edge,
                    "top_dist": [0.05] * self.bins_per_edge,
                    "right_dist": [0.05] * self.bins_per_edge,
                    "bottom_dist": [0.05] * self.bins_per_edge,
                }
            )
            ref_pts.append(MOCK_REF_POINTS[i])
        return objects, ref_pts
