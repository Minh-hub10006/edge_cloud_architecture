"""
D-FINE paths for edge_multimodel:
  - YOLO  → local only (detectors/yolo)
  - DFINE mono (default): JPEG uplink → cloud dfine_full (high FPS on LAN)
  - DFINE cut_stem: DEAD for primary use (loses FPS vs mono; archive opt-in only)
  - DFINE split: segment_a on NPU → tier1 → cloud B+C
  - DFINE fdraas: full model CPU → 0x02 → cloud refiner
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np

from detectors.dfine.mono_edge_encode import encode_frame_jpeg
from detectors.dfine.ort_providers import dfine_encode_mode


class DFINEEncoder:
    """
    DFINE_ENCODE=mono (default): JPEG → cloud full D-FINE.
    DFINE_ENCODE=cut_stem: deprecated; requires DFINE_ALLOW_DEAD_CUT_STEM=1.
    DFINE_ENCODE=split: segment A backbone → tier1_flat (NPU path).
    DFINE_ENCODE=fdraas: FDRaaS 0x02 bytes.
    """

    def __init__(
        self,
        use_fdraas: Optional[bool] = None,
        use_backbone_only: bool = False,
        **kwargs,
    ):
        mode = dfine_encode_mode()
        if use_fdraas is True:
            mode = "fdraas"
        if use_backbone_only:
            mode = "split"
        self.encode_mode = mode
        self.runtime = "unknown"
        if self.encode_mode == "mono":
            self.runtime = "jpeg_cpu"
        elif self.encode_mode == "cut_stem":
            from detectors.dfine.edge_stem import EdgeStem

            self._stem = EdgeStem(
                context_bin=kwargs.get("stem_qnn_bin"),
                onnx_path=kwargs.get("stem_onnx_path"),
            )
            self.runtime = self._stem.runtime
        elif self.encode_mode == "fdraas":
            from detectors.dfine.fdraas_edge_infer import FDRaaSEdgeInfer

            self._fdraas = FDRaaSEdgeInfer(**kwargs)
            self.runtime = getattr(self._fdraas, "runtime", "onnxruntime_cpu")
        else:
            from detectors.dfine.edge_backbone import EdgeBackbone

            self._backbone = EdgeBackbone(kwargs.get("onnx_path"))
            self.runtime = self._backbone.runtime

    def encode(self, frame: np.ndarray) -> Tuple[float, Union[bytes, np.ndarray]]:
        if self.encode_mode == "mono":
            return encode_frame_jpeg(frame)
        if self.encode_mode == "cut_stem":
            return self._stem.encode(frame)
        if self.encode_mode == "fdraas":
            t, payload, _n = self._fdraas.encode_fdraas(frame)
            return t, payload
        return self._backbone.forward_flat(frame)

    @property
    def uplink_layout(self) -> str:
        if self.encode_mode == "mono":
            return "mono_jpeg"
        if self.encode_mode == "cut_stem":
            from detectors.dfine.stem_codec import stem_uplink_mode

            return "cut_stem_int4gz" if stem_uplink_mode() == "int4gz" else "cut_stem_int8"
        if self.encode_mode == "fdraas":
            return "fdraas"
        from detectors.dfine.tier1_codec import tier1_uplink_int8

        if tier1_uplink_int8():
            return "tier1_int8"
        return "tier1"


DFINERunner = DFINEEncoder
