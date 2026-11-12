"""Candidate model modes; availability is controlled by the runtime catalog."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from .messages import ModelMode, Placement


@dataclass(frozen=True, slots=True)
class ModelProfile:
    mode: ModelMode
    model_id: str
    placement: Placement
    runtime_family: str
    status: str = "candidate"

    def __post_init__(self) -> None:
        if self.mode.placement is not self.placement:
            raise ValueError("model mode and placement disagree")


DEFAULT_MODEL_REGISTRY: Mapping[ModelMode, ModelProfile] = MappingProxyType({
    ModelMode.YOLO11S_EDGE: ModelProfile(ModelMode.YOLO11S_EDGE, "yolo11s320_w8a8_qcs6490", Placement.EDGE, "snpe_npu", "diagnostic_candidate_parity_fail"),
    ModelMode.YOLO_EDGE: ModelProfile(ModelMode.YOLO_EDGE, "yolov8n_320_int8_split_traffic100", Placement.EDGE, "snpe_npu"),
    ModelMode.RTMDET_EDGE: ModelProfile(ModelMode.RTMDET_EDGE, "rtmdet_s_8xb32_300e_coco", Placement.EDGE, "snpe_npu"),
    ModelMode.DFINE_CLOUD: ModelProfile(ModelMode.DFINE_CLOUD, "dfine_full", Placement.CLOUD, "onnxruntime_cuda"),
    ModelMode.RTDETR_CLOUD: ModelProfile(ModelMode.RTDETR_CLOUD, "rtdetrv2_r18vd_sp2_120e_coco", Placement.CLOUD, "onnxruntime_cuda"),
    ModelMode.LWDETR_CLOUD: ModelProfile(ModelMode.LWDETR_CLOUD, "lwdetr_large_coco", Placement.CLOUD, "onnxruntime_cuda"),
    ModelMode.DFINEX_CLOUD: ModelProfile(ModelMode.DFINEX_CLOUD, "dfine_x_obj2coco", Placement.CLOUD, "onnxruntime_cuda"),
    ModelMode.RTDETR_SPLIT_CLOUD: ModelProfile(ModelMode.RTDETR_SPLIT_CLOUD, "rtdetrv2_r18vd_sp2_120e_coco+layer2_split",
                                               Placement.CLOUD, "triton_onnxruntime"),
})
