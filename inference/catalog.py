"""Lazy model registration; transport and tracking do not depend on model code."""

from __future__ import annotations

from collections.abc import Callable
import os

from contracts import ModelMode, Placement

from .adapters import DfineCloudAdapter, FrameDetector, YoloSnpeAdapter


class ModelAdapterCatalog:
    def __init__(self) -> None:
        self._factories: dict[ModelMode, Callable[[], FrameDetector]] = {}

    def register(self, mode: ModelMode, factory: Callable[[], FrameDetector]) -> None:
        if not isinstance(mode, ModelMode):
            raise TypeError("mode must be a ModelMode")
        if not callable(factory):
            raise TypeError("adapter factory must be callable")
        if mode in self._factories:
            raise ValueError(f"adapter already registered for {mode.value}")
        self._factories[mode] = factory

    def supports(self, mode: ModelMode) -> bool:
        return mode in self._factories

    def modes(self, placement: Placement | None = None) -> tuple[ModelMode, ...]:
        return tuple(mode for mode in self._factories if placement is None or mode.placement is placement)

    def load(self, mode: ModelMode) -> FrameDetector:
        try:
            detector = self._factories[mode]()
        except KeyError as exc:
            raise ValueError(f"no adapter registered for {mode.value}") from exc
        if not getattr(detector, "model_id", None) or not callable(getattr(detector, "infer", None)):
            raise TypeError(f"adapter for {mode.value} must provide model_id and infer")
        return detector


def phase7_edge_catalog(
    *, yolo_dlc: str | None = None, yolo_imgsz: int = 320,
    yolo_layout: str = "nchw", confidence: float = 0.25,
    rtmdet_dlc: str | None = None,
    yolo11s_dlc: str | None = None, yolo11s_allow_diagnostic: bool = False,
    yolo11s_confidence: float = .1, yolo11s_nms_iou: float = .5,
) -> ModelAdapterCatalog:
    catalog = ModelAdapterCatalog()
    catalog.register(ModelMode.YOLO_EDGE, lambda: YoloSnpeAdapter(
        dlc_path=yolo_dlc, imgsz=yolo_imgsz,
        input_layout=yolo_layout, score_threshold=confidence,
    ))
    if rtmdet_dlc:
        from .rtmdet_snpe import RtmdetSmallSnpeAdapter
        catalog.register(ModelMode.RTMDET_EDGE, lambda: RtmdetSmallSnpeAdapter(rtmdet_dlc, confidence=confidence))
    if yolo11s_dlc:
        from .yolo11s_snpe import Yolo11sSnpeAdapter
        catalog.register(ModelMode.YOLO11S_EDGE, lambda: Yolo11sSnpeAdapter(yolo11s_dlc,
            allow_diagnostic=yolo11s_allow_diagnostic, confidence=yolo11s_confidence, nms_iou=yolo11s_nms_iou))
    return catalog


def phase7_cloud_catalog(*, dfine_onnx: str | None = None,
                         rtdetrv2_onnx: str | None = None, lwdetr_onnx: str | None = None,
                         dfinex_onnx: str | None = None, enabled: tuple[str, ...] | None = None) -> ModelAdapterCatalog:
    catalog = ModelAdapterCatalog()
    enabled = enabled if enabled is not None else ("dfine",)
    if enabled == ("none",):
        # Receive Edge results and run shared tracking/media without a GPU model.
        return catalog
    if not enabled or len(set(enabled)) != len(enabled) or set(enabled) - {"dfine", "rtdetrv2_small", "lwdetr_large", "dfine_x",
                                                                           "rtdetr_split"}:
        raise ValueError("CLOUD_MODELS must select distinct dfine,rtdetrv2_small,lwdetr_large,dfine_x,rtdetr_split")
    if "dfine" in enabled:
        catalog.register(ModelMode.DFINE_CLOUD, lambda: DfineCloudAdapter(dfine_onnx))
    if "rtdetrv2_small" in enabled:
        if not rtdetrv2_onnx:
            raise ValueError("RTDETRV2_SMALL_ONNX required for selected model")
        from .cloud_detr import RtdetrV2SmallAdapter
        catalog.register(ModelMode.RTDETR_CLOUD, lambda: RtdetrV2SmallAdapter(rtdetrv2_onnx,
            confidence=float(os.getenv("RTDETRV2_CONF_THR", "0.10")), top_k=int(os.getenv("RTDETRV2_TOP_K", "300")),
            allow_imported_candidate=os.getenv("RTDETRV2_ALLOW_IMPORTED_CANDIDATE", "0") == "1"))
    if "lwdetr_large" in enabled:
        if not lwdetr_onnx:
            raise ValueError("LWDETR_LARGE_ONNX required for selected model")
        from .cloud_detr import LwDetrLargeAdapter
        catalog.register(ModelMode.LWDETR_CLOUD, lambda: LwDetrLargeAdapter(lwdetr_onnx,
            confidence=float(os.getenv("LWDETR_CONF_THR", "0.10")), top_k=int(os.getenv("LWDETR_TOP_K", "300")),
            allow_imported_candidate=os.getenv("LWDETR_ALLOW_IMPORTED_CANDIDATE", "0") == "1"))
    if "dfine_x" in enabled:
        if not dfinex_onnx:
            raise ValueError("DFINEX_ONNX required for selected model")
        from .cloud_detr import DfineXAdapter
        catalog.register(ModelMode.DFINEX_CLOUD, lambda: DfineXAdapter(dfinex_onnx,
            confidence=float(os.getenv("DFINEX_CONF_THR", "0.10")), top_k=int(os.getenv("DFINEX_TOP_K", "300")),
            allow_imported_candidate=os.getenv("DFINEX_ALLOW_IMPORTED_CANDIDATE", "0") == "1"))
    if "rtdetr_split" in enabled:  # path B tail, Triton only (inference/triton_detr.py)
        from .triton_detr import RtdetrSplitAdapter
        catalog.register(ModelMode.RTDETR_SPLIT_CLOUD, lambda: RtdetrSplitAdapter(
            head_id=os.getenv("RTDETR_SPLIT_HEAD_ID", ""),
            confidence=float(os.getenv("RTDETRV2_CONF_THR", "0.10")), top_k=int(os.getenv("RTDETRV2_TOP_K", "300"))))
    return catalog
