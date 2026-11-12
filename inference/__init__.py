"""Real detector adapters and result normalization for the Edge/Cloud runtime."""

from .adapters import DfineCloudAdapter, YoloSnpeAdapter
from .catalog import ModelAdapterCatalog, phase7_cloud_catalog, phase7_edge_catalog
from .cloud_detr import RtdetrV2SmallAdapter, LwDetrLargeAdapter
from .rtmdet_snpe import RtmdetSmallSnpeAdapter

__all__ = [
    "DfineCloudAdapter", "YoloSnpeAdapter", "ModelAdapterCatalog",
    "phase7_cloud_catalog", "phase7_edge_catalog",
    "RtdetrV2SmallAdapter", "LwDetrLargeAdapter", "RtmdetSmallSnpeAdapter",
]
