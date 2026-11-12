"""
D-FINE detectors — import submodules directly to avoid pulling cv2 on cloud.

  from detectors.dfine.runner import DFINERunner
  from detectors.dfine.cloud_segment_bc import CloudSegmentBC
"""

__all__ = [
    "DFINEEncoder",
    "DFINERunner",
    "EdgeBackbone",
    "EdgeSegmentA",
    "FDRaaSEdgeInfer",
    "CloudSegmentBC",
]
