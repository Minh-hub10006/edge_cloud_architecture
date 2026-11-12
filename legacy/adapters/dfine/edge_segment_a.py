"""Alias: EdgeSegmentA = EdgeBackbone (Segment A from pipeline export)."""

from detectors.dfine.edge_backbone import EdgeBackbone, default_segment_a_onnx


class EdgeSegmentA(EdgeBackbone):
    def encode(self, frame):
        return self.forward_flat(frame)
