"""Python types mirroring docs/COMMUNICATION_PROTOCOL.md (not generated from proto)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

FPS_LIST = [5, 10, 15, 20, 25, 30]
N_NODES = 20
STATE_DIM_PER_NODE = 9


@dataclass
class NodeState:
    bandwidth: float = 50.0
    latency: float = 20.0
    queue_norm: float = 0.0
    u_edge: float = 0.5
    u_cloud: float = 0.5
    t_process_ms: float = 0.0
    q_cloud_norm: float = 0.0
    task_gamma: float = 0.0
    imgsz_norm: float = 0.5

    def to_array(self) -> np.ndarray:
        return np.array(
            [
                self.bandwidth,
                self.latency,
                self.queue_norm,
                self.u_edge,
                self.u_cloud,
                self.t_process_ms,
                self.q_cloud_norm,
                self.task_gamma,
                self.imgsz_norm,
            ],
            dtype=np.float32,
        )

    @classmethod
    def from_array(cls, arr: np.ndarray) -> "NodeState":
        a = np.asarray(arr, dtype=np.float32).flatten()
        if a.size < STATE_DIM_PER_NODE:
            padded = np.zeros(STATE_DIM_PER_NODE, dtype=np.float32)
            padded[: min(a.size, 6)] = a[: min(a.size, 6)]
            if a.size >= 5:
                padded[6] = float(np.clip(a[4], 0.0, 1.0))
            padded[8] = 0.5
            a = padded
        if a.size != STATE_DIM_PER_NODE:
            raise ValueError(f"expected {STATE_DIM_PER_NODE} floats, got {a.size}")
        return cls(
            bandwidth=float(a[0]),
            latency=float(a[1]),
            queue_norm=float(a[2]),
            u_edge=float(a[3]),
            u_cloud=float(a[4]),
            t_process_ms=float(a[5]),
            q_cloud_norm=float(a[6]),
            task_gamma=float(a[7]),
            imgsz_norm=float(a[8]),
        )


@dataclass
class GlobalState:
    nodes: List[NodeState] = field(default_factory=list)
    u_cloud: float = 0.5
    updated_at_ms: int = 0

    def flatten(self) -> np.ndarray:
        if len(self.nodes) < N_NODES:
            padded = list(self.nodes) + [
                NodeState() for _ in range(N_NODES - len(self.nodes))
            ]
        else:
            padded = self.nodes[:N_NODES]
        return np.stack([n.to_array() for n in padded], axis=0).flatten()


@dataclass
class Action:
    mode: int  # 0=YOLO, 1=DFINE
    fps_idx: int
    log_prob: float = 0.0

    @property
    def mode_str(self) -> str:
        return "YOLO" if self.mode == 0 else "DFINE"

    @property
    def target_fps(self) -> int:
        return FPS_LIST[self.fps_idx]


@dataclass
class TensorMeta:
    shape: List[int]
    dtype: str = "float32"
    layout: str = "C"  # "C" | "tier1_flat"
    compression: str = "none"  # "none" | "gzip"


@dataclass
class FeaturePacketData:
    frame_id: str
    node_id: str
    payload: bytes
    meta: TensorMeta
    sent_at_ms: int = 0


@dataclass
class CloudFeedbackData:
    frame_id: str
    node_id: str
    matched_id: str
    cloud_usage: float
    decode_ms: float = 0.0
    recv_at_ms: int = 0


@dataclass
class TrajectoryStep:
    node_id: str
    state_local: List[float]
    mode: int
    fps_idx: int
    reward: float
    log_prob: float
    timestamp_ms: int = 0
