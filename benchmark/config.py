"""Benchmark identities and validation, independent of model libraries."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any


@dataclass(frozen=True)
class RunConfig:
    campaign_id: str
    run_id: str
    node_id: str
    model_key: str
    placement: str
    measured_frames: int = 900
    warmup_frames: int = 30
    profile: str = "model_capacity"
    transport: str = "none"
    repetition: int = 1

    def __post_init__(self) -> None:
        for name in ("campaign_id", "run_id", "node_id", "model_key"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a nonempty string")
            if name != "model_key" and (len(value) > 128 or value in {".", ".."}
                    or not all(c.isascii() and (c.isalnum() or c in "-_.") for c in value)):
                raise ValueError(f"{name} must be a safe ASCII identifier, not a path")
        if self.placement not in {"edge", "cloud"}:
            raise ValueError("placement must be edge or cloud")
        if self.profile not in {"model_capacity", "pipeline_capacity", "realtime"}:
            raise ValueError("unknown profile")
        if self.transport not in {"none", "grpc", "kafka"}:
            raise ValueError("unknown transport")
        if (self.profile == "model_capacity") != (self.transport == "none"):
            raise ValueError("model capacity has no transport; pipeline requires grpc/kafka")
        for name, low, high in (("measured_frames", 1, 100_000),
                                ("warmup_frames", 0, 100_000), ("repetition", 1, 10_000)):
            value = getattr(self, name)
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f"{name} must be an integer in [{low}, {high}]")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_campaign(config: dict[str, Any]) -> dict[str, Any]:
    """Validate topology now; distributed execution is a separate next milestone."""
    if config.get("schema_version") != 1:
        raise ValueError("campaign schema_version must be 1")
    nodes = config.get("nodes")
    if not isinstance(nodes, list) or not nodes:
        raise ValueError("campaign requires nodes")
    ids = set()
    for node in nodes:
        if not isinstance(node, dict):
            raise ValueError("each node must be an object")
        node_id = node.get("node_id")
        if not isinstance(node_id, str) or not node_id.strip() or node_id in ids:
            raise ValueError("node_id must be unique and nonempty")
        ids.add(node_id)
        if not isinstance(node.get("video"), str) or not node["video"]:
            raise ValueError("each node requires its local video path")
    cloud = config.get("cloud")
    if not isinstance(cloud, dict) or not cloud.get("node_id") or cloud["node_id"] in ids:
        raise ValueError("cloud requires a distinct node_id")
    cases = config.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("campaign requires cases")
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("each case must be an object")
        RunConfig("validate", "validate", "validate", case.get("model_key", ""),
                  case.get("placement", ""), profile=case.get("profile", ""),
                  transport=case.get("transport", ""))
    return config


def finite_nonnegative(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be numeric")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return number
