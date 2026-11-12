"""Bounded, debuggable aggregation of the ten DRL state features.

The monitor is deliberately policy-neutral: it aggregates measurements but
never chooses a model/FPS. Missing measurements remain ``None`` and are
reported separately instead of being silently converted to zero.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, fields
import math
from typing import Any, Mapping


STATE_FEATURES = (
    "media_match_rate",
    "transport_rtt_ms",
    "edge_backlog_norm",
    "cloud_inflight_norm",
    "edge_inference_ms",
    "cloud_e2e_ms",
    "effective_fps",
    "blur_difficulty_mean",
    "calibrated_confidence_mean",
    "box_change_mean",
)

_BOUNDED_FEATURES = {
    "media_match_rate", "edge_backlog_norm", "cloud_inflight_norm",
    "blur_difficulty_mean", "calibrated_confidence_mean", "box_change_mean",
}
_NONNEGATIVE_FEATURES = {
    "transport_rtt_ms", "edge_inference_ms", "cloud_e2e_ms", "effective_fps",
}


@dataclass(frozen=True, slots=True)
class StateObservation:
    """Measurements belonging to one source-frame window.

    ``media_matched`` is a tri-state measurement: ``None`` means the producer
    did not observe the Cloud decode/cache join (not a failed match).
    Confidence values are per-detection calibrated probabilities.
    """

    stream_key: tuple[str, str, str, int]
    window_id: int
    frame_id: int
    monotonic_ns: int
    media_matched: bool | None = None
    transport_rtt_ms: float | None = None
    edge_backlog_norm: float | None = None
    cloud_inflight_norm: float | None = None
    edge_inference_ms: float | None = None
    cloud_e2e_ms: float | None = None
    blur_difficulty: float | None = None
    calibrated_confidences: tuple[float, ...] = ()
    box_change: float | None = None
    completed: bool = False
    model_id: str = "unknown"
    transport: str = "unknown"

    def __post_init__(self) -> None:
        if len(self.stream_key) != 4 or not all(self.stream_key[:3]):
            raise ValueError("stream_key must be (node, camera, stream, epoch)")
        if self.window_id < 0 or self.frame_id < 0 or self.monotonic_ns < 0:
            raise ValueError("window, frame and monotonic time must be non-negative")
        for name in ("transport_rtt_ms", "edge_backlog_norm", "cloud_inflight_norm",
                     "edge_inference_ms", "cloud_e2e_ms", "blur_difficulty", "box_change"):
            _validate_optional(name, getattr(self, name))
        for confidence in self.calibrated_confidences:
            _validate_optional("calibrated_confidence_mean", confidence)
        if self.media_matched is not None and not isinstance(self.media_matched, bool):
            raise ValueError("media_matched must be bool or None")


@dataclass(frozen=True, slots=True)
class WindowState:
    """Immutable aggregate plus per-feature counts for logs/debugging."""

    stream_key: tuple[str, str, str, int]
    window_id: int
    first_frame_id: int
    last_frame_id: int
    model_id: str
    transport: str
    observations: int
    media_match_rate: float | None
    transport_rtt_ms: float | None
    edge_backlog_norm: float | None
    cloud_inflight_norm: float | None
    edge_inference_ms: float | None
    cloud_e2e_ms: float | None
    effective_fps: float | None
    blur_difficulty_mean: float | None
    calibrated_confidence_mean: float | None
    box_change_mean: float | None
    sample_counts: Mapping[str, int]
    missing_features: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        result = {item.name: getattr(self, item.name) for item in fields(self)}
        result["stream_key"] = list(self.stream_key)
        result["sample_counts"] = dict(self.sample_counts)
        result["missing_features"] = list(self.missing_features)
        return result


def _validate_optional(name: str, value: float | None) -> None:
    if value is None:
        return
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{name} must be finite or None")
    if name in _BOUNDED_FEATURES and not 0.0 <= parsed <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    if name in _NONNEGATIVE_FEATURES and parsed < 0.0:
        raise ValueError(f"{name} must be non-negative")


class WindowStateAggregator:
    """Aggregate out-of-order observations with bounded per-stream retention.

    Windows are explicitly sealed by ``finish``. Duplicate source frame IDs
    are rejected so retries cannot bias a mean. The most recent ``max_windows``
    aggregates are retained for an operator/debug endpoint.
    """

    def __init__(self, *, max_windows: int = 32, max_observations_per_window: int = 4096):
        if max_windows <= 0 or max_observations_per_window <= 0:
            raise ValueError("retention limits must be positive")
        self.max_windows = max_windows
        self.max_observations_per_window = max_observations_per_window
        self._pending: OrderedDict[
            tuple[tuple[str, str, str, int], int], dict[int, StateObservation]
        ] = OrderedDict()
        self._finished: OrderedDict[tuple[tuple[str, str, str, int], int], WindowState] = OrderedDict()
        self.duplicate_observations = 0
        self.capacity_drops = 0
        self.pending_window_drops = 0

    def record(self, observation: StateObservation) -> bool:
        key = (observation.stream_key, observation.window_id)
        if key in self._finished:
            return False
        frames = self._pending.get(key)
        if frames is None:
            stream_windows = [pending_key for pending_key in self._pending
                              if pending_key[0] == observation.stream_key]
            if len(stream_windows) >= self.max_windows:
                evicted_key = min(stream_windows, key=lambda item: item[1])
                evicted = self._pending.pop(evicted_key)
                self.capacity_drops += len(evicted)
                self.pending_window_drops += 1
            frames = {}
            self._pending[key] = frames
        if observation.frame_id in frames:
            self.duplicate_observations += 1
            return False
        if len(frames) >= self.max_observations_per_window:
            self.capacity_drops += 1
            return False
        frames[observation.frame_id] = observation
        return True

    def finish(self, stream_key: tuple[str, str, str, int], window_id: int) -> WindowState:
        key = (stream_key, window_id)
        observations = self._pending.pop(key, None)
        if not observations:
            raise ValueError(f"no observations for stream/window {key!r}")
        ordered = sorted(observations.values(), key=lambda item: item.frame_id)

        def values(attr: str) -> list[float]:
            return [float(value) for item in ordered
                    if (value := getattr(item, attr)) is not None]

        counts: dict[str, int] = {}
        aggregates: dict[str, float | None] = {}
        for feature, attr in (
            ("transport_rtt_ms", "transport_rtt_ms"),
            ("edge_backlog_norm", "edge_backlog_norm"),
            ("cloud_inflight_norm", "cloud_inflight_norm"),
            ("edge_inference_ms", "edge_inference_ms"),
            ("cloud_e2e_ms", "cloud_e2e_ms"),
            ("blur_difficulty_mean", "blur_difficulty"),
            ("box_change_mean", "box_change"),
        ):
            sample = values(attr)
            counts[feature] = len(sample)
            aggregates[feature] = sum(sample) / len(sample) if sample else None

        match_samples = [item.media_matched for item in ordered if item.media_matched is not None]
        counts["media_match_rate"] = len(match_samples)
        match_rate = (sum(match_samples) / len(match_samples)) if match_samples else None

        confidences = [score for item in ordered for score in item.calibrated_confidences]
        counts["calibrated_confidence_mean"] = len(confidences)
        confidence_mean = sum(confidences) / len(confidences) if confidences else None

        completed = [item for item in ordered if item.completed]
        counts["effective_fps"] = len(completed)
        duration_ns = ordered[-1].monotonic_ns - ordered[0].monotonic_ns
        # Observations represent intervals: include one estimated frame
        # interval beyond the first→last timestamp span (as for an N-frame
        # clip whose duration is N/FPS rather than (N-1)/FPS).
        covered_ns = duration_ns * len(ordered) / (len(ordered) - 1) if len(ordered) > 1 else 0
        effective_fps = (len(completed) * 1e9 / covered_ns) if covered_ns > 0 else None

        result = WindowState(
            stream_key=stream_key, window_id=window_id,
            first_frame_id=ordered[0].frame_id, last_frame_id=ordered[-1].frame_id,
            model_id=ordered[-1].model_id, transport=ordered[-1].transport,
            observations=len(ordered), media_match_rate=match_rate,
            transport_rtt_ms=aggregates["transport_rtt_ms"],
            edge_backlog_norm=aggregates["edge_backlog_norm"],
            cloud_inflight_norm=aggregates["cloud_inflight_norm"],
            edge_inference_ms=aggregates["edge_inference_ms"],
            cloud_e2e_ms=aggregates["cloud_e2e_ms"], effective_fps=effective_fps,
            blur_difficulty_mean=aggregates["blur_difficulty_mean"],
            calibrated_confidence_mean=confidence_mean,
            box_change_mean=aggregates["box_change_mean"],
            sample_counts=counts,
            missing_features=tuple(name for name in STATE_FEATURES
                                   if (match_rate if name == "media_match_rate" else
                                       confidence_mean if name == "calibrated_confidence_mean" else
                                       effective_fps if name == "effective_fps" else
                                       aggregates[name]) is None),
        )
        self._finished[key] = result
        while len(self._finished) > self.max_windows:
            self._finished.popitem(last=False)
        return result

    def recent(self, stream_key: tuple[str, str, str, int] | None = None) -> tuple[WindowState, ...]:
        states = tuple(self._finished.values())
        if stream_key is not None:
            states = tuple(state for state in states if state.stream_key == stream_key)
        return states


class StatePrometheusExporter:
    """Optional latest-window Prometheus gauges; no import unless enabled."""

    def __init__(self, *, node: str, camera: str, stream: str, epoch: int,
                 model: str = "unknown", transport: str = "unknown"):
        try:
            from prometheus_client import CollectorRegistry, Gauge
        except ImportError as exc:
            raise RuntimeError("monitor endpoint needs prometheus-client; install '.[monitor]'") from exc
        self.registry = CollectorRegistry()
        self.labels = (node, camera, stream, str(epoch), model, transport)
        label_names = ("node", "camera", "stream", "epoch", "model", "transport")
        self.values = {
            feature: Gauge(f"edge_state_{feature}", f"Latest completed-window {feature}",
                           label_names, registry=self.registry)
            for feature in STATE_FEATURES
        }
        self.available = Gauge("edge_state_feature_available", "1 when latest state feature is measured",
                               (*label_names, "feature"), registry=self.registry)
        self.samples = Gauge("edge_state_feature_samples", "Samples used in latest state aggregate",
                             (*label_names, "feature"), registry=self.registry)
        self.window = Gauge("edge_state_window_id", "Latest completed source-frame window",
                            label_names, registry=self.registry)
        self.duplicate_count = Gauge("edge_monitor_duplicate_observations_total",
                                     "Duplicate frame observations rejected", registry=self.registry)
        self.capacity_drops = Gauge("edge_monitor_capacity_drops_total",
                                    "Observations rejected at configured capacity", registry=self.registry)
        self.pending_window_drops = Gauge("edge_monitor_pending_window_drops_total",
                                          "Unfinished windows evicted at retention limit", registry=self.registry)
        self._httpd = None

    def publish(self, state: WindowState, aggregator: WindowStateAggregator) -> None:
        if state.stream_key != (self.labels[0], self.labels[1], self.labels[2], int(self.labels[3])):
            raise ValueError("state stream labels do not match exporter identity")
        labels = self.labels
        self.window.labels(*labels).set(state.window_id)
        for feature in STATE_FEATURES:
            value = getattr(state, feature)
            self.available.labels(*labels, feature).set(1 if value is not None else 0)
            self.samples.labels(*labels, feature).set(state.sample_counts.get(feature, 0))
            if value is not None:
                self.values[feature].labels(*labels).set(value)
        self.duplicate_count.set(aggregator.duplicate_observations)
        self.capacity_drops.set(aggregator.capacity_drops)
        self.pending_window_drops.set(aggregator.pending_window_drops)

    def start(self, port: int, *, addr: str = "0.0.0.0") -> None:
        if not 0 <= port <= 65535:
            raise ValueError("metrics port must be between 0 and 65535")
        if port == 0:
            return
        from prometheus_client import start_http_server
        self._httpd, _thread = start_http_server(port, addr=addr, registry=self.registry)

    def close(self) -> None:
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None
