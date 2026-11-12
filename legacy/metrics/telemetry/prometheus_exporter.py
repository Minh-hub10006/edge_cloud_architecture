"""Prometheus metrics derived from the canonical trace schema."""
from __future__ import annotations

from typing import Any

try:
    from prometheus_client import (
        CollectorRegistry,
        Counter,
        Gauge,
        Histogram,
        start_http_server,
    )

    HAS_PROMETHEUS = True
except ImportError:
    HAS_PROMETHEUS = False
    CollectorRegistry = Any  # type: ignore[misc,assignment]


LATENCY_BUCKETS_MS = (
    0.1,
    0.25,
    0.5,
    1.0,
    2.5,
    5.0,
    10.0,
    20.0,
    50.0,
    100.0,
    200.0,
    500.0,
    1_000.0,
    2_000.0,
    5_000.0,
    10_000.0,
    30_000.0,
)

PAYLOAD_BUCKETS_BYTES = (
    1_024,
    4_096,
    16_384,
    65_536,
    262_144,
    1_048_576,
    4_194_304,
    8_388_608,
    16_777_216,
    33_554_432,
)


class PrometheusExporter:
    def __init__(self, *, node: str, registry: Any | None = None):
        if not HAS_PROMETHEUS:
            raise RuntimeError("prometheus-client is not installed")
        self.node = str(node)
        self.registry = registry or CollectorRegistry()
        common = ("node", "model", "mode", "backend", "tracking", "transport")
        self.stage_latency = Histogram(
            "benchmark_stage_latency_ms",
            "Per-frame pipeline stage latency in milliseconds",
            (*common, "stage"),
            buckets=LATENCY_BUCKETS_MS,
            registry=self.registry,
        )
        self.frames = Counter(
            "benchmark_frames_total",
            "Frames observed by status and selected mode",
            (*common, "status", "selected_mode"),
            registry=self.registry,
        )
        self.payload = Histogram(
            "benchmark_payload_bytes",
            "Encoded uplink payload size",
            common,
            buckets=PAYLOAD_BUCKETS_BYTES,
            registry=self.registry,
        )
        self.queue_depth = Gauge(
            "benchmark_grpc_queue_depth",
            "Latest edge gRPC queue depth",
            common,
            registry=self.registry,
        )
        self.pending_feedback = Gauge(
            "benchmark_pending_feedback",
            "Latest frames awaiting cloud feedback",
            common,
            registry=self.registry,
        )
        self.cloud_inflight = Gauge(
            "benchmark_cloud_inflight",
            "Latest cloud inflight value returned to edge",
            common,
            registry=self.registry,
        )
        self.transport_pending = Gauge(
            "benchmark_transport_pending",
            "Latest bounded transport requests awaiting a result",
            common,
            registry=self.registry,
        )
        self.model_switches = Counter(
            "benchmark_model_switch_total",
            "Blend model switches",
            (*common, "selected_mode"),
            registry=self.registry,
        )
        self.drl_decisions = Counter(
            "benchmark_drl_decision_total",
            "DRL decisions by selected mode",
            (*common, "selected_mode"),
            registry=self.registry,
        )
        self.resource_utilization = Gauge(
            "benchmark_resource_utilization_ratio",
            "Non-blocking externally sampled resource utilization in [0,1]",
            ("node", "resource"),
            registry=self.registry,
        )
        self.resource_temperature = Gauge(
            "benchmark_resource_temperature_celsius",
            "Sampled CPU temperature in degrees Celsius",
            ("node", "resource"),
            registry=self.registry,
        )
        self.process_rss = Gauge(
            "benchmark_process_rss_bytes",
            "Resident memory of the benchmark process",
            ("node",),
            registry=self.registry,
        )
        self.run_effective_fps = Gauge(
            "benchmark_run_effective_fps",
            "Final effective FPS for the latest completed benchmark",
            common,
            registry=self.registry,
        )
        self.summary_stage_latency = Gauge(
            "benchmark_summary_stage_latency_ms",
            "Final stage latency statistic from the completed benchmark",
            (*common, "stage", "stat"),
            registry=self.registry,
        )
        self.summary_payload = Gauge(
            "benchmark_summary_payload_bytes",
            "Final payload statistic from the completed benchmark",
            (*common, "stat"),
            registry=self.registry,
        )
        self.run_status = Gauge(
            "benchmark_run_status",
            "Latest benchmark status as a one-hot gauge",
            (*common, "result"),
            registry=self.registry,
        )
    def start_http(self, port: int, address: str = "0.0.0.0") -> None:
        start_http_server(int(port), addr=address, registry=self.registry)

    def observe(self, trace: Any) -> None:
        row = trace.to_dict() if callable(getattr(trace, "to_dict", None)) else trace
        mode = str(row.get("mode", "UNKNOWN")).upper()
        backend = str(row.get("backend", "unknown"))
        status = str(row.get("status", "unknown"))
        attributes = row.get("attributes") or {}
        model = str(attributes.get("model", "unknown"))
        tracking = "on" if bool(attributes.get("tracking", False)) else "off"
        transport = str(attributes.get("transport", "grpc" if (row.get("durations_ms") or {}).get("grpc_rtt") is not None else "local"))
        selected_mode = str(attributes.get("selected_mode", mode)).upper()
        labels = {
            "node": self.node,
            "model": model,
            "mode": mode,
            "backend": backend,
            "tracking": tracking,
            "transport": transport,
        }

        self.frames.labels(
            **labels, status=status, selected_mode=selected_mode
        ).inc()
        for stage, value in (row.get("durations_ms") or {}).items():
            if value is not None:
                self.stage_latency.labels(**labels, stage=str(stage)).observe(float(value))

        payload_bytes = attributes.get("payload_bytes")
        if payload_bytes is not None:
            self.payload.labels(**labels).observe(float(payload_bytes))
        self._set_if_present(
            self.queue_depth,
            labels,
            attributes.get("grpc_queue_depth_after"),
        )
        self._set_if_present(
            self.pending_feedback,
            labels,
            attributes.get("grpc_pending_after"),
        )
        self._set_if_present(
            self.cloud_inflight,
            labels,
            attributes.get("cloud_inflight"),
        )
        self._set_if_present(
            self.transport_pending,
            labels,
            attributes.get("transport_pending_after"),
        )
        if attributes.get("mode_changed"):
            self.model_switches.labels(**labels, selected_mode=selected_mode).inc()
        if attributes.get("decision_made"):
            self.drl_decisions.labels(**labels, selected_mode=selected_mode).inc()

    def observe_summary(self, summary: dict[str, Any]) -> None:
        config = summary.get("config") or {}
        backends = summary.get("backends") or {}
        backend = max(backends, key=backends.get) if backends else "unknown"
        labels = {
            "node": self.node,
            "model": str(config.get("model", "unknown")),
            "mode": str(config.get("mode", "UNKNOWN")).upper(),
            "backend": str(backend),
            "tracking": "on" if bool(config.get("tracking", False)) else "off",
            "transport": str(config.get("transport", "local")),
        }
        fps = summary.get("effective_fps")
        if fps is not None:
            self.run_effective_fps.labels(**labels).set(float(fps))
        for stage, values in (summary.get("stages_ms") or {}).items():
            for stat in ("mean", "p50", "p95", "p99"):
                value = (values or {}).get(stat)
                if value is not None:
                    self.summary_stage_latency.labels(**labels, stage=str(stage), stat=stat).set(float(value))
        for stat in ("mean", "p50", "p95", "p99"):
            value = (summary.get("payload_bytes") or {}).get(stat)
            if value is not None:
                self.summary_payload.labels(**labels, stat=stat).set(float(value))
        if summary.get("accepted"):
            result = "accepted"
        elif not summary.get("execution_valid"):
            result = "execution_failed"
        elif summary.get("artifact_verification") == "failed":
            result = "artifact_failed"
        elif summary.get("quality_verification") == "failed":
            result = "quality_failed"
        else:
            result = "pending_verification"
        self.run_status.labels(**labels, result=result).set(1.0)
    def observe_resource(self, resource: str, value: float) -> None:
        ratio = max(0.0, min(1.0, float(value)))
        self.resource_utilization.labels(node=self.node, resource=str(resource)).set(ratio)

    def observe_resource_sample(self, sample: dict[str, Any]) -> None:
        for key, resource in (
            ("cpu_utilization_ratio", "edge_cpu"),
            ("memory_utilization_ratio", "edge_memory"),
        ):
            value = sample.get(key)
            if value is not None:
                self.observe_resource(resource, float(value))
        temperature = sample.get("temperature_celsius")
        if temperature is not None:
            self.resource_temperature.labels(
                node=self.node, resource="edge_cpu"
            ).set(float(temperature))
        rss = sample.get("process_rss_bytes")
        if rss is not None:
            self.process_rss.labels(node=self.node).set(float(rss))

    @staticmethod
    def _set_if_present(metric: Any, labels: dict[str, str], value: Any) -> None:
        if value is not None:
            metric.labels(**labels).set(float(value))
