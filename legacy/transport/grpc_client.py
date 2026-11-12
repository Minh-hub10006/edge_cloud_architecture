"""CloudClient subclass that extracts optional benchmark timing metadata."""
from __future__ import annotations

import time

import edge_cloud_pb2

from benchmark.cloud.timing_wire import decode_timing_message
from detectors.dfine.tier1_codec import (
    PROJ256_INT8_SHAPE_MARKER,
    MONO_JPEG_SHAPE_MARKER,
    TIER1_INT8_SHAPE_MARKER,
)
from detectors.dfine.stem_codec import (
    CUT_STEM_INT4GZ_SHAPE_MARKER,
    CUT_STEM_INT8_SHAPE_MARKER,
)
from grpc_client import CloudClient


class BenchmarkCloudClient(CloudClient):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._benchmark_enqueued_ns: dict[str, int] = {}
        self._benchmark_send_metrics: dict[str, dict[str, float]] = {}

    async def poll_policy_updates(self):
        """Disable production policy downloads during an isolated benchmark."""
        return None

    def push_feature(self, data, frame_id, layout: str = "fdraas"):
        self._benchmark_enqueued_ns[str(frame_id)] = time.perf_counter_ns()
        return super().push_feature(data, frame_id, layout=layout)

    async def _feature_generator(self):
        """Production generator with queue and packet-build timing added."""
        while True:
            data, frame_id, layout = await self.feature_queue.get()
            fid = str(frame_id)
            dequeued_ns = time.perf_counter_ns()
            enqueued_ns = self._benchmark_enqueued_ns.pop(fid, dequeued_ns)
            if layout == "heartbeat":
                payload = b""
                shape = [-2.0]
            elif layout == "tier1_int8":
                payload = data if isinstance(data, bytes) else bytes(data)
                shape = [TIER1_INT8_SHAPE_MARKER]
            elif layout == "proj256_int8":
                payload = data if isinstance(data, bytes) else bytes(data)
                shape = [PROJ256_INT8_SHAPE_MARKER]
            elif layout == "cut_stem_int8":
                payload = data if isinstance(data, bytes) else bytes(data)
                shape = [CUT_STEM_INT8_SHAPE_MARKER]
            elif layout == "cut_stem_int4gz":
                payload = data if isinstance(data, bytes) else bytes(data)
                shape = [CUT_STEM_INT4GZ_SHAPE_MARKER]
            elif layout == "mono_jpeg":
                payload = data if isinstance(data, bytes) else bytes(data)
                shape = [MONO_JPEG_SHAPE_MARKER]
            elif isinstance(data, bytes):
                payload = data
                shape = [-1.0]
            else:
                payload = data.tobytes()
                shape = list(data.shape)
            packet_started_ns = time.perf_counter_ns()
            packet = edge_cloud_pb2.FeaturePacket(
                frame_id=fid,
                node_id=str(self.node_id),
                feature_tensor=payload,
                shape=shape,
            )
            packet_finished_ns = time.perf_counter_ns()
            self._benchmark_send_metrics[fid] = {
                "edge_queue_wait_ms": (dequeued_ns - enqueued_ns) / 1_000_000.0,
                "protobuf_pack_ms":
                    (packet_finished_ns - packet_started_ns) / 1_000_000.0,
            }
            yield packet
            self.feature_queue.task_done()

    def _on_feedback(self, response):
        callback_started_ns = time.perf_counter_ns()
        timings, original_message = decode_timing_message(response.server_message)
        entry = super()._on_feedback(response)
        if timings:
            clean = {
                key: value
                for key, value in timings.items()
                if key
                in {
                    "cloud_total_ms",
                    "cloud_decode_ms",
                    "cloud_inference_ms",
                    "cloud_timing_scope",
                    "cloud_layout",
                    "cloud_decoder_reported_ms",
                    "cloud_queue_ms",
                    "cloud_service_ms",
                    "cloud_providers_b",
                    "cloud_providers_c",
                    "cloud_providers_mono",
                }
            }
            entry.update(clean)
            self._feedback_by_frame[str(response.frame_id)].update(clean)
            self.latest_server_message = original_message
        fid = str(response.frame_id)
        entry.update(self._benchmark_send_metrics.pop(fid, {}))
        entry["feedback_deserialize_ms"] = (
            time.perf_counter_ns() - callback_started_ns
        ) / 1_000_000.0
        self._feedback_by_frame[fid].update(entry)
        return entry
