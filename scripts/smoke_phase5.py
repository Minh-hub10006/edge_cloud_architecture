"""Exercise the real Cloud ByteTrack and ResultHub with three interleaved streams."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
import sys
from time import time_ns

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cloud_core import CloudByteTrack, ResultHub, SubmitStatus
from contracts import Box, DetectionResult, FrameIdentity, ModelMode


def make_result(node_id: str, frame_id: int) -> DetectionResult:
    return DetectionResult(
        frame=FrameIdentity(
            node_id=node_id,
            camera_id="traffic-01",
            stream_id="preview",
            stream_epoch=1,
            source_frame_id=frame_id,
            capture_utc_ns=1_790_000_000_000_000_000 + frame_id * 66_666_667,
            source_pts=frame_id * 66_666_667,
        ),
        model_mode=ModelMode.DFINE_CLOUD,
        model_id="phase5-smoke-detector",
        decision_id="phase5-smoke-window-0",
        boxes=(Box(0.20, 0.25, 0.55, 0.70, 0.95, 2, "car"),),
        inference_ms=1.0,
        produced_utc_ns=time_ns(),
        request_id=f"{node_id}-{frame_id}",
        preview_pts=frame_id * 66_666_667,
    )


async def run() -> dict[str, object]:
    tracker = CloudByteTrack(frame_rate=15)
    hub = ResultHub(tracker=tracker, subscriber_queue_size=32)
    stream_nodes = ("edge-smoke-01", "edge-smoke-02", "edge-smoke-03")
    queues = {
        node: hub.subscribe((node, "traffic-01", "preview", 1))[1]
        for node in stream_nodes
    }
    track_history: dict[str, list[int]] = {node: [] for node in stream_nodes}

    # Interleave streams as frames from multiple Edge nodes arrive concurrently.
    submitted_frame_ids = (0, 1, 2, 4, 5)
    for frame_id in submitted_frame_ids:
        for node in stream_nodes:
            submitted = await hub.submit(make_result(node, frame_id))
            if submitted.status is not SubmitStatus.ACCEPTED or submitted.result is None:
                raise AssertionError(f"unexpected submit outcome for {node}/{frame_id}: {submitted.status}")
            track_id = submitted.result.boxes[0].track_id
            if track_id is None:
                raise AssertionError(f"ByteTrack did not assign an ID for {node}/{frame_id}")
            track_history[node].append(track_id)

    if tracker.stream_count != len(stream_nodes):
        raise AssertionError(f"expected 3 isolated trackers, got {tracker.stream_count}")
    if any(len(set(ids)) != 1 for ids in track_history.values()):
        raise AssertionError(f"track IDs changed for stable boxes: {track_history}")
    if any(queue.qsize() != 5 for queue in queues.values()):
        raise AssertionError("each stream subscriber must receive its five tracked results")

    # Duplicate and late results must not update/reverse the tracker's state.
    duplicate = await hub.submit(make_result(stream_nodes[0], 5))
    late = await hub.submit(make_result(stream_nodes[0], 3))
    if duplicate.status is not SubmitStatus.DUPLICATE:
        raise AssertionError(f"duplicate frame was not rejected: {duplicate.status}")
    if late.status is not SubmitStatus.OUT_OF_ORDER:
        raise AssertionError(f"late frame was not rejected: {late.status}")

    # Serialize one tracked result and verify fields consumed by WebLive.
    payload = queues[stream_nodes[0]].get_nowait()
    result_to_payload_keys = {
        "schema_version", "node_id", "camera_id", "stream_id", "stream_epoch",
        "source_frame_id", "capture_utc_ns", "source_pts", "model_mode",
        "model_id", "decision_id", "request_id", "inference_ms",
        "produced_utc_ns", "boxes", "attributes",
    }
    json.dumps(payload, allow_nan=False)
    if not result_to_payload_keys.issubset(payload):
        raise AssertionError(f"metadata payload lacks fields: {result_to_payload_keys - payload.keys()}")
    if payload["boxes"][0]["track_id"] is None:
        raise AssertionError("serialized metadata is missing track_id")

    return {
        "status": "PASS",
        "streams_tested": len(stream_nodes),
        "frames_per_stream": 5,
        "tracker_instances": tracker.stream_count,
        "stable_track_ids": track_history,
        "duplicate_rejected": duplicate.status.value,
        "out_of_order_rejected": late.status.value,
        "metadata_json": "valid",
        "websocket_events_received": {node: 5 for node in stream_nodes},
    }


if __name__ == "__main__":
    import numpy  # noqa: F401 - fail early if runtime dependency is missing
    import torch
    import ultralytics

    output = asyncio.run(run())
    output["python_torch"] = torch.__version__
    output["ultralytics"] = ultralytics.__version__
    output["numpy"] = numpy.__version__
    print(json.dumps(output, indent=2, sort_keys=True))
