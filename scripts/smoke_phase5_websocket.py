"""One-stream Cloud HTTP -> ResultHub/ByteTrack -> real ASGI WebSocket smoke."""

from __future__ import annotations

import sys
from pathlib import Path
from time import time_ns

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cloud_core import CloudByteTrack, ResultHub, result_to_payload
from contracts import Box, DetectionResult, FrameIdentity, ModelMode
from web.cloud_app import create_app


def make_result(frame_id: int) -> DetectionResult:
    return DetectionResult(
        frame=FrameIdentity(
            node_id="edge-phase5-smoke", camera_id="traffic-01",
            stream_id="preview", stream_epoch=1, source_frame_id=frame_id,
            capture_utc_ns=1_790_000_000_000_000_000 + frame_id * 66_666_667,
            source_pts=frame_id * 66_666_667,
        ),
        model_mode=ModelMode.DFINE_CLOUD,
        model_id="phase5-smoke-detector", decision_id="phase5-window-0",
        boxes=(Box(0.2, 0.25, 0.55, 0.7, 0.95, 2, "car"),),
        inference_ms=1.0, produced_utc_ns=time_ns(),
        request_id=f"phase5-{frame_id}", preview_pts=frame_id * 66_666_667,
    )


def main() -> None:
    import numpy  # noqa: F401
    import torch
    import ultralytics
    from starlette.testclient import TestClient

    tracker = CloudByteTrack(frame_rate=15)
    # Exercise a real ASGI WebSocket without requiring a port or external client.
    hub = ResultHub(tracker=tracker)
    app = create_app(hub)
    ids: list[int] = []
    received = 0
    ws_path = "/v1/streams/edge-phase5-smoke/traffic-01/preview/1/metadata"

    with TestClient(app) as client:
        health = client.get("/healthz")
        if health.status_code != 200:
            raise AssertionError(f"health endpoint failed: {health.status_code}")
        with client.websocket_connect(ws_path) as socket:
            for frame_id in (0, 1, 2, 4, 5):
                response = client.post(
                    "/v1/results", json=result_to_payload(make_result(frame_id)),
                )
                if response.status_code != 200:
                    raise AssertionError(f"result ingest failed: {response.status_code} {response.text}")
                event = socket.receive_json()
                received += 1
                if event["source_frame_id"] != frame_id:
                    raise AssertionError(f"wrong frame order: {event['source_frame_id']} != {frame_id}")
                ids.append(event["boxes"][0]["track_id"])
                if ids[-1] is None:
                    raise AssertionError("WebSocket metadata is missing a ByteTrack ID")

            duplicate = client.post(
                "/v1/results", json=result_to_payload(make_result(5)),
            )
            stale = client.post(
                "/v1/results", json=result_to_payload(make_result(3)),
            )

    if received != 5 or len(set(ids)) != 1:
        raise AssertionError(f"expected five events with one stable track ID, got {received}/{ids}")
    if duplicate.status_code != 409 or duplicate.json()["status"] != "duplicate":
        raise AssertionError("duplicate frame was not rejected")
    if stale.status_code != 409 or stale.json()["status"] != "out_of_order":
        raise AssertionError("late frame was not rejected")
    print({
        "status": "PASS", "http_ingest": 5, "websocket_events": received,
        "stable_track_ids": ids, "duplicate": duplicate.json()["status"],
        "late_result": stale.json()["status"], "torch": torch.__version__,
        "ultralytics": ultralytics.__version__, "numpy": numpy.__version__,
    })


if __name__ == "__main__":
    main()
