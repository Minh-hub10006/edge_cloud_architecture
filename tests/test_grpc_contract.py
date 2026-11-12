from __future__ import annotations

import asyncio
from dataclasses import replace
from time import time_ns
import unittest

import grpc
import numpy as np

from cloud_core import CloudFrameCache, DecodedMediaFrame, ResultHub
from cloud_core.grpc_service import CloudDataPlaneServicer
from contracts import Box, CloudInferenceTicket, DetectionResult, FrameIdentity, MediaFrameRef, ModelMode
from contracts.grpc_codec import (
    frame_to_proto, reference_from_proto, reference_to_proto, result_from_proto, result_to_proto,
    ticket_from_proto, ticket_to_proto,
)
from contracts.proto import edge_video_platform_pb2 as pb
from contracts.proto import edge_video_platform_pb2_grpc as pb_grpc


class GrpcContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = FrameIdentity(
            node_id="edge-246", camera_id="traffic-01", stream_id="preview",
            stream_epoch=3, source_frame_id=12,
            capture_utc_ns=1_790_000_000_000_000_000, source_pts=800_000_000,
        )

    def test_detection_result_round_trip_preserves_identity_and_optional_fields(self) -> None:
        expected = DetectionResult(
            frame=self.frame, model_mode=ModelMode.YOLO_EDGE, model_id="yolo-320-int8",
            decision_id="window-4", boxes=(Box(
                0.1, 0.2, 0.8, 0.9, 0.97, 2, "car", track_id=7,
            ),), inference_ms=4.5, produced_utc_ns=1_790_000_000_005_000_000,
            request_id="req-12", preview_pts=733_333_333,
            attributes={"placement": "edge", "hardware": True},
        )
        restored = result_from_proto(result_to_proto(expected))
        self.assertEqual(restored.frame, expected.frame)
        self.assertEqual(restored.model_mode, expected.model_mode)
        self.assertEqual(restored.model_id, expected.model_id)
        self.assertEqual(restored.request_id, expected.request_id)
        self.assertEqual(restored.attributes, expected.attributes)
        self.assertEqual(restored.boxes[0].class_id, expected.boxes[0].class_id)
        self.assertEqual(restored.boxes[0].class_name, expected.boxes[0].class_name)
        for field in ("left", "top", "right", "bottom", "confidence"):
            self.assertAlmostEqual(getattr(restored.boxes[0], field), getattr(expected.boxes[0], field), places=6)

    def test_reference_and_cloud_ticket_round_trip(self) -> None:
        reference = MediaFrameRef(
            frame=self.frame, encoded_seq=11, media_pts_ns=733_333_333,
            submitted_utc_ns=1_790_000_000_006_000_000,
        )
        self.assertEqual(reference_from_proto(reference_to_proto(reference)), reference)
        ticket = CloudInferenceTicket(
            frame=self.frame, model_mode=ModelMode.DFINE_CLOUD, model_id="dfine",
            decision_id="window-4", preview_pts=733_333_333,
            deadline_utc_ns=1_790_000_000_500_000_000, request_id="req-12",
        )
        self.assertEqual(ticket_from_proto(ticket_to_proto(ticket)), ticket)

    def test_ticket_carries_jpeg_and_rejects_non_jpeg_or_oversize(self) -> None:
        base = dict(frame=self.frame, model_mode=ModelMode.DFINEX_CLOUD, model_id="dx", decision_id="w",
                    preview_pts=1, deadline_utc_ns=1_790_000_000_500_000_000)
        ticket = CloudInferenceTicket(**base, jpeg=b"\xff\xd8 pixels \xff\xd9")
        self.assertEqual(ticket_from_proto(ticket_to_proto(ticket)), ticket)
        self.assertEqual(ticket_from_proto(ticket_to_proto(CloudInferenceTicket(**base))).jpeg, b"")
        for bad in (b"\x89PNG", b"\xff\xd8" + b"x" * (1 << 20)):
            with self.assertRaisesRegex(ValueError, "jpeg"):
                CloudInferenceTicket(**base, jpeg=bad)


class CloudGrpcMultiNodeTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        from cloud_core.tracking import CloudByteTrack
        await asyncio.to_thread(CloudByteTrack().prepare)

    async def test_cloud_ticket_runs_on_cached_frame_and_reports_loaded_model_id(self) -> None:
        hub = ResultHub()
        cache = CloudFrameCache()
        now_ns = time_ns()
        frame = FrameIdentity(
            node_id="edge-246", camera_id="traffic-01", stream_id="preview",
            stream_epoch=0, source_frame_id=4,
            capture_utc_ns=now_ns, source_pts=200_000_000,
        )
        reference = MediaFrameRef(
            frame=frame, encoded_seq=3, media_pts_ns=200_000_000,
            submitted_utc_ns=now_ns,
        )
        await cache.announce(reference)
        await cache.ingest(DecodedMediaFrame(
            stream_key=frame.stream_key, encoded_seq=3,
            media_pts_ns=200_000_000, pixels="decoded-frame",
            received_utc_ns=now_ns,
        ))

        class FakeDFINE:
            model_id = "dfine_full_traffic"
            runtime = "cuda"

            def infer(self, pixels):
                if pixels != "decoded-frame":
                    raise AssertionError("D-FINE must receive the cached RTSP frame")
                return [Box(0.1, 0.2, 0.4, 0.5, 0.9, 2, "car")], 3.0

        servicer = CloudDataPlaneServicer(
            hub=hub, frame_cache=cache, frame_taps=object(),
            dfine_adapter=FakeDFINE(),
        )
        reply = await servicer.SubmitCloudInference(pb.CloudInferenceTicket(
            schema_version=2, frame=frame_to_proto(frame),
            model_mode=pb.DFINE_CLOUD, model_id="dfine",
            decision_id="phase7-window", preview_pts=200_000_000,
            deadline_utc_ns=now_ns + 1_000_000_000,
            request_id="cloud-frame-4",
        ), None)

        self.assertEqual(reply.status, "accepted")
        result = result_from_proto(reply.result)
        self.assertEqual(result.model_id, "dfine_full_traffic")
        self.assertEqual(result.attributes["requested_model_id"], "dfine")
        self.assertEqual(result.frame, frame)
        self.assertEqual(result.boxes[0].class_name, "car")

    async def test_jpeg_ticket_skips_frame_cache_and_bad_jpeg_is_rejected(self) -> None:
        import cv2
        now_ns = time_ns()
        frame = FrameIdentity(node_id="edge-247", camera_id="traffic-01", stream_id="s1", stream_epoch=0,
                              source_frame_id=9, capture_utc_ns=now_ns, source_pts=600_000_000)
        image = np.zeros((640, 640, 3), np.uint8)
        image[100:300, 200:400] = (0, 0, 255)
        jpeg = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()
        seen = []

        class FakeDfineX:
            model_id = "dfine_x_candidate_test"
            runtime = "cuda"

            def infer(self, pixels):
                seen.append(pixels)
                return [Box(0.1, 0.2, 0.4, 0.5, 0.9, 2, "car")], 3.0

        class NoCache:
            async def wait_for(self, ticket):
                raise AssertionError("a JPEG ticket must not wait for the RTSP frame cache")

        servicer = CloudDataPlaneServicer(hub=ResultHub(), frame_cache=NoCache(), frame_taps=object(),
                                          cloud_adapters={ModelMode.DFINEX_CLOUD: FakeDfineX()}, preview_fps=15)

        def ticket(data, frame_id):
            return pb.CloudInferenceTicket(
                schema_version=2, frame=frame_to_proto(replace(frame, source_frame_id=frame_id)),
                model_mode=pb.DFINEX_CLOUD, model_id="dfine_x_candidate_test", decision_id="w",
                preview_pts=600_000_000, deadline_utc_ns=now_ns + 1_000_000_000, request_id=f"r{frame_id}", jpeg=data)

        reply = await servicer.SubmitCloudInference(ticket(jpeg, 9), None)
        self.assertEqual(reply.status, "accepted")
        result = result_from_proto(reply.result)
        self.assertEqual(seen[0].shape, (640, 640, 3))
        self.assertGreater(int(seen[0][200, 300, 2]), 200)  # red block survived the JPEG round trip
        self.assertEqual((result.attributes["frame_source"], result.attributes["media_fps"],
                          result.attributes["media_encoded_seq"]), ("jpeg", 15, 9))  # 0.6 s at 15 fps
        reply = await servicer.SubmitCloudInference(ticket(b"\xff\xd8 not a jpeg", 10), None)
        self.assertEqual(reply.status, "invalid_jpeg")

    async def test_registered_second_cloud_mode_uses_same_cache_and_service(self) -> None:
        hub = ResultHub()
        cache = CloudFrameCache()
        now_ns = time_ns()
        frame = FrameIdentity(
            node_id="edge-b", camera_id="cam-01", stream_id="main",
            stream_epoch=2, source_frame_id=1, capture_utc_ns=now_ns,
            source_pts=100_000_000,
        )
        reference = MediaFrameRef(frame, 1, 100_000_000, now_ns)
        await cache.announce(reference)
        await cache.ingest(DecodedMediaFrame(frame.stream_key, 1, 100_000_000, "decoded-b", now_ns))

        class FakeSecondCloudModel:
            model_id = "candidate-cloud"
            runtime = "fake"

            def infer(self, pixels):
                if pixels != "decoded-b":
                    raise AssertionError("Cloud model did not receive its decoded RTSP frame")
                return [Box(0.1, 0.1, 0.2, 0.2, 0.8, 2, "car")], 1.0

        servicer = CloudDataPlaneServicer(
            hub=hub, frame_cache=cache, frame_taps=object(),
            cloud_adapters={ModelMode.RTDETR_CLOUD: FakeSecondCloudModel()},
        )
        reply = await servicer.SubmitCloudInference(pb.CloudInferenceTicket(
            schema_version=2, frame=frame_to_proto(frame),
            model_mode=pb.RTDETR_CLOUD, model_id="requested",
            decision_id="test-window", preview_pts=100_000_000,
            deadline_utc_ns=now_ns + 1_000_000_000,
        ), None)
        self.assertEqual(reply.status, "accepted")
        self.assertEqual(result_from_proto(reply.result).model_id, "candidate-cloud")

    async def test_one_cloud_grpc_service_handles_three_edges_concurrently(self) -> None:
        hub = ResultHub()
        servicer = CloudDataPlaneServicer(
            hub=hub, frame_cache=CloudFrameCache(),
            frame_taps=object(), dfine_adapter=None,
        )
        server = grpc.aio.server()
        pb_grpc.add_EdgeCloudDataPlaneServicer_to_server(servicer, server)
        port = server.add_insecure_port("127.0.0.1:0")
        await server.start()
        channel = grpc.aio.insecure_channel(f"127.0.0.1:{port}")
        stub = pb_grpc.EdgeCloudDataPlaneStub(channel)
        subscriptions = {
            node_id: hub.subscribe((node_id, "cam-01", "main", 0))
            for node_id in ("edge-a", "edge-b", "edge-c")
        }
        try:
            async def send_node(node_id: str) -> None:
                for frame_id in (1, 2):
                    frame = FrameIdentity(
                        node_id=node_id, camera_id="cam-01", stream_id="main",
                        stream_epoch=0, source_frame_id=frame_id,
                        capture_utc_ns=1_790_000_000_000_000_000 + frame_id,
                        source_pts=frame_id * 33_333_333,
                    )
                    result = DetectionResult(
                        frame=frame, model_mode=ModelMode.YOLO_EDGE,
                        model_id="yolo-int8", decision_id="fixed",
                        boxes=(Box(0.1, 0.1, 0.3, 0.4, 0.9, 2, "car"),),
                        inference_ms=2.0, produced_utc_ns=frame.capture_utc_ns + 1,
                        preview_pts=frame.source_pts,
                    )
                    ack = await stub.SubmitDetectionResult(
                        result_to_proto(result), timeout=2,
                    )
                    self.assertEqual(ack.status, "accepted")
                    _subscriber_id, queue = subscriptions[node_id]
                    event = await asyncio.wait_for(queue.get(), timeout=1)
                    self.assertEqual(event["boxes"][0]["track_id"], 1)
                    queue.task_done()
            await asyncio.gather(*(send_node(node_id) for node_id in subscriptions))
        finally:
            for subscriber_id, _queue in subscriptions.values():
                hub.unsubscribe(subscriber_id)
            await channel.close()
            await server.stop(grace=0)


if __name__ == "__main__":
    unittest.main()


class CloudProbeTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        import cv2
        self.now = time_ns()
        self.frame = FrameIdentity(node_id="edge-246", camera_id="c", stream_id="s", stream_epoch=0,
                                   source_frame_id=3, capture_utc_ns=self.now, source_pts=0)
        self.jpeg = cv2.imencode(".jpg", np.zeros((64, 64, 3), np.uint8))[1].tobytes()

        class Reference:
            model_id = "dx"
            runtime = "cuda"
            calls = 0

            def infer(self, pixels):
                Reference.calls += 1
                return [Box(0.1, 0.2, 0.4, 0.5, 0.9, 2, "car")], 3.0

        class NoHub:
            async def submit(self, *args, **kwargs):
                raise AssertionError("a probe must not reach the ResultHub")

        self.reference = Reference
        self.servicer = CloudDataPlaneServicer(hub=NoHub(), frame_cache=object(), frame_taps=object(),
                                               cloud_adapters={ModelMode.DFINEX_CLOUD: Reference()},
                                               max_cloud_inflight=1)

    def ticket(self, jpeg=None):
        return CloudInferenceTicket(frame=self.frame, model_mode=ModelMode.DFINEX_CLOUD, model_id="dx",
                                    decision_id="probe/w1", preview_pts=0, deadline_utc_ns=self.now + 10**9,
                                    request_id="p1", jpeg=self.jpeg if jpeg is None else jpeg)

    async def test_probe_returns_reference_boxes_outside_the_hub(self):
        status, result, _ = await self.servicer.handle_cloud_ticket(self.ticket())
        self.assertEqual(status, "accepted")
        self.assertTrue(result.attributes["probe"])
        self.assertEqual(result.boxes[0].class_name, "car")

    async def test_probe_never_waits_behind_real_work(self):
        await self.servicer._slots.acquire()  # the only GPU slot is busy with a real ticket
        try:
            status, result, _ = await self.servicer.handle_cloud_ticket(self.ticket())
        finally:
            self.servicer._slots.release()
        self.assertEqual((status, result, self.reference.calls), ("probe_busy", None, 0))

    async def test_probe_without_jpeg_is_rejected(self):
        status, _r, _d = await self.servicer.handle_cloud_ticket(self.ticket(jpeg=b""))
        self.assertEqual(status, "invalid")
