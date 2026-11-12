from __future__ import annotations

import asyncio
from contextlib import redirect_stdout
import io
import json
from time import time_ns
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from contracts import FrameIdentity, MediaFrameRef, ModelMode
from contracts.errors import CloudInferenceRejected
from edge_agent import BoundedLatestQueue, CapturedFrame
from edge_agent.phase7 import _run
from inference.edge_worker import run_edge_inference_worker
from inference.frame_refs import MediaReferenceMap
from media.pipeline import MediaCadence


def frame(index):
    return CapturedFrame(
        FrameIdentity("edge-test", "cam", "main", 3, index, time_ns(), index),
        pixels=None, capture_mono_ns=index * 33_333_333,
    )


class RuntimeRegressionTests(unittest.IsolatedAsyncioTestCase):
    async def test_cloud_admits_only_media_frames_and_bounds_shutdown_drain(self):
        closed = asyncio.Event()
        published = []
        tickets = []
        wait_until_close = True
        closed_at_reply = []

        class Source:
            def __init__(self, *args, **kwargs):
                pass

            async def __aiter__(self):
                for index in range(12):
                    yield frame(index)
                    await asyncio.sleep(0.002)

        class Publisher:
            def __init__(self, **kwargs):
                self.output = kwargs["output"]
                self.callback = kwargs["on_frame_submitted"]
                self.frames_submitted = 0
                self.padding_frames_submitted = 0

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                await self.close()

            async def close(self):
                closed.set()

            async def publish(self, item):
                seq = self.frames_submitted
                self.frames_submitted += 1
                published.append(item.identity)
                await self.callback(MediaFrameRef(item.identity, seq, seq * 66_666_667, time_ns()))

            async def publish_padding(self, item):
                self.padding_frames_submitted += 1

        class Client:
            async def start(self):
                pass

            async def close(self):
                pass

            async def publish_media_reference(self, ref):
                return "accepted"

            async def submit_cloud_ticket(self, ticket):
                tickets.append(ticket)
                if wait_until_close:
                    await asyncio.wait_for(closed.wait(), timeout=0.5)
                closed_at_reply.append(closed.is_set())
                return SimpleNamespace(boxes=[])

        args = SimpleNamespace(
            mode="dfine_cloud", node_id="edge-test", camera_id="cam", stream_id="main",
            stream_epoch=3, width=64, height=48, preview_fps=15, bitrate_kbps=300,
            video="synthetic", frames=12, media_queue=16, analytics_queue=16,
            reference_capacity=32, yolo_dlc="", yolo_imgsz=320, yolo_layout="nchw",
            confidence=0.25, transport="kafka", kafka_bootstrap="unused",
            kafka_max_pending=8, rpc_timeout=1, rtsp_base="rtsp://localhost:8554",
            decision_id="test", cloud_model_id="dfine-test", reference_wait=0.01,
            ticket_ttl=1,
            shutdown_padding_frames=2, shutdown_drain_s=0.1,
        )
        output = io.StringIO()
        with patch("edge_agent.phase7.OpenCVFrameSource", Source), \
             patch("edge_agent.phase7.FFmpegH264Publisher", Publisher), \
             patch("contracts.kafka_transport.KafkaCloudClient", return_value=Client()), \
             redirect_stdout(output):
            await asyncio.wait_for(_run(args), timeout=2)
        stats = json.loads(output.getvalue())
        self.assertGreater(len(tickets), 0)
        self.assertLess(len(published), 12)
        self.assertTrue(all(ticket.frame in published for ticket in tickets))
        self.assertEqual([ticket.frame.source_frame_id for ticket in tickets], sorted(ticket.frame.source_frame_id for ticket in tickets))
        self.assertEqual(len(tickets) + stats["analytics_queue_dropped"], len(published))
        self.assertEqual(stats["inference"]["frames_without_media_ref"], 0)
        self.assertEqual(stats["inference"]["transport_errors"], 0)
        self.assertEqual(stats["media_padding_encoded"], 2)
        self.assertTrue(all(closed_at_reply))

        # When Cloud replies while RTSP is still published, drain tickets
        # before closing the video path.
        closed = asyncio.Event()
        published.clear()
        tickets.clear()
        closed_at_reply.clear()
        wait_until_close = False
        output = io.StringIO()
        with patch("edge_agent.phase7.OpenCVFrameSource", Source), \
             patch("edge_agent.phase7.FFmpegH264Publisher", Publisher), \
             patch("contracts.kafka_transport.KafkaCloudClient", return_value=Client()), \
             redirect_stdout(output):
            await asyncio.wait_for(_run(args), timeout=2)
        stats = json.loads(output.getvalue())
        self.assertTrue(closed.is_set())
        self.assertTrue(tickets)
        self.assertTrue(all(not was_closed for was_closed in closed_at_reply))
        self.assertEqual(stats["inference"]["cloud_rejections"], 0)

    async def test_rejection_and_transport_timeout_are_separate(self):
        queue = BoundedLatestQueue(3)
        refs = MediaReferenceMap()
        for i in range(3):
            item = frame(i)
            queue.put_latest(item)
            await refs.publish(MediaFrameRef(item.identity, i, i, time_ns()))

        class Client:
            async def submit_cloud_ticket(self, ticket):
                if ticket.frame.source_frame_id == 0:
                    raise CloudInferenceRejected("pts_mismatch", "test mismatch")
                if ticket.frame.source_frame_id == 1:
                    raise TimeoutError("broker reply timeout")
                return SimpleNamespace(boxes=[object()])

        stop = asyncio.Event()
        stop.set()
        with redirect_stdout(io.StringIO()):
            result = await run_edge_inference_worker(
                queue, mode=ModelMode.DFINE_CLOUD, client=Client(), references=refs, stop=stop, latest_only=False,
            )
        self.assertEqual(result["frames_inferred"], 1)
        self.assertEqual(result["cloud_rejection_reasons"], {"pts_mismatch": 1})
        self.assertEqual(result["transport_error_reasons"], {"TimeoutError": 1})
        self.assertEqual(result["cloud_rejections"], 1)
        self.assertEqual(result["transport_errors"], 1)

    async def test_monitor_callback_failure_does_not_fail_inference(self):
        queue, refs = BoundedLatestQueue(2), MediaReferenceMap()
        item = frame(4)
        queue.put_latest(item)
        await refs.publish(MediaFrameRef(item.identity, 4, 400, time_ns()))

        class Client:
            async def submit_cloud_ticket(self, ticket):
                return SimpleNamespace(boxes=[object()], attributes={})

        def broken_monitor(_frame, _measurement):
            raise OSError("disk full")

        stop = asyncio.Event()
        stop.set()
        with redirect_stdout(io.StringIO()):
            result = await run_edge_inference_worker(
                queue, mode=ModelMode.DFINE_CLOUD, client=Client(), references=refs,
                stop=stop, on_observation=broken_monitor,
            )
        self.assertEqual(result["frames_inferred"], 1)
        self.assertEqual(result["detections_total"], 1)
        self.assertEqual(result["monitor_errors"], 1)

    def test_preview_cadence_does_not_accumulate_capture_jitter(self):
        cadence = MediaCadence(15)
        selected = sum(cadence.accept(round(i * 1e9 / 30) + (i % 7) * 100_000)
                       for i in range(300))
        self.assertGreaterEqual(selected, 149)
        self.assertLessEqual(selected, 151)
