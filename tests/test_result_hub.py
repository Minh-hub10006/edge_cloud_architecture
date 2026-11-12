from __future__ import annotations

import asyncio
from dataclasses import replace
import importlib.util
from time import time_ns
import unittest

from cloud_core import CloudByteTrack, ResultHub, SubmitStatus, result_to_json
from contracts import Box, DetectionResult, FrameIdentity, ModelMode
from web import serve_metadata_websocket


class FakeTracker:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, str, int]] = []
        self.resets: list[tuple[str, str, str, int]] = []

    def update(self, result: DetectionResult) -> DetectionResult:
        self.calls.append(result.frame.stream_key)
        boxes = tuple(replace(box, track_id=17) for box in result.boxes)
        return replace(result, boxes=boxes)

    def reset(self, stream_key: tuple[str, str, str, int]) -> None:
        self.resets.append(stream_key)


def make_result(frame_id: int, *, node: str = "edge-01", epoch: int = 0) -> DetectionResult:
    return DetectionResult(
        frame=FrameIdentity(
            node_id=node,
            camera_id="traffic-01",
            stream_id="preview",
            stream_epoch=epoch,
            source_frame_id=frame_id,
            capture_utc_ns=1_790_000_000_000_000_000 + frame_id,
            source_pts=frame_id * 1_000,
        ),
        model_mode=ModelMode.DFINE_CLOUD,
        model_id="dfine-cloud",
        decision_id="window-0",
        boxes=(Box(0.1, 0.2, 0.4, 0.6, 0.9, 2, "car"),),
        inference_ms=12.5,
        produced_utc_ns=time_ns(),
        request_id=f"request-{frame_id}",
        preview_pts=frame_id * 66_666_667,
    )


class ResultHubTests(unittest.IsolatedAsyncioTestCase):
    async def test_stream_status_lists_recent_streams_per_node(self) -> None:
        hub = ResultHub(tracker=FakeTracker())
        hub.status_capacity = 2
        for frame_id in range(3):
            await hub.submit(make_result(frame_id, node="edge-246", epoch=1))
        await hub.submit(make_result(0, node="edge-252", epoch=5))
        await hub.submit(make_result(1, node="edge-246", epoch=1))  # duplicate: not counted
        rows = hub.stream_status()
        self.assertEqual([(r["node_id"], r["stream_epoch"]) for r in rows], [("edge-252", 5), ("edge-246", 1)])
        self.assertEqual(rows[1]["results"], 3)
        self.assertEqual(rows[1]["source_frame_id"], 2)
        self.assertEqual((rows[1]["model_mode"], rows[1]["boxes"]), ("dfine_cloud", 1))
        self.assertIsNone(rows[0]["result_fps"])  # one result: no rate yet
        await hub.submit(make_result(0, node="edge-247", epoch=0))  # capacity 2 evicts the oldest
        self.assertEqual([r["node_id"] for r in hub.stream_status()], ["edge-247", "edge-252"])

    async def test_tracks_and_publishes_json_metadata(self) -> None:
        tracker = FakeTracker()
        hub = ResultHub(tracker=tracker)
        _, queue = hub.subscribe(make_result(0).frame.stream_key)

        outcome = await hub.submit(make_result(0))

        self.assertEqual(outcome.status, SubmitStatus.ACCEPTED)
        self.assertEqual(outcome.result.boxes[0].track_id, 17)
        event = queue.get_nowait()
        self.assertEqual(event["source_frame_id"], 0)
        self.assertEqual(event["boxes"][0]["track_id"], 17)
        self.assertIn('"stream_epoch":0', result_to_json(outcome.result))

    async def test_calibrated_confidence_is_preserved_for_browser_metadata(self) -> None:
        hub = ResultHub(tracker=FakeTracker())
        result = make_result(0)
        calibrated_box = replace(result.boxes[0], calibrated_confidence=0.73)
        result = replace(
            result,
            boxes=(calibrated_box,),
            confidence_calibration_id="traffic-val-v1",
        )

        outcome = await hub.submit(result)

        serialized = result_to_json(outcome.result)
        self.assertIn('"calibrated_confidence":0.73', serialized)
        self.assertIn('"confidence_calibration_id":"traffic-val-v1"', serialized)

    async def test_duplicate_and_out_of_order_frames_do_not_rewind_tracker(self) -> None:
        tracker = FakeTracker()
        hub = ResultHub(tracker=tracker)

        self.assertEqual((await hub.submit(make_result(4))).status, SubmitStatus.ACCEPTED)
        self.assertEqual((await hub.submit(make_result(4))).status, SubmitStatus.DUPLICATE)
        self.assertEqual((await hub.submit(make_result(3))).status, SubmitStatus.OUT_OF_ORDER)
        self.assertEqual(len(tracker.calls), 1)

    async def test_stream_epoch_and_node_have_independent_ordering(self) -> None:
        tracker = FakeTracker()
        hub = ResultHub(tracker=tracker)

        for result in (
            make_result(2, node="edge-01", epoch=0),
            make_result(2, node="edge-02", epoch=0),
            make_result(2, node="edge-01", epoch=1),
        ):
            self.assertEqual((await hub.submit(result)).status, SubmitStatus.ACCEPTED)
        self.assertEqual(len(tracker.calls), 3)

        await hub.close_stream(("edge-01", "traffic-01", "preview", 0))
        self.assertEqual(tracker.resets, [("edge-01", "traffic-01", "preview", 0)])

    async def test_slow_subscriber_drops_oldest_event(self) -> None:
        hub = ResultHub(tracker=FakeTracker(), subscriber_queue_size=1)
        _, queue = hub.subscribe()

        await hub.submit(make_result(0))
        await hub.submit(make_result(1))

        self.assertEqual(hub.subscriber_dropped, 1)
        self.assertEqual(queue.qsize(), 1)
        self.assertEqual(queue.get_nowait()["source_frame_id"], 1)

    async def test_websocket_disconnect_unsubscribes_client(self) -> None:
        hub = ResultHub(tracker=FakeTracker())

        class DisconnectingSocket:
            async def accept(self) -> None:
                return None

            async def send_json(self, _payload) -> None:
                raise ConnectionError("browser closed")

            async def close(self) -> None:
                return None

        task = asyncio.create_task(serve_metadata_websocket(DisconnectingSocket(), hub))
        await asyncio.sleep(0)
        await hub.submit(make_result(0))
        with self.assertRaises(ConnectionError):
            await task
        self.assertEqual(hub._subscribers, {})

    @unittest.skipUnless(
        all(importlib.util.find_spec(name) for name in ("numpy", "torch", "ultralytics")),
        "ByteTrack runtime dependencies unavailable",
    )
    async def test_real_ultralytics_tracker_keeps_id_for_stable_box(self) -> None:
        tracker = CloudByteTrack(frame_rate=15)
        first = tracker.update(make_result(0))
        second = tracker.update(make_result(1))

        self.assertIsNotNone(first.boxes[0].track_id)
        self.assertEqual(second.boxes[0].track_id, first.boxes[0].track_id)
        self.assertEqual(tracker.stream_count, 1)




class ByteTrackOptionsTests(unittest.TestCase):
    def test_env_override_is_opt_in_and_validated(self) -> None:
        from unittest import mock
        self.assertEqual((CloudByteTrack().options["match_thresh"], CloudByteTrack().options["new_track_thresh"]), (0.8, 0.6))  # 8/11 tracker_eval
        with mock.patch.dict("os.environ", {"CLOUD_BYTETRACK_OPTIONS": '{"match_thresh":0.9,"new_track_thresh":0.2}'}):
            options = CloudByteTrack().options
        self.assertEqual((options["match_thresh"], options["new_track_thresh"], options["track_buffer"]), (0.9, 0.2, 60))
        with mock.patch.dict("os.environ", {"CLOUD_BYTETRACK_OPTIONS": '{"match_threshold":0.9}'}):
            with self.assertRaises(ValueError):
                CloudByteTrack()


class VehicleMergeTests(unittest.TestCase):
    def test_merges_only_cross_class_vehicle_overlap(self) -> None:
        from cloud_core.tracking import merge_vehicle_duplicates
        car = Box(.1, .1, .5, .5, .6, 2, "car")
        truck = Box(.11, .1, .5, .52, .4, 7, "truck")
        other_car = Box(.12, .1, .5, .5, .3, 2, "car")
        person = Box(.1, .1, .5, .5, .2, 0, "person")
        far_bus = Box(.6, .6, .9, .9, .5, 5, "bus")
        kept = merge_vehicle_duplicates([truck, car, other_car, person, far_bus], .6)
        self.assertEqual(kept, [car, other_car, person, far_bus])


class ClassAllowlistTests(unittest.TestCase):
    def test_env_allowlist_filters_before_tracking(self) -> None:
        from unittest import mock
        with mock.patch.dict("os.environ", {"CLOUD_CLASS_ALLOWLIST": "0,2"}):
            tracker = CloudByteTrack()
        self.assertEqual(tracker.class_allowlist, frozenset({0, 2}))
        with mock.patch.dict("os.environ", {"CLOUD_CLASS_ALLOWLIST": "2,99"}):
            with self.assertRaises(ValueError):
                CloudByteTrack()

    @unittest.skipUnless(all(importlib.util.find_spec(n) for n in ("numpy", "torch", "ultralytics")),
                         "ByteTrack runtime dependencies unavailable")
    def test_filtered_class_never_reaches_tracker_output(self) -> None:
        from unittest import mock
        with mock.patch.dict("os.environ", {"CLOUD_CLASS_ALLOWLIST": "2"}):
            tracker = CloudByteTrack(frame_rate=15)
        result = make_result(0)
        result = replace(result, boxes=(*result.boxes, Box(.6, .1, .9, .3, .9, 6, "train")))
        tracked = tracker.update(result)
        self.assertEqual([box.class_id for box in tracked.boxes], [2])
        self.assertEqual(tracked.attributes["class_filtered"], 1)


if __name__ == "__main__":
    unittest.main()
