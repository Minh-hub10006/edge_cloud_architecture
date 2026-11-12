from __future__ import annotations

import asyncio
import unittest

import numpy as np

from contracts import FrameIdentity, MediaFrameRef
from inference.adapters import _normalize_xyxy
from inference.frame_refs import MediaReferenceMap


class Phase7AdapterContractTests(unittest.TestCase):
    def test_yolo_pixel_boxes_are_clipped_and_normalized(self) -> None:
        box = _normalize_xyxy({
            "left": -5, "top": 10, "right": 120, "bottom": 80,
            "confidence": 0.9, "class_id": 2, "label": "car",
        }, 100, 100)
        self.assertIsNotNone(box)
        assert box is not None
        self.assertEqual((box.left, box.top, box.right, box.bottom), (0.0, 0.1, 1.0, 0.8))
        self.assertEqual((box.class_id, box.class_name), (2, "car"))

    def test_degenerate_detection_is_dropped(self) -> None:
        box = _normalize_xyxy({
            "left": 80, "top": 10, "right": 20, "bottom": 30,
            "confidence": 0.9, "class_id": 0,
        }, 100, 100)
        self.assertIsNone(box)


class MediaReferenceMapTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.identity = FrameIdentity(
            node_id="edge-246", camera_id="traffic-01", stream_id="preview",
            stream_epoch=1, source_frame_id=9,
            capture_utc_ns=1_790_000_000_000_000_000, source_pts=600_000_000,
        )
        self.reference = MediaFrameRef(
            frame=self.identity, encoded_seq=4, media_pts_ns=266_666_667,
            submitted_utc_ns=1_790_000_000_010_000_000,
        )

    async def test_waiter_resolves_when_media_branch_publishes(self) -> None:
        refs = MediaReferenceMap(capacity=2)
        waiter = asyncio.create_task(refs.wait_for(self.identity, timeout_s=0.5))
        await asyncio.sleep(0)
        await refs.publish(self.reference)
        self.assertEqual(await waiter, self.reference)

    async def test_lookup_timeout_is_bounded(self) -> None:
        refs = MediaReferenceMap(capacity=2)
        self.assertIsNone(await refs.wait_for(self.identity, timeout_s=0.01))

    async def test_reference_cache_is_bounded(self) -> None:
        refs = MediaReferenceMap(capacity=1)
        await refs.publish(self.reference)
        other = FrameIdentity(
            node_id="edge-247", camera_id="traffic-01", stream_id="preview",
            stream_epoch=1, source_frame_id=9,
            capture_utc_ns=1_790_000_000_000_000_000, source_pts=600_000_000,
        )
        await refs.publish(MediaFrameRef(
            frame=other, encoded_seq=4, media_pts_ns=266_666_667,
            submitted_utc_ns=1_790_000_000_010_000_000,
        ))
        self.assertIsNone(await refs.wait_for(self.identity, timeout_s=0.01))


if __name__ == "__main__":
    unittest.main()
