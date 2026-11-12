from __future__ import annotations

import asyncio
from time import time_ns
import unittest

from cloud_core import CloudFrameCache
from cloud_core.rtsp_frame_tap import RTSPFrameTapManager, MediaDisconnected
from cloud_core import DecodedMediaFrame
from threading import Event
from contracts import FrameIdentity, MediaFrameRef


class RTSPTapLifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def test_disconnect_marker_is_consumed_after_pixels(self):
        cache = CloudFrameCache()
        manager = RTSPFrameTapManager(cache)
        key = ('a', 'b', 'c', 1)
        queue = asyncio.Queue()
        queue.put_nowait(DecodedMediaFrame(key, 7, 466666667, b'pixels', time_ns()))
        queue.put_nowait(MediaDisconnected(key))
        stop = Event()
        stop.set()
        await manager._consume(queue, stop)
        snapshot = cache.diagnostics(key)
        self.assertEqual(snapshot['decoded_watermark'], 7)
        self.assertTrue(snapshot['media_disconnected'])
        self.assertEqual(snapshot['pending_decoded'], 1)

    async def test_idle_epoch_releases_reader_and_cache(self) -> None:
        cache = CloudFrameCache()
        retired = []

        async def on_idle(key):
            retired.append(key)

        manager = RTSPFrameTapManager(
            cache, idle_timeout_s=0.05, on_stream_idle=on_idle,
        )
        manager._decode_forever = lambda _loop, _queue, stopper, _url, _key: stopper.wait(1)
        frame = FrameIdentity("edge-a", "cam-a", "main", 7, 1, time_ns(), 1)
        await manager.publish_reference(MediaFrameRef(frame, 1, 1, time_ns()))
        self.assertEqual(cache.pending_count, 1)
        await asyncio.wait_for(asyncio.gather(*manager._tasks.values()), timeout=2)
        self.assertEqual(retired, [frame.stream_key])
        self.assertEqual(cache.pending_count, 0)
        self.assertEqual(manager._tasks, {})
        await manager.close()
