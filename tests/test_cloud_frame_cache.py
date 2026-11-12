from __future__ import annotations

import asyncio
from time import time_ns
import unittest

from cloud_core import (
    CloudFrameCache,
    CloudFrameTap,
    DecodedMediaFrame,
    FrameLookupStatus,
)
from contracts import CloudInferenceTicket, FrameIdentity, MediaFrameRef, ModelMode


KEY = ("edge-01", "traffic-01", "preview", 0)


def make_identity(frame_id: int, *, epoch: int = 0) -> FrameIdentity:
    return FrameIdentity(
        node_id="edge-01",
        camera_id="traffic-01",
        stream_id="preview",
        stream_epoch=epoch,
        source_frame_id=frame_id,
        capture_utc_ns=1_790_000_000_000_000_000 + frame_id,
        source_pts=frame_id,
    )


def make_reference(
    frame_id: int, *, encoded_seq: int | None = None, epoch: int = 0,
    media_pts_ns: int | None = None,
) -> MediaFrameRef:
    sequence = frame_id if encoded_seq is None else encoded_seq
    pts = sequence * 100_000_000 if media_pts_ns is None else media_pts_ns
    return MediaFrameRef(
        frame=make_identity(frame_id, epoch=epoch),
        encoded_seq=sequence,
        media_pts_ns=pts,
        submitted_utc_ns=time_ns(),
    )


def make_ticket(reference: MediaFrameRef, *, deadline_offset_ns: int = 1_000_000_000) -> CloudInferenceTicket:
    return CloudInferenceTicket(
        frame=reference.frame,
        model_mode=ModelMode.DFINE_CLOUD,
        model_id="dfine-cloud",
        decision_id="window-1",
        preview_pts=reference.media_pts_ns,
        deadline_utc_ns=time_ns() + deadline_offset_ns,
    )


def make_decoded(reference: MediaFrameRef, *, media_pts_ns: int | None = None, received_utc_ns: int | None = None) -> DecodedMediaFrame:
    return DecodedMediaFrame(
        stream_key=reference.frame.stream_key,
        encoded_seq=reference.encoded_seq,
        media_pts_ns=reference.media_pts_ns if media_pts_ns is None else media_pts_ns,
        pixels={"source_frame_id": reference.frame.source_frame_id},
        received_utc_ns=time_ns() if received_utc_ns is None else received_utc_ns,
    )


class CloudFrameCacheTests(unittest.IsolatedAsyncioTestCase):
    async def test_ordered_reader_skips_startup_frame_without_deadline_wait(self):
        cache = CloudFrameCache()
        ref = make_reference(0)
        await cache.announce(ref)
        waiter = asyncio.create_task(cache.wait_for(make_ticket(ref)))
        await cache.ingest(make_decoded(make_reference(7)))
        await cache.media_progress(KEY, 7)
        result = await asyncio.wait_for(waiter, 0.2)
        self.assertEqual(result.status, FrameLookupStatus.MEDIA_MISSED)

    async def test_disconnect_preserves_cache_and_late_sidecar(self):
        cache = CloudFrameCache()
        ref, late, missing = make_reference(7), make_reference(8), make_reference(9)
        await cache.announce(ref)
        await cache.ingest(make_decoded(ref))
        await cache.ingest(make_decoded(late))
        await cache.media_progress(KEY, 8)
        await cache.announce(missing)
        waiter = asyncio.create_task(cache.wait_for(make_ticket(missing)))
        await cache.media_disconnected(KEY)
        self.assertEqual((await asyncio.wait_for(waiter, 0.2)).status,
                         FrameLookupStatus.MEDIA_DISCONNECTED)
        await cache.announce(late)
        for reference in (ref, late):
            self.assertEqual((await cache.resolve(make_ticket(reference))).status,
                             FrameLookupStatus.AVAILABLE)
        other = make_reference(9, epoch=1)
        await cache.announce(other)
        self.assertEqual((await cache.resolve(make_ticket(other))).status, FrameLookupStatus.WAITING)
        await cache.ingest(make_decoded(missing))
        await cache.media_progress(KEY, 9)
        self.assertEqual((await cache.resolve(make_ticket(missing))).status, FrameLookupStatus.AVAILABLE)
        await cache.drop_stream(KEY)
        self.assertIsNone(cache.diagnostics(KEY)['decoded_watermark'])

    async def test_expired_ticket_cannot_use_an_available_cached_frame(self) -> None:
        cache = CloudFrameCache()
        reference = make_reference(3)
        await cache.announce(reference)
        await cache.ingest(make_decoded(reference))
        ticket = make_ticket(reference, deadline_offset_ns=-1)
        self.assertEqual((await cache.wait_for(ticket)).status, FrameLookupStatus.DEADLINE_EXCEEDED)

    async def test_joins_reference_then_decoded_and_resolves_ticket(self) -> None:
        cache = CloudFrameCache()
        reference = make_reference(3)

        self.assertIsNone(await cache.announce(reference))
        joined = await cache.ingest(make_decoded(reference))

        self.assertIsNotNone(joined)
        lookup = await cache.resolve(make_ticket(reference))
        self.assertEqual(lookup.status, FrameLookupStatus.AVAILABLE)
        self.assertEqual(lookup.frame.reference, reference)
        self.assertEqual(lookup.frame.decoded.pixels["source_frame_id"], 3)

    async def test_joins_decoded_then_reference(self) -> None:
        cache = CloudFrameCache()
        reference = make_reference(4)

        self.assertIsNone(await cache.ingest(make_decoded(reference)))
        joined = await cache.announce(reference)

        self.assertIsNotNone(joined)
        self.assertEqual((await cache.resolve(make_ticket(reference))).status, FrameLookupStatus.AVAILABLE)

    async def test_waiter_unblocks_when_both_inputs_arrive(self) -> None:
        cache = CloudFrameCache()
        tap = CloudFrameTap(cache)
        reference = make_reference(5)
        ticket = make_ticket(reference)
        waiter = asyncio.create_task(cache.wait_for(ticket))
        await asyncio.sleep(0)

        await tap.accept_reference(reference)
        await tap.accept_decoded(
            stream_key=reference.frame.stream_key,
            encoded_seq=reference.encoded_seq,
            media_pts_ns=reference.media_pts_ns,
            pixels="frame-5",
        )

        lookup = await waiter
        self.assertEqual(lookup.status, FrameLookupStatus.AVAILABLE)
        self.assertEqual(lookup.frame.decoded.pixels, "frame-5")

    async def test_deadline_is_bounded_when_frame_never_arrives(self) -> None:
        cache = CloudFrameCache()
        reference = make_reference(6)
        lookup = await cache.wait_for(make_ticket(reference, deadline_offset_ns=10_000_000))
        self.assertEqual(lookup.status, FrameLookupStatus.DEADLINE_EXCEEDED)

    async def test_pts_mismatch_is_rejected(self) -> None:
        cache = CloudFrameCache()
        reference = make_reference(7)
        await cache.announce(reference)
        self.assertIsNone(await cache.ingest(make_decoded(reference, media_pts_ns=reference.media_pts_ns + 1)))

        lookup = await cache.resolve(make_ticket(reference))
        self.assertEqual(lookup.status, FrameLookupStatus.PTS_MISMATCH)

    async def test_eviction_and_ttl_are_visible_to_ticket(self) -> None:
        cache = CloudFrameCache(max_frames_per_stream=1, ttl_ns=100)
        first = make_reference(8)
        second = make_reference(9)
        await cache.announce(first)
        await cache.ingest(make_decoded(first))
        await cache.announce(second)
        await cache.ingest(make_decoded(second))
        self.assertEqual((await cache.resolve(make_ticket(first))).status, FrameLookupStatus.EVICTED)

        stale = make_reference(10)
        await cache.announce(stale)
        await cache.ingest(make_decoded(stale, received_utc_ns=time_ns() - 101))
        self.assertEqual((await cache.resolve(make_ticket(stale))).status, FrameLookupStatus.STALE)

    async def test_epoch_isolation_and_duplicate_sequence_rejection(self) -> None:
        cache = CloudFrameCache()
        epoch_zero = make_reference(11, encoded_seq=0, epoch=0)
        epoch_one = make_reference(11, encoded_seq=0, epoch=1)
        for reference in (epoch_zero, epoch_one):
            await cache.announce(reference)
            await cache.ingest(make_decoded(reference))
            self.assertEqual((await cache.resolve(make_ticket(reference))).status, FrameLookupStatus.AVAILABLE)

        conflicting = make_reference(12, encoded_seq=0, epoch=0)
        with self.assertRaisesRegex(ValueError, "encoded sequence"):
            await cache.announce(conflicting)

    async def test_completed_stream_keeps_only_bounded_sequence_ledger(self) -> None:
        cache = CloudFrameCache(max_frames_per_stream=2, ttl_ns=10_000_000_000)
        for frame_id in range(30):
            reference = make_reference(frame_id)
            await cache.announce(reference)
            await cache.ingest(make_decoded(reference))

        self.assertEqual(cache.cached_count, 2)
        self.assertEqual(cache.pending_count, 0)
        self.assertEqual(len(cache._refs_by_identity), 0)
        self.assertLessEqual(len(cache._identity_by_sequence), 8)
        self.assertLessEqual(len(cache._sequence_order[KEY]), 8)

    async def test_reference_retry_after_join_is_idempotent(self) -> None:
        cache = CloudFrameCache()
        reference = make_reference(31)
        await cache.announce(reference)
        joined = await cache.ingest(make_decoded(reference))
        self.assertEqual(await cache.announce(reference), joined)
        self.assertEqual(cache.cached_count, 1)
        self.assertEqual(cache.pending_count, 0)

    async def test_drop_stream_clears_only_one_node_epoch(self) -> None:
        cache = CloudFrameCache()
        old = make_reference(1, epoch=0)
        other = make_reference(1, epoch=1)
        for reference in (old, other):
            await cache.announce(reference)
            await cache.ingest(make_decoded(reference))
        await cache.drop_stream(old.frame.stream_key)
        self.assertEqual(cache.cached_count, 1)
        self.assertEqual((await cache.resolve(make_ticket(other))).status, FrameLookupStatus.AVAILABLE)
        self.assertEqual((await cache.resolve(make_ticket(old))).status, FrameLookupStatus.MISSING)
