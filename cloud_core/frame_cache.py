"""Bounded, per-stream Cloud cache that joins media bytes with source identity."""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum
from time import time_ns
from typing import Any

from contracts import CloudInferenceTicket, FrameIdentity, MediaFrameRef

StreamKey = tuple[str, str, str, int]
SequenceKey = tuple[StreamKey, int]


class FrameLookupStatus(str, Enum):
    AVAILABLE = "available"
    WAITING = "waiting"
    MISSING = "missing"
    DEADLINE_EXCEEDED = "deadline_exceeded"
    STALE = "stale"
    EVICTED = "evicted"
    PTS_MISMATCH = "pts_mismatch"
    MEDIA_MISSED = "media_missed"
    MEDIA_DISCONNECTED = "media_disconnected"


@dataclass(frozen=True, slots=True)
class DecodedMediaFrame:
    """A Cloud-decoded H.264 frame, identified by its no-B-frame sequence."""

    stream_key: StreamKey
    encoded_seq: int
    media_pts_ns: int
    pixels: Any
    received_utc_ns: int
    media_fps: int = 15

    def __post_init__(self) -> None:
        if len(self.stream_key) != 4 or not all(self.stream_key[:3]):
            raise ValueError("stream_key must contain node, camera, stream and epoch")
        if self.stream_key[3] < 0 or self.encoded_seq < 0 or self.media_pts_ns < 0:
            raise ValueError("decoded frame sequence and PTS must be non-negative")
        if self.received_utc_ns <= 0:
            raise ValueError("received_utc_ns must be positive")


@dataclass(frozen=True, slots=True)
class CachedCloudFrame:
    """Decoded pixels joined to the immutable identity supplied by the Edge."""

    reference: MediaFrameRef
    decoded: DecodedMediaFrame


@dataclass(frozen=True, slots=True)
class FrameLookup:
    status: FrameLookupStatus
    frame: CachedCloudFrame | None = None


class CloudFrameCache:
    """Join MediaFrameRef and decoded media by (stream epoch, encoded sequence).

    The H.264 path deliberately has no route/control-plane detour. The compact
    reference is a parallel sidecar, and this cache refuses to expose a decoded
    frame unless the two inputs agree on stream, sequence and media PTS.
    """

    def __init__(
        self, *, max_frames_per_stream: int = 64, ttl_ns: int = 5_000_000_000,
        max_pending_per_stream: int | None = None,
    ) -> None:
        if max_frames_per_stream <= 0 or ttl_ns <= 0:
            raise ValueError("max_frames_per_stream and ttl_ns must be positive")
        self.max_frames_per_stream = max_frames_per_stream
        self.ttl_ns = ttl_ns
        self.max_pending_per_stream = max_pending_per_stream or max_frames_per_stream
        if self.max_pending_per_stream <= 0:
            raise ValueError("max_pending_per_stream must be positive")
        self._references: dict[SequenceKey, MediaFrameRef] = {}
        self._decoded: dict[SequenceKey, DecodedMediaFrame] = {}
        self._refs_by_identity: dict[FrameIdentity, MediaFrameRef] = {}
        # The sequence belongs to exactly one source frame for one stream
        # epoch. Keep this small ledger so a transport retry cannot silently
        # attach a second payload to an already consumed sequence.
        self._identity_by_sequence: dict[SequenceKey, FrameIdentity] = {}
        self._recent_reference_by_sequence: dict[SequenceKey, MediaFrameRef] = {}
        self._sequence_order: dict[StreamKey, OrderedDict[SequenceKey, None]] = {}
        self._cache: dict[FrameIdentity, CachedCloudFrame] = {}
        self._cache_order: dict[StreamKey, OrderedDict[FrameIdentity, None]] = {}
        self._pending_order: dict[StreamKey, OrderedDict[SequenceKey, None]] = {}
        self._terminal: OrderedDict[FrameIdentity, FrameLookupStatus] = OrderedDict()
        self._changed = asyncio.Condition()
        self._media_watermark: dict[StreamKey, int] = {}
        self._media_disconnected: set[StreamKey] = set()

    async def media_progress(self, stream_key: StreamKey, encoded_seq: int) -> None:
        """Called after ingest by the ordered, no-B-frame RTSP consumer only."""
        async with self._changed:
            self._media_watermark[stream_key] = max(
                encoded_seq, self._media_watermark.get(stream_key, -1))
            self._media_disconnected.discard(stream_key)
            self._changed.notify_all()

    async def media_disconnected(self, stream_key: StreamKey) -> None:
        """Wake waiters without deleting usable decoded/cached frames."""
        async with self._changed:
            self._media_disconnected.add(stream_key)
            self._changed.notify_all()

    @property
    def cached_count(self) -> int:
        return len(self._cache)

    @property
    def pending_count(self) -> int:
        return len(self._references) + len(self._decoded)

    async def announce(self, reference: MediaFrameRef) -> CachedCloudFrame | None:
        """Accept a reference emitted only after the Edge encoded the frame."""

        sequence_key = (reference.frame.stream_key, reference.encoded_seq)
        async with self._changed:
            assigned_identity = self._identity_by_sequence.get(sequence_key)
            if assigned_identity is not None:
                if assigned_identity != reference.frame:
                    raise ValueError("encoded sequence was assigned to another source frame")
                known = self._recent_reference_by_sequence[sequence_key]
                if known != reference:
                    raise ValueError("conflicting MediaFrameRef for sequence")
                return self._cache.get(reference.frame)
            existing = self._references.get(sequence_key)
            if existing is not None:
                if existing != reference:
                    raise ValueError("conflicting MediaFrameRef for sequence")
                return self._join_locked(sequence_key)
            known = self._refs_by_identity.get(reference.frame)
            if known is not None and known != reference:
                raise ValueError("source frame identity was assigned twice")
            self._references[sequence_key] = reference
            self._refs_by_identity[reference.frame] = reference
            self._remember_sequence_locked(sequence_key, reference)
            self._touch_pending_locked(sequence_key)
            joined = self._join_locked(sequence_key)
            self._changed.notify_all()
            return joined

    async def ingest(self, decoded: DecodedMediaFrame) -> CachedCloudFrame | None:
        """Accept one decoded media frame from the Cloud frame tap."""

        sequence_key = (decoded.stream_key, decoded.encoded_seq)
        async with self._changed:
            assigned_identity = self._identity_by_sequence.get(sequence_key)
            if assigned_identity is not None and assigned_identity in self._cache:
                cached = self._cache[assigned_identity]
                if cached.decoded.media_pts_ns != decoded.media_pts_ns:
                    raise ValueError("conflicting decoded media PTS for sequence")
                return cached
            existing = self._decoded.get(sequence_key)
            if existing is not None:
                if existing.media_pts_ns != decoded.media_pts_ns:
                    raise ValueError("conflicting decoded media PTS for sequence")
                return self._join_locked(sequence_key)
            self._decoded[sequence_key] = decoded
            self._remember_sequence_locked(sequence_key, None)
            self._touch_pending_locked(sequence_key)
            joined = self._join_locked(sequence_key)
            self._changed.notify_all()
            return joined

    async def resolve(
        self, ticket: CloudInferenceTicket, *, now_utc_ns: int | None = None,
    ) -> FrameLookup:
        async with self._changed:
            return self._resolve_locked(ticket, now_utc_ns or time_ns())

    async def wait_for(self, ticket: CloudInferenceTicket) -> FrameLookup:
        """Wait only until the Edge-provided absolute deadline, never indefinitely."""

        async with self._changed:
            while True:
                lookup = self._resolve_locked(ticket, time_ns())
                if lookup.status not in {FrameLookupStatus.WAITING, FrameLookupStatus.MISSING}:
                    return lookup
                remaining_ns = ticket.deadline_utc_ns - time_ns()
                if remaining_ns <= 0:
                    return FrameLookup(FrameLookupStatus.DEADLINE_EXCEEDED)
                try:
                    await asyncio.wait_for(self._changed.wait(), remaining_ns / 1_000_000_000)
                except asyncio.TimeoutError:
                    return FrameLookup(FrameLookupStatus.DEADLINE_EXCEEDED)

    def _resolve_locked(self, ticket: CloudInferenceTicket, now_utc_ns: int) -> FrameLookup:
        if now_utc_ns >= ticket.deadline_utc_ns:
            return FrameLookup(FrameLookupStatus.DEADLINE_EXCEEDED)
        self._expire_locked(ticket.frame.stream_key, now_utc_ns)
        cached = self._cache.get(ticket.frame)
        if cached is not None:
            if cached.reference.media_pts_ns != ticket.preview_pts:
                return FrameLookup(FrameLookupStatus.PTS_MISMATCH)
            return FrameLookup(FrameLookupStatus.AVAILABLE, cached)
        terminal = self._terminal.get(ticket.frame)
        if terminal is not None:
            return FrameLookup(terminal)
        reference = self._refs_by_identity.get(ticket.frame)
        if reference is not None:
            if reference.media_pts_ns != ticket.preview_pts:
                return FrameLookup(FrameLookupStatus.PTS_MISMATCH)
            if reference.encoded_seq <= self._media_watermark.get(ticket.frame.stream_key, -1):
                return FrameLookup(FrameLookupStatus.MEDIA_MISSED)
            if ticket.frame.stream_key in self._media_disconnected:
                return FrameLookup(FrameLookupStatus.MEDIA_DISCONNECTED)
            return FrameLookup(FrameLookupStatus.WAITING)
        if ticket.frame.stream_key in self._media_disconnected:
            return FrameLookup(FrameLookupStatus.MEDIA_DISCONNECTED)
        return FrameLookup(FrameLookupStatus.MISSING)

    def diagnostics(self, stream_key: StreamKey) -> dict[str, Any]:
        """Bounded metadata-only snapshot, called on the owning asyncio loop."""
        refs = [(key[1], ref.media_pts_ns) for key, ref in self._references.items()
                if key[0] == stream_key]
        decoded = [(key[1], frame.media_pts_ns) for key, frame in self._decoded.items()
                   if key[0] == stream_key]
        return {"cached": len(self._cache_order.get(stream_key, {})),
                "decoded_watermark": self._media_watermark.get(stream_key),
                "media_disconnected": stream_key in self._media_disconnected,
                "pending_refs": len(refs), "pending_decoded": len(decoded),
                "recent_refs_seq_pts": refs[-3:], "recent_decoded_seq_pts": decoded[-3:]}

    def _join_locked(self, sequence_key: SequenceKey) -> CachedCloudFrame | None:
        reference = self._references.get(sequence_key)
        decoded = self._decoded.get(sequence_key)
        if reference is None or decoded is None:
            return None
        self._references.pop(sequence_key)
        self._decoded.pop(sequence_key)
        self._remove_pending_locked(sequence_key)
        if reference.media_pts_ns != decoded.media_pts_ns:
            self._refs_by_identity.pop(reference.frame, None)
            self._terminal[reference.frame] = FrameLookupStatus.PTS_MISMATCH
            self._trim_terminal_locked()
            return None
        cached = CachedCloudFrame(reference=reference, decoded=decoded)
        self._refs_by_identity.pop(reference.frame, None)
        self._cache[reference.frame] = cached
        order = self._cache_order.setdefault(reference.frame.stream_key, OrderedDict())
        order[reference.frame] = None
        while len(order) > self.max_frames_per_stream:
            evicted_identity, _ = order.popitem(last=False)
            self._cache.pop(evicted_identity, None)
            self._terminal[evicted_identity] = FrameLookupStatus.EVICTED
        self._trim_terminal_locked()
        return cached

    def _touch_pending_locked(self, sequence_key: SequenceKey) -> None:
        stream_key, _ = sequence_key
        order = self._pending_order.setdefault(stream_key, OrderedDict())
        order[sequence_key] = None
        while len(order) > self.max_pending_per_stream:
            stale_key, _ = order.popitem(last=False)
            reference = self._references.pop(stale_key, None)
            self._decoded.pop(stale_key, None)
            if reference is not None:
                self._refs_by_identity.pop(reference.frame, None)
                self._terminal[reference.frame] = FrameLookupStatus.EVICTED
        self._trim_terminal_locked()

    def _remove_pending_locked(self, sequence_key: SequenceKey) -> None:
        stream_key, _ = sequence_key
        order = self._pending_order.get(stream_key)
        if order is not None:
            order.pop(sequence_key, None)
            if not order:
                self._pending_order.pop(stream_key, None)

    def _remember_sequence_locked(
        self, sequence_key: SequenceKey, reference: MediaFrameRef | None,
    ) -> None:
        """Keep a bounded replay/conflict ledger independently per stream."""
        stream_key, _ = sequence_key
        order = self._sequence_order.setdefault(stream_key, OrderedDict())
        order[sequence_key] = None
        order.move_to_end(sequence_key)
        if reference is not None:
            self._identity_by_sequence[sequence_key] = reference.frame
            self._recent_reference_by_sequence[sequence_key] = reference
        limit = max(self.max_frames_per_stream * 4, self.max_pending_per_stream * 2)
        while len(order) > limit:
            expired_key, _ = order.popitem(last=False)
            self._identity_by_sequence.pop(expired_key, None)
            self._recent_reference_by_sequence.pop(expired_key, None)
        if not order:
            self._sequence_order.pop(stream_key, None)

    def _expire_locked(self, stream_key: StreamKey, now_utc_ns: int) -> None:
        order = self._cache_order.get(stream_key)
        if order is None:
            return
        for identity in tuple(order):
            cached = self._cache.get(identity)
            if cached is None:
                order.pop(identity, None)
                continue
            if now_utc_ns - cached.decoded.received_utc_ns < self.ttl_ns:
                continue
            order.pop(identity, None)
            self._cache.pop(identity, None)
            self._terminal[identity] = FrameLookupStatus.STALE
        if not order:
            self._cache_order.pop(stream_key, None)
        self._trim_terminal_locked()

    def _trim_terminal_locked(self) -> None:
        limit = self.max_frames_per_stream * 4
        while len(self._terminal) > limit:
            self._terminal.popitem(last=False)

    async def drop_stream(self, stream_key: StreamKey) -> None:
        """Release one disconnected stream epoch without disturbing other nodes."""
        async with self._changed:
            self._media_watermark.pop(stream_key, None)
            self._media_disconnected.discard(stream_key)
            for sequence_key in self._pending_order.pop(stream_key, {}):
                reference = self._references.pop(sequence_key, None)
                self._decoded.pop(sequence_key, None)
                if reference is not None:
                    self._refs_by_identity.pop(reference.frame, None)
            for identity in self._cache_order.pop(stream_key, {}):
                self._cache.pop(identity, None)
            for sequence_key in self._sequence_order.pop(stream_key, {}):
                self._identity_by_sequence.pop(sequence_key, None)
                self._recent_reference_by_sequence.pop(sequence_key, None)
            for identity in tuple(self._terminal):
                if identity.stream_key == stream_key:
                    self._terminal.pop(identity, None)
            self._changed.notify_all()


class CloudFrameTap:
    """Model-neutral Cloud ingress facade used by the future RTSP decoder."""

    def __init__(self, cache: CloudFrameCache) -> None:
        self.cache = cache

    async def accept_reference(self, reference: MediaFrameRef) -> CachedCloudFrame | None:
        return await self.cache.announce(reference)

    async def accept_decoded(
        self, *, stream_key: StreamKey, encoded_seq: int, media_pts_ns: int,
        pixels: Any, received_utc_ns: int | None = None, media_fps: int = 15,
    ) -> CachedCloudFrame | None:
        return await self.cache.ingest(DecodedMediaFrame(
            stream_key=stream_key,
            encoded_seq=encoded_seq,
            media_pts_ns=media_pts_ns,
            pixels=pixels,
            received_utc_ns=received_utc_ns or time_ns(),
            media_fps=media_fps,
        ))
