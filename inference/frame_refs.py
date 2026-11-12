"""Bounded Edge-side map from analytics source frames to published media PTS."""

from __future__ import annotations

import asyncio
from collections import OrderedDict

from contracts import FrameIdentity, MediaFrameRef


class MediaReferenceMap:
    def __init__(self, *, capacity: int = 2048) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self._refs: OrderedDict[FrameIdentity, MediaFrameRef] = OrderedDict()
        self._waiters: dict[FrameIdentity, asyncio.Future[MediaFrameRef | None]] = {}

    async def publish(self, reference: MediaFrameRef) -> None:
        identity = reference.frame
        previous = self._refs.get(identity)
        if previous is not None and previous != reference:
            raise ValueError("source frame received conflicting media references")
        self._refs[identity] = reference
        self._refs.move_to_end(identity)
        waiter = self._waiters.pop(identity, None)
        if waiter is not None and not waiter.done():
            waiter.set_result(reference)
        while len(self._refs) > self.capacity:
            evicted, _ = self._refs.popitem(last=False)
            stale_waiter = self._waiters.pop(evicted, None)
            if stale_waiter is not None and not stale_waiter.done():
                stale_waiter.set_result(None)

    async def wait_for(
        self, identity: FrameIdentity, *, timeout_s: float = 0.2,
    ) -> MediaFrameRef | None:
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        reference = self._refs.get(identity)
        if reference is not None:
            self._refs.move_to_end(identity)
            return reference
        waiter = self._waiters.get(identity)
        if waiter is None:
            waiter = asyncio.get_running_loop().create_future()
            self._waiters[identity] = waiter
        try:
            return await asyncio.wait_for(asyncio.shield(waiter), timeout_s)
        except asyncio.TimeoutError:
            return None
        finally:
            if self._waiters.get(identity) is waiter:
                self._waiters.pop(identity, None)
