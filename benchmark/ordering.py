"""Bounded benchmark reorder gate before the stateful Cloud tracker.

Ordinals are dense dispatch sequence numbers, not source frame IDs. A dropped
source frame therefore creates no hole. A missing network request fails by
deadline instead of silently advancing ByteTrack past an unseen result.
"""
from __future__ import annotations

import asyncio
from collections import OrderedDict
import re
from time import time_ns


def ordinal(request):
    match = re.fullmatch(r"bp1-(\d+)-[0-9a-f]{32}", request.request_id or "")
    return int(match[1]) if match else None


class OrderReservation:
    def __init__(self, lane, sequence):
        self.lane, self.sequence = lane, sequence

    async def wait(self, deadline_utc_ns):
        remaining = (deadline_utc_ns - time_ns()) / 1e9
        if remaining <= 0:
            raise TimeoutError("benchmark deadline before ordered tracking")
        await asyncio.wait_for(self.lane.pending[self.sequence][0].wait(), remaining)

    def finish(self):
        entry = self.lane.pending.get(self.sequence)
        if entry is None:
            return
        entry[1] = True
        while self.lane.next in self.lane.pending and self.lane.pending[self.lane.next][1]:
            self.lane.pending.pop(self.lane.next)
            self.lane.next += 1
        if self.lane.next in self.lane.pending:
            self.lane.pending[self.lane.next][0].set()


class _Lane:
    def __init__(self):
        self.next = 0
        self.pending = {}


class BenchmarkOrdering:
    def __init__(self, capacity=32, max_lanes=128):
        self.capacity, self.max_lanes = capacity, max_lanes
        self.lanes = OrderedDict()

    def reserve(self, request):
        sequence = ordinal(request)
        if sequence is None:
            return None  # Previous release keeps its serial contract.
        key = (request.frame.stream_key, request.decision_id)
        lane = self.lanes.get(key)
        if lane is None:
            if len(self.lanes) >= self.max_lanes:
                idle = next((k for k, v in self.lanes.items() if not v.pending), None)
                if idle is None:
                    raise ValueError("benchmark reorder lanes full")
                self.lanes.pop(idle)
            lane = self.lanes[key] = _Lane()
        self.lanes.move_to_end(key)
        if sequence < lane.next or sequence in lane.pending:
            raise ValueError("duplicate/stale benchmark dispatch ordinal")
        if sequence >= lane.next + self.capacity:
            raise ValueError("benchmark reorder window full")
        event = asyncio.Event()
        if sequence == lane.next:
            event.set()
        lane.pending[sequence] = [event, False]
        return OrderReservation(lane, sequence)
