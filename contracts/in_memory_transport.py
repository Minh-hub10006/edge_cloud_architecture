"""Deterministic async transport used only by contract tests and local mocks."""

from __future__ import annotations

import asyncio

from .transport import TransportAdapter, TransportEnvelope, TransportKind


class InMemoryTransport(TransportAdapter):
    def __init__(self, kind: TransportKind, max_pending: int = 64) -> None:
        if max_pending <= 0:
            raise ValueError("max_pending must be positive")
        self.kind = kind
        self._queue: asyncio.Queue[TransportEnvelope] = asyncio.Queue(maxsize=max_pending)

    @property
    def pending(self) -> int:
        return self._queue.qsize()

    async def send(self, envelope: TransportEnvelope) -> None:
        await self._queue.put(envelope)

    async def receive(self) -> TransportEnvelope:
        return await self._queue.get()
