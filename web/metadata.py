"""Framework-neutral WebSocket handler for ResultHub detection metadata."""

from __future__ import annotations

import asyncio
from typing import Any

from cloud_core import ResultHub

StreamKey = tuple[str, str, str, int]


async def serve_metadata_websocket(
    websocket: Any, hub: ResultHub, *, stream_key: StreamKey | None = None,
) -> None:
    """Serve tracked JSON events to a WebSocket-compatible connection.

    The connection object is expected to provide ``accept()``, ``send_json``
    and ``close()`` async methods (for example a Starlette WebSocket).
    Video remains on WebRTC; this endpoint sends metadata only.
    """

    await websocket.accept()
    subscriber_id, queue = hub.subscribe(stream_key)
    try:
        while True:
            payload = await queue.get()
            try:
                await websocket.send_json(payload)
            finally:
                queue.task_done()
    finally:
        hub.unsubscribe(subscriber_id)
        try:
            await websocket.close()
        except Exception:
            pass
