from __future__ import annotations

import asyncio
import unittest

from contracts import InMemoryTransport, TransportEnvelope, TransportKind


class TransportContractTests(unittest.IsolatedAsyncioTestCase):
    async def test_in_memory_transport_preserves_send_order(self) -> None:
        transport = InMemoryTransport(TransportKind.GRPC, max_pending=2)
        first = TransportEnvelope("ticket", "a", ("edge-01", "cam-01", "main", 2), {"sequence": 1})
        second = TransportEnvelope("ticket", "b", ("edge-01", "cam-01", "main", 2), {"sequence": 2})

        await transport.send(first)
        await transport.send(second)

        self.assertEqual((await transport.receive()).correlation_id, "a")
        self.assertEqual((await transport.receive()).correlation_id, "b")
        self.assertEqual(transport.pending, 0)

    async def test_bounded_transport_applies_backpressure(self) -> None:
        transport = InMemoryTransport(TransportKind.KAFKA, max_pending=1)
        first = TransportEnvelope("result", "a", ("edge-01", "cam-01", "main", 2), {"sequence": 1})
        second = TransportEnvelope("result", "b", ("edge-01", "cam-01", "main", 2), {"sequence": 2})

        await transport.send(first)
        blocked_send = asyncio.create_task(transport.send(second))
        await asyncio.sleep(0)
        self.assertFalse(blocked_send.done())

        self.assertEqual((await transport.receive()).correlation_id, "a")
        await asyncio.wait_for(blocked_send, timeout=0.5)
        self.assertEqual((await transport.receive()).correlation_id, "b")


if __name__ == "__main__":
    unittest.main()
