"""P5 (11/11): an idempotent producer the broker forgot (Kafka restarted) is replaced, not retried forever."""
import asyncio
import unittest


class OutOfOrderSequenceNumber(Exception):
    pass


class FakeProducer:
    made = []

    def __init__(self):
        self.broken, self.sent, self.stopped = False, [], False
        FakeProducer.made.append(self)

    async def start(self):
        pass

    async def stop(self):
        self.stopped = True

    async def send_and_wait(self, topic, value=None, key=None):
        if self.broken:
            raise OutOfOrderSequenceNumber("[Error 45]")
        self.sent.append(value)

    async def send(self, topic, value=None, key=None):
        future = asyncio.get_running_loop().create_future()
        if self.broken:
            future.set_exception(OutOfOrderSequenceNumber("[Error 45]"))
        else:
            self.sent.append(value)
            future.set_result(None)
        return future


class RenewTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        from contracts.kafka_transport import RenewingProducer
        FakeProducer.made.clear()
        self.p = RenewingProducer(FakeProducer)
        await self.p.start()

    async def test_a_waited_send_renews_the_producer_and_is_delivered(self):
        FakeProducer.made[0].broken = True               # the broker restarted
        await self.p.send_and_wait("t", value=b"event")
        self.assertEqual((len(FakeProducer.made), FakeProducer.made[1].sent, self.p.renewed), (2, [b"event"], 1))
        self.assertTrue(FakeProducer.made[0].stopped)

    async def test_a_send_without_wait_marks_it_broken_and_the_next_send_renews(self):
        FakeProducer.made[0].broken = True
        future = await self.p.send("t", value=b"count-1")   # lost: nobody waits for it
        await asyncio.sleep(0)
        self.assertIsInstance(future.exception(), OutOfOrderSequenceNumber)
        await self.p.send("t", value=b"count-2")
        self.assertEqual((len(FakeProducer.made), FakeProducer.made[1].sent), (2, [b"count-2"]))

    async def test_other_errors_are_not_renewed(self):
        async def down(*a, **k):
            raise ConnectionError("broker down")
        FakeProducer.made[0].send_and_wait = down
        with self.assertRaises(ConnectionError):
            await self.p.send_and_wait("t", value=b"x")
        self.assertEqual((len(FakeProducer.made), self.p.renewed), (1, 0))


class SuperviseTests(unittest.IsolatedAsyncioTestCase):
    async def test_a_consume_loop_that_failed_restarts_with_a_new_consumer(self):
        # 11/11 P5: a reply publish timed out while Kafka was down and the ticket loop ended for good (modes C/D dead)
        from contracts.kafka_transport import KafkaCloudWorker

        class Consumer:
            made = 0

            def __init__(self):
                Consumer.made += 1
                self.started = self.stopped = False

            async def start(self):
                self.started = True

            async def stop(self):
                self.stopped = True

        worker = KafkaCloudWorker(service=None, bootstrap_servers="fake")
        calls = []

        async def consume(topic, consumer):
            calls.append(consumer)
            if len(calls) == 1:
                raise asyncio.TimeoutError()
            return None   # stopped normally
        worker._consume = consume
        first = Consumer()
        worker._consumers.append(first)
        await worker._supervise("t", Consumer, first, backoff_s=0)
        self.assertEqual(len(calls), 2)
        self.assertTrue(first.stopped)
        self.assertIsNot(calls[1], first)
        self.assertTrue(calls[1].started)
        self.assertEqual(worker._consumers, [calls[1]])



class BackgroundSendTests(unittest.IsolatedAsyncioTestCase):
    async def test_a_broker_that_does_not_answer_never_stalls_the_edge(self):
        # 11/11 P5: Edge-only mode N lost 781 frames while Kafka was down: each frame awaited its database copy
        from contracts import Box, DetectionResult, FrameIdentity, ModelMode
        from contracts.kafka_transport import KafkaCloudClient
        client = KafkaCloudClient("fake", node_id="edge-246")
        hang = asyncio.Event()

        class Stuck:
            async def send(self, *a, **k):
                await hang.wait()   # metadata never arrives
        client.producer = Stuck()
        client.BACKGROUND_MAX, client.BACKGROUND_TIMEOUT_S = 3, 60
        ident = FrameIdentity("edge-246", "cam-1", "s", 1, 1, 1_790_000_000_000_000_000, 1)
        result = DetectionResult(frame=ident, model_mode=ModelMode.YOLO_EDGE, model_id="m", decision_id="d",
                                 boxes=(Box(.1, .1, .2, .2, .9, 2, "car"),), inference_ms=1.0,
                                 produced_utc_ns=1_790_000_000_000_000_001, request_id="r")
        for _ in range(5):
            await asyncio.wait_for(client.publish_tracked_result(result), 0.1)   # returns at once
        await asyncio.wait_for(client.publish_slow("count", {"node_id": "n", "camera_id": "c", "stream_id": "s",
                                                             "stream_epoch": 1}), 0.1)
        self.assertEqual((len(client._background), client.background_dropped), (3, 3))
        hang.set()
        await asyncio.sleep(0.01)
        self.assertEqual(len(client._background), 0)


if __name__ == "__main__":
    unittest.main()
