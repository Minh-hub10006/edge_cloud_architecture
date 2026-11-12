"""Cloud slow-lane worker (P1, 10/11): one consumer group for the slow-lane topics; each topic has a handler (P2: UC3
crowd frames, P3: UC4 pose clips) that calls Triton and returns the messages to publish (rows, events).

  KAFKA_BOOTSTRAP_SERVERS=localhost:29092 TRITON_URL=localhost:18001 python -m cloud_core.slow_worker [--inflight 8]

Rules every message goes through (architecture P1):
  * late: past its deadline_utc_ns (set by the Edge, e.g. capture + 10 s) -> dropped, no GPU spent;
  * bounded: at most `inflight` messages being handled; the rest waits in Kafka (not in this process's memory);
  * lag: every message's age feeds a LagController, which asks the Edges to slow down when this worker falls behind;
  * isolation: a failing message is counted and skipped, never stops the worker.
Start more processes (same consumer group) to share the load; the slow-lane Triton models run at low priority.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from time import time_ns
from typing import Awaitable, Callable

from cloud_core.slow_feedback import LagController, encode, worker_source

# topic -> async handler(message: dict) -> [(topic, key bytes | None, value dict)]
HANDLERS: dict[str, Callable[[dict], Awaitable[list]]] = {}


def handler(topic: str):
    def register(fn):
        HANDLERS[topic] = fn
        return fn
    return register


# async sweep() -> [(topic, key, value)]: run every few seconds (e.g. close an alert whose frames stopped coming)
SWEEPS: list[Callable[[], Awaitable[list]]] = []


def sweeper(fn):
    SWEEPS.append(fn)
    return fn


async def process(topic: str, value: bytes, handlers=HANDLERS, lag: LagController | None = None,
                  now_ns: int | None = None) -> tuple[str, list]:
    """-> (status, outputs). status: ok | late | no_handler. Raises what the handler raises."""
    m = json.loads(value)
    now_ns = now_ns or time_ns()
    if lag is not None and "capture_utc_ns" in m:
        lag.observe(m["capture_utc_ns"], now_ns)
    if m.get("deadline_utc_ns") and now_ns > m["deadline_utc_ns"]:
        return "late", []
    fn = handlers.get(topic)
    if fn is None:
        return "no_handler", []
    return "ok", await fn(m)


async def serve(bootstrap: str, inflight: int, topics=None) -> None:
    from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
    from contracts.kafka_transport import SLOW_FEEDBACK_TOPIC
    topics = topics or sorted(HANDLERS)
    if not topics:
        raise SystemExit("no slow-lane handlers registered")
    consumer = AIOKafkaConsumer(*topics, bootstrap_servers=bootstrap, group_id="v4-slow-lane",
                                auto_offset_reset="latest", enable_auto_commit=True, max_partition_fetch_bytes=8 << 20)
    producer = AIOKafkaProducer(bootstrap_servers=bootstrap, max_request_size=4 << 20)
    await consumer.start()
    await producer.start()
    lag = LagController(worker_source("slow"))
    slots = asyncio.Semaphore(inflight)
    counts = {"ok": 0, "late": 0, "no_handler": 0, "failed": 0}

    async def handle(message):
        try:
            status, outputs = await process(message.topic, message.value, lag=lag)
            counts[status] += 1
            for topic, key, value in outputs:
                await producer.send(topic, key=key, value=json.dumps(value, separators=(",", ":")).encode())
        except Exception as exc:
            counts["failed"] += 1
            if counts["failed"] <= 5:
                print(f"[slow_worker] {message.topic}: {type(exc).__name__}: {exc}", flush=True)
        finally:
            slots.release()

    async def feedback():
        next_sweep = 0.0
        while True:
            await asyncio.sleep(.5)
            if asyncio.get_running_loop().time() >= next_sweep:
                next_sweep = asyncio.get_running_loop().time() + 5
                for sweep in SWEEPS:
                    try:
                        for topic, key, value in await sweep():
                            await producer.send(topic, key=key, value=json.dumps(value, separators=(",", ":")).encode())
                    except Exception as exc:
                        print(f"[slow_worker] sweep {sweep.__name__}: {type(exc).__name__}: {exc}", flush=True)
            if (m := lag.tick()) is not None:
                await producer.send(SLOW_FEEDBACK_TOPIC, value=encode(m))
                if m["messages"] or m["throttle"] > 1:
                    print(f"[slow_worker] {counts} {m}", flush=True)

    print(f"[slow_worker] reading {topics}, {inflight} in flight", flush=True)
    ticker = asyncio.create_task(feedback())
    try:
        async for message in consumer:
            await slots.acquire()
            asyncio.create_task(handle(message))
    finally:
        ticker.cancel()
        await consumer.stop()
        await producer.stop()


def main() -> None:
    p = argparse.ArgumentParser(description="Cloud slow-lane worker (UC3 crowd density, UC4 pose + action)")
    p.add_argument("--inflight", type=int, default=8)
    p.add_argument("--list", action="store_true", help="print the registered topics and sweeps, then exit")
    a = p.parse_args()
    # 10/11: run as `python -m cloud_core.slow_worker` this file is __main__, while the use-case modules register into
    # the importable cloud_core.slow_worker - a second copy whose HANDLERS this __main__ never saw (worker exited with
    # "no slow-lane handlers registered"). Always use that one.
    import cloud_core.slow_worker as registry
    for module in ("cloud_core.uc3_crowd", "cloud_core.uc4_disorder"):   # importing a use case registers it
        try:
            __import__(module)
        except ModuleNotFoundError as exc:
            if exc.name != module:
                raise
    if a.list:
        print(json.dumps({"topics": sorted(registry.HANDLERS), "sweeps": [f.__name__ for f in registry.SWEEPS]}))
        return
    asyncio.run(registry.serve(os.environ["KAFKA_BOOTSTRAP_SERVERS"], a.inflight))


if __name__ == "__main__":
    main()
