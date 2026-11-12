"""Kafka over the LAN (v4 main lane, step 1): round trip Edge -> Kafka (249) -> echo -> Edge, path-B-sized messages.

  249:  python kafka_ping.py echo --bootstrap 192.168.180.249:29092
  Edge: python kafka_ping.py ping --bootstrap 192.168.180.249:29092 [--n 10000 --size 460000 --rate 60]

ping sends n messages of `size` bytes at `rate` msg/s (default = 4 cameras x 15 fps of ~450 KB path-B tensors) to
v4.test.ping; echo sends each one back on v4.test.pong; ping reports the round trip (send -> pong received; one way
is about half, no clock sync needed), losses and achieved Mbit/s.
"""
import argparse
import asyncio
import os
import struct
import time

import numpy as np
from aiokafka import AIOKafkaConsumer, AIOKafkaProducer

PING, PONG = "v4.test.ping", "v4.test.pong"
MAX = 2_097_152


async def echo(a):
    c = AIOKafkaConsumer(PING, bootstrap_servers=a.bootstrap, auto_offset_reset="latest", fetch_max_bytes=MAX * 4,
                         max_partition_fetch_bytes=MAX)
    p = AIOKafkaProducer(bootstrap_servers=a.bootstrap, max_request_size=MAX, linger_ms=0)
    await c.start()
    await p.start()
    print("echo ready", flush=True)
    try:
        async for m in c:
            await p.send(PONG, m.value)
    finally:
        await c.stop()
        await p.stop()


async def ping(a):
    c = AIOKafkaConsumer(PONG, bootstrap_servers=a.bootstrap, auto_offset_reset="latest", fetch_max_bytes=MAX * 4,
                         max_partition_fetch_bytes=MAX)
    p = AIOKafkaProducer(bootstrap_servers=a.bootstrap, max_request_size=MAX, linger_ms=0)
    await c.start()
    await p.start()
    await asyncio.sleep(2)  # let the consumer join before the first ping
    run = os.urandom(4)
    body = os.urandom(a.size - 16)  # random = incompressible, like the worst case on the wire
    sent, rtt = {}, []

    async def receive():
        async for m in c:
            if m.value[:4] == run:
                (i,) = struct.unpack("!I", m.value[4:8])
                rtt.append(time.perf_counter() - sent.pop(i))
                if len(rtt) == a.n:
                    return

    rx = asyncio.create_task(receive())
    t0 = time.perf_counter()
    for i in range(a.n):
        await asyncio.sleep(max(0.0, t0 + i / a.rate - time.perf_counter()))
        sent[i] = time.perf_counter()
        await p.send(PING, run + struct.pack("!I", i) + b"\0" * 8 + body)
    try:
        await asyncio.wait_for(rx, timeout=10)
    except asyncio.TimeoutError:
        pass
    dur = time.perf_counter() - t0
    await c.stop()
    await p.stop()
    r = np.array(rtt) * 1000
    print(f"{a.n} sent, {len(r)} back, lost {a.n - len(r)} | {a.size / 1000:.0f} KB at {a.rate}/s = "
          f"{a.size * 8 * a.rate / 1e6:.0f} Mbit/s offered, {a.size * 8 * len(r) / dur / 1e6:.0f} achieved")
    if len(r):
        print(f"round trip ms: p50 {np.percentile(r, 50):.1f}  p95 {np.percentile(r, 95):.1f}  "
              f"p99 {np.percentile(r, 99):.1f}  max {r.max():.1f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["echo", "ping"])
    ap.add_argument("--bootstrap", default="192.168.180.249:29092")
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--size", type=int, default=460_000)
    ap.add_argument("--rate", type=float, default=60)
    a = ap.parse_args()
    assert 16 < a.size <= MAX - 1024, "message must fit the 2 MB topic limit"
    asyncio.run(echo(a) if a.mode == "echo" else ping(a))


if __name__ == "__main__":
    main()
