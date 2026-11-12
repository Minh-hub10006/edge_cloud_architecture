"""Where does Kafka hang from this host? (8/11: from Edge252 kafka_ping hangs although port 29092 is open and 246 works.)

  python scripts/kafka_steps.py [--bootstrap 192.168.180.249:29092]

Each step prints its time, or TIMEOUT after 15 s and stops; aiokafka warnings/errors are printed too.
"""
import argparse
import asyncio
import logging
import os
import time

from aiokafka import AIOKafkaConsumer, AIOKafkaProducer

logging.basicConfig(level=logging.WARNING, format="  [aiokafka %(levelname)s] %(message)s")
TOPIC = "v4.test.ping"


async def step(name, coro):
    t = time.perf_counter()
    try:
        out = await asyncio.wait_for(coro, 15)
    except asyncio.TimeoutError:
        print(f"{name}: TIMEOUT after 15 s -> stops here", flush=True)
        raise SystemExit(1)
    except Exception as exc:
        print(f"{name}: FAILED {type(exc).__name__}: {exc}", flush=True)
        raise SystemExit(1)
    print(f"{name}: ok in {(time.perf_counter() - t) * 1000:.0f} ms", flush=True)
    return out


async def main(a):
    p = AIOKafkaProducer(bootstrap_servers=a.bootstrap, max_request_size=2_097_152, linger_ms=0)
    await step("1 producer connects (metadata)", p.start())
    await step("2 send 100 B and wait for the ack", p.send_and_wait(TOPIC, b"x" * 100))
    await step("3 send 105 KB and wait for the ack", p.send_and_wait(TOPIC, os.urandom(105_000)))
    t = time.perf_counter()
    for _ in range(30):
        await p.send_and_wait(TOPIC, os.urandom(105_000))
    print(f"4 thirty 105 KB sends: {(time.perf_counter() - t) * 1000 / 30:.1f} ms each", flush=True)
    c = AIOKafkaConsumer(TOPIC, bootstrap_servers=a.bootstrap, auto_offset_reset="earliest",
                         max_partition_fetch_bytes=2_097_152, fetch_max_bytes=8_388_608)
    await step("5 consumer connects", c.start())
    await step("6 consumer reads one message", c.getone())
    await c.stop()
    await p.stop()
    print("all steps ok", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bootstrap", default="192.168.180.249:29092")
    asyncio.run(main(ap.parse_args()))
