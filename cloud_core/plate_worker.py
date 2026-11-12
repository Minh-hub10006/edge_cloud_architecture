"""UC1 on the Cloud (8/11): plate crops from Kafka -> the team's EasyOCR reader -> plate reads on Kafka (database).

  KAFKA_BOOTSTRAP_SERVERS=localhost:29092 python -m cloud_core.plate_worker --model-home ~/plate_ocr_home --workers 4

Reads edge.plate-crop.v1 (consumer group v4-plate-ocr: start more processes to share the load), runs each crop on a
pool of OCR processes (one PlateReader on the GPU each, as cloud_core/plate_ocr_service.py), and publishes one message
per crop on cloud.plate-read.v1, which cloud_core/results_writer stores in table plates. Nothing replies to the Edge:
the slow lane does not wait (decision slow-lane-v4).
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
from concurrent.futures import ProcessPoolExecutor
from time import time_ns

from cloud_core.media_store import MEDIA_DIR

CROPS_TOPIC = "edge.plate-crop.v1"
READS_TOPIC = "cloud.plate-read.v1"
KEEP = ("node_id", "camera_id", "stream_id", "stream_epoch", "track_id", "frame_id", "capture_utc_ns", "vehicle_class",
        "read_no", "det_conf")


def read_message(value: bytes, ocr, media_dir: str | None = None) -> dict:
    """One crop message -> the read to publish. ocr(jpeg, arrived_ns) -> {"text", "ocr_ms", "wait_ms", ...}.
    media_dir: the crop is also kept as a file there (evidence for the web; path in table plates)."""
    from inference.plate_ocr import valid_plate
    arrived = time_ns()
    m = json.loads(value)
    reply = ocr(base64.b64decode(m["jpeg_b64"]), arrived)
    text = reply.get("text", "")
    path = None
    if media_dir is not None:
        from cloud_core.media_store import save_media
        path = save_media("plates", f"{m['node_id']}_{m['camera_id']}_{m['stream_epoch']}_{m['track_id']}_"
                                    f"{m['read_no']}", m["jpeg_b64"], media_dir)
    return {"schema_version": 1, **{k: m[k] for k in KEEP}, "text": text, "valid": valid_plate(text),
            "ocr_ms": round(reply.get("ocr_ms", 0.0), 1), "wait_ms": round(reply.get("wait_ms", 0.0), 1),
            "error": reply.get("error"), "image_path": path}


async def serve(bootstrap: str, executor, workers: int) -> None:
    from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
    from cloud_core.plate_ocr_service import _ocr
    consumer = AIOKafkaConsumer(CROPS_TOPIC, bootstrap_servers=bootstrap, group_id="v4-plate-ocr",
                                auto_offset_reset="latest", enable_auto_commit=True, max_partition_fetch_bytes=4 << 20)
    producer = AIOKafkaProducer(bootstrap_servers=bootstrap)
    await consumer.start()
    await producer.start()
    loop = asyncio.get_running_loop()
    slots = asyncio.Semaphore(workers * 2)   # a few crops queued per OCR process, no unbounded backlog in memory
    done = {"read": 0, "valid": 0, "bad": 0}
    from cloud_core.slow_feedback import LagController, encode, worker_source
    from contracts.kafka_transport import SLOW_FEEDBACK_TOPIC
    lag = LagController(worker_source("ocr"))   # P1: OCR falling behind -> Edges send fewer crops

    async def handle(message):
        try:
            capture = json.loads(message.value).get("capture_utc_ns")
            if capture:
                lag.observe(capture)
            out = await loop.run_in_executor(executor, read_message, message.value, _ocr, MEDIA_DIR)
            await producer.send(READS_TOPIC, key=message.key, value=json.dumps(out, separators=(",", ":")).encode())
            done["read"] += 1
            done["valid"] += out["valid"]
        except Exception as exc:   # one bad crop never stops the worker
            done["bad"] += 1
            print(f"[plate_worker] bad crop: {type(exc).__name__}: {exc}", flush=True)
        finally:
            slots.release()

    print(f"[plate_worker] reading {CROPS_TOPIC} with {workers} OCR processes", flush=True)
    last_log = time_ns()
    try:
        async for message in consumer:
            await slots.acquire()
            asyncio.create_task(handle(message))
            if (m := lag.tick()) is not None:
                await producer.send(SLOW_FEEDBACK_TOPIC, value=encode(m))
            if time_ns() - last_log > 30e9:
                last_log = time_ns()
                print(f"[plate_worker] {done} throttle {lag.throttle}", flush=True)
    finally:
        await consumer.stop()
        await producer.stop()


def main() -> None:
    p = argparse.ArgumentParser(description="UC1 plate OCR worker on Kafka")
    p.add_argument("--model-home", required=True)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--passes", type=int, default=3)
    p.add_argument("--cpu", action="store_true")
    a = p.parse_args()
    from cloud_core.plate_ocr_service import _init, _warm, make_reader

    async def run():
        with ProcessPoolExecutor(a.workers, initializer=_init,
                                 initargs=(make_reader, a.model_home, a.passes, not a.cpu)) as pool:
            await asyncio.gather(*(asyncio.get_running_loop().run_in_executor(pool, _warm) for _ in range(a.workers)))
            await serve(os.environ["KAFKA_BOOTSTRAP_SERVERS"], pool, a.workers)

    asyncio.run(run())


if __name__ == "__main__":
    main()
