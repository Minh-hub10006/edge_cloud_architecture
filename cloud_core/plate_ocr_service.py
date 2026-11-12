"""Cloud plate OCR service: its own process (and Python env with easyocr), separate from the detection server.

  python -m cloud_core.plate_ocr_service --model-home ~/plate_ocr_home --port 50062 --workers 4 --log ocr.jsonl

Each worker process holds one PlateReader on the GPU; requests queue for a free worker. Every reply carries the
measured queue wait and OCR time, and --log appends one JSON line per crop (the load measurement).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from concurrent.futures import Executor, ProcessPoolExecutor
from time import time_ns

from contracts.plate_ocr_wire import METHOD, unpack

_reader = None


def make_reader(model_home: str, passes: int, gpu: bool):
    from inference.plate_ocr import PlateReader
    return PlateReader(model_home, passes=passes, gpu=gpu)


def _init(factory, *args) -> None:
    global _reader
    _reader = factory(*args)


def _warm() -> int:
    import time
    time.sleep(.5)  # overlapping calls make the pool start every worker (and load every model)
    return os.getpid()


def _ocr(jpeg: bytes, arrived_ns: int) -> dict:
    import cv2
    import numpy as np
    started = time_ns()
    crop = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    if crop is None:
        return {"text": "", "calls": 0, "error": "decode", "wait_ms": (started - arrived_ns) / 1e6, "ocr_ms": 0.0}
    text, calls = _reader.read(crop)
    return {"text": text, "calls": calls, "lines": 2 if crop.shape[1] / crop.shape[0] < 2.5 else 1,
            "w": int(crop.shape[1]), "h": int(crop.shape[0]), "wait_ms": (started - arrived_ns) / 1e6,
            "ocr_ms": (time_ns() - started) / 1e6, "worker": os.getpid()}


async def start_server(address: str, executor: Executor, *, log=None):
    import grpc
    loop = asyncio.get_running_loop()

    async def read(request, context):
        arrived = time_ns()
        try:
            header, jpeg = unpack(request)
        except ValueError as exc:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
        reply = await loop.run_in_executor(executor, _ocr, jpeg, arrived)
        if log is not None:
            log.write(json.dumps({**header, **reply, "arrived_utc_ns": arrived}, separators=(",", ":")) + "\n")
            log.flush()
        return json.dumps(reply, separators=(",", ":")).encode()

    service, method = METHOD.strip("/").split("/")
    server = grpc.aio.server(options=(("grpc.max_receive_message_length", 4 * 1024 * 1024),))
    server.add_generic_rpc_handlers((grpc.method_handlers_generic_handler(
        service, {method: grpc.unary_unary_rpc_method_handler(read)}),))
    port = server.add_insecure_port(address)
    if not port:
        raise RuntimeError(f"cannot bind {address}")
    await server.start()
    return server, port


def main() -> None:
    p = argparse.ArgumentParser(description="Cloud plate OCR service (team EasyOCR recognizer)")
    p.add_argument("--model-home", required=True, help="folder made by setup_model.py --home (model/, user_network/)")
    p.add_argument("--port", type=int, default=50062)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--passes", type=int, default=3)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--log", default="", help="new JSONL file, one line per crop")
    a = p.parse_args()
    if a.workers < 1:
        raise SystemExit("--workers >= 1")
    log = open(a.log, "x", encoding="utf-8") if a.log else None

    async def run():
        with ProcessPoolExecutor(a.workers, initializer=_init,
                                 initargs=(make_reader, a.model_home, a.passes, not a.cpu)) as pool:
            # Load every worker's model before accepting crops, so the first requests do not time out.
            pids = await asyncio.gather(*(asyncio.get_running_loop().run_in_executor(pool, _warm)
                                          for _ in range(a.workers)))
            print(f"[Plate OCR] workers ready: {sorted(set(pids))}", flush=True)
            server, _ = await start_server(f"0.0.0.0:{a.port}", pool, log=log)
            print(f"[Plate OCR] :{a.port} workers={a.workers} passes={a.passes} gpu={not a.cpu}", flush=True)
            await server.wait_for_termination()

    try:
        asyncio.run(run())
    finally:
        if log is not None:
            log.close()


if __name__ == "__main__":
    main()
