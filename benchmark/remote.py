"""Manual worker registration/barrier over a trusted LAN, not remote execution.

The coordinator sends only validated RunConfig data, never shell commands,
artifact paths or credentials. Each worker's CLI owns all local paths/runtimes.
No authentication/TLS: bind only to the intended trusted experiment network.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from uuid import uuid4

from .campaign import run_case
from .config import RunConfig, finite_nonnegative

MAX_LINE = 1024 * 1024


async def send(writer, row):
    data = json.dumps(row, allow_nan=False, separators=(",", ":")).encode() + b"\n"
    if len(data) > MAX_LINE:
        raise ValueError("coordinator message too large")
    writer.write(data)
    await writer.drain()


async def receive(reader):
    data = await reader.readline()
    if not data:
        raise ConnectionError("benchmark peer disconnected")
    if len(data) > MAX_LINE:
        raise ValueError("coordinator message too large")
    row = json.loads(data)
    if not isinstance(row, dict):
        raise ValueError("coordinator message must be an object")
    if row.get("type") == "ERROR":
        raise RuntimeError(row.get("message", "coordinator error"))
    return row


class WorkerConnection:
    def __init__(self, reader, writer, timeout_s):
        self.reader, self.writer, self.timeout_s = reader, writer, timeout_s

    @classmethod
    async def open(cls, endpoint, node_id, timeout_s=180):
        host, port = endpoint.rsplit(":", 1)
        reader, writer = await asyncio.wait_for(asyncio.open_connection(host, int(port), limit=MAX_LINE), timeout_s)
        connection = cls(reader, writer, timeout_s)
        try:
            await send(writer, {"type": "HELLO", "schema_version": 1, "node_id": node_id})
        except BaseException:
            await connection.close()
            raise
        return connection

    async def config(self, local):
        message = await asyncio.wait_for(receive(self.reader), self.timeout_s)
        if message.get("type") != "PREPARE":
            raise ValueError("expected PREPARE")
        config = RunConfig(**message["config"])
        for key in ("node_id", "model_key", "placement", "measured_frames", "warmup_frames", "profile", "transport", "repetition"):
            if getattr(config, key) != getattr(local, key):
                await send(self.writer, {"type": "ERROR", "message": f"worker configuration mismatch: {key}"})
                raise ValueError(f"worker configuration mismatch: {key}")
        return config

    async def ready(self, config):
        await send(self.writer, {"type": "READY", "run_id": config.run_id})
        message = await asyncio.wait_for(receive(self.reader), self.timeout_s)
        if message != {"type": "START", "run_id": config.run_id}:
            raise ValueError("invalid START/run identity")

    async def done(self, summary):
        await send(self.writer, {"type": "DONE", "summary": summary})

    async def close(self):
        self.writer.close()
        try:
            await asyncio.wait_for(self.writer.wait_closed(), 3)
        except (ConnectionError, TimeoutError):
            pass


class RemoteWorker:
    """Socket lives for the campaign; each runner drains its resources before DONE."""
    def __init__(self, reader, writer):
        self.reader, self.writer = reader, writer

    async def prepare(self, config):
        await send(self.writer, {"type": "PREPARE", "config": config.as_dict()})
        if await receive(self.reader) != {"type": "READY", "run_id": config.run_id}:
            raise ValueError("invalid READY/run identity")

    async def measure(self, config):
        await send(self.writer, {"type": "START", "run_id": config.run_id})
        message = await receive(self.reader)
        if message.get("type") != "DONE":
            raise ValueError("expected DONE")
        return message["summary"]

    async def close(self):
        pass  # Campaign owns this connection, not the individual repetition.

    async def disconnect(self):
        self.writer.close()
        try:
            await asyncio.wait_for(self.writer.wait_closed(), 3)
        except (ConnectionError, TimeoutError):
            pass


async def coordinate(base: RunConfig, nodes: list[str], directory: Path, *, host="127.0.0.1", port=9200,
                     repetitions=3, ready_timeout_s=180, measure_timeout_s=600):
    if not nodes or len(nodes) > 32 or len(set(nodes)) != len(nodes) or not all(isinstance(n, str) and n.strip() for n in nodes):
        raise ValueError("unique nonempty node IDs required")
    if type(repetitions) is not int or not 1 <= repetitions <= 100:
        raise ValueError("repetitions must be 1..100")
    for value in (ready_timeout_s, measure_timeout_s):
        if finite_nonnegative(value, "timeout") <= 0:
            raise ValueError("positive timeouts required")
    directory.mkdir(parents=True, exist_ok=False)
    futures = {node: asyncio.get_running_loop().create_future() for node in nodes}
    connections = set()
    handshakes = set()
    workers = {}
    reports = []

    async def register(reader, writer):
        handshakes.add(asyncio.current_task())
        connections.add(writer)
        retained = False
        try:
            if len(connections) > len(nodes) + 8:
                return
            message = await asyncio.wait_for(receive(reader), 5)
            node = message.get("node_id")
            if (message.get("type") != "HELLO" or message.get("schema_version") != 1
                    or node not in futures or futures[node].done()):
                await send(writer, {"type": "ERROR", "message": "unknown/duplicate benchmark worker"})
                return
            futures[node].set_result(RemoteWorker(reader, writer))
            retained = True
        except (ValueError, RuntimeError, ConnectionError, TimeoutError):
            pass
        finally:
            handshakes.discard(asyncio.current_task())
            if not retained:
                writer.close()
                connections.discard(writer)

    server = await asyncio.start_server(register, host, port, limit=MAX_LINE)
    print(json.dumps({"coordinator_listen": [str(s.getsockname()) for s in server.sockets],
                      "expected_nodes": nodes, "campaign_id": base.campaign_id}), flush=True)
    try:
        registered = await asyncio.wait_for(asyncio.gather(*futures.values()), ready_timeout_s)
        workers = dict(zip(nodes, registered))
        for repetition in range(1, repetitions + 1):
            run_id = uuid4().hex
            configs = [RunConfig(base.campaign_id, run_id, node, base.model_key, base.placement,
                base.measured_frames, base.warmup_frames, base.profile, base.transport, repetition) for node in nodes]
            report = await run_case(configs, workers, ready_timeout_s=ready_timeout_s, measure_timeout_s=measure_timeout_s)
            report.update(campaign_id=base.campaign_id, run_id=run_id, repetition=repetition)
            reports.append(report)
            with (directory / f"repetition-{repetition}.json").open("x", encoding="utf-8") as target:
                json.dump(report, target, indent=2, ensure_ascii=False, allow_nan=False)
            print(json.dumps({"repetition": repetition, "status": report["status"],
                              "coordinated_fps": report["coordinated_fps"]}), flush=True)
            if report["status"] != "PASS":
                break
    except Exception as exc:
        reports.append({"status": "INCOMPLETE", "error": f"{type(exc).__name__}: {exc}"})
    finally:
        server.close()
        for task in list(handshakes):
            task.cancel()
        await asyncio.gather(*list(handshakes), return_exceptions=True)
        for writer in connections:
            writer.close()
        await asyncio.gather(*(worker.disconnect() for worker in workers.values()), return_exceptions=True)
        await asyncio.wait_for(server.wait_closed(), 5)
    record = {"schema_version": 1, "campaign_id": base.campaign_id, "expected_nodes": nodes,
              "reports": reports, "status": "PASS" if len(reports) == repetitions and all(r["status"] == "PASS" for r in reports) else "INCOMPLETE"}
    with (directory / "campaign.json").open("x", encoding="utf-8") as target:
        json.dump(record, target, indent=2, ensure_ascii=False, allow_nan=False)
    return record
