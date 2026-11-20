"""Forward AI Cloud plate reads into the VMS-owned ingestion pipeline.

This is a separate consumer group: it never changes the AI platform's results
writer or the existing Cloud Kafka topics. VMS acceptance is acknowledged
before this consumer commits its Kafka offset.
"""
from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import ssl
from http.client import HTTPSConnection
from urllib.error import HTTPError, URLError
from urllib.request import HTTPSHandler, Request, build_opener
from uuid import UUID

from cloud_core.media_store import MEDIA_DIR

TOPIC = "cloud.plate-read.v1"
MAX_EVIDENCE_BYTES = 20 << 20
PLATE_PATTERN = re.compile(r"^\d{2}[A-Z][A-Z0-9]?\d{4,5}$")


class _PinnedAddressHTTPSConnection(HTTPSConnection):
    """Connect to a fixed address while retaining URL-host TLS verification and SNI."""

    def __init__(self, host: str, connect_ip: str, **kwargs):
        self.connect_ip = connect_ip
        super().__init__(host, **kwargs)
        self._create_connection = self._connect_to_address

    def _connect_to_address(self, address, timeout=socket._GLOBAL_DEFAULT_TIMEOUT, source_address=None):
        return socket.create_connection((self.connect_ip, address[1]), timeout, source_address)


class _PinnedAddressHTTPSHandler(HTTPSHandler):
    def __init__(self, context: ssl.SSLContext, connect_ip: str):
        super().__init__(context=context)
        self.connect_ip = connect_ip

    def https_open(self, request):
        return self.do_open(
            lambda host, **kwargs: _PinnedAddressHTTPSConnection(host, self.connect_ip, **kwargs),
            request,
            context=self._context,
        )


class VMSResponseError(RuntimeError):
    def __init__(self, status: int, code: str):
        super().__init__(f"VMS returned HTTP {status} ({code or 'unknown'})")
        self.status = status


def uuid7_for_event(message: dict) -> str:
    """Create a retry-stable UUIDv7 from the source capture timestamp and identity."""
    capture_ns = int(message["capture_utc_ns"])
    timestamp_ms = capture_ns // 1_000_000
    if timestamp_ms <= 0 or timestamp_ms >= 1 << 48:
        raise ValueError("capture_utc_ns is outside the UUIDv7 timestamp range")
    source = "\0".join(str(message[key]) for key in (
        "node_id", "camera_id", "stream_id", "stream_epoch", "track_id", "frame_id", "read_no"))
    random_bits = int.from_bytes(hashlib.sha256(source.encode("utf-8")).digest()[:10], "big") & ((1 << 74) - 1)
    rand_a, rand_b = random_bits >> 62, random_bits & ((1 << 62) - 1)
    value = (timestamp_ms << 80) | (0x7 << 76) | (rand_a << 64) | (0b10 << 62) | rand_b
    return str(UUID(int=value))


def captured_at(message: dict) -> str:
    value = datetime.fromtimestamp(int(message["capture_utc_ns"]) / 1_000_000_000, timezone.utc)
    return value.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def evidence_bytes(message: dict, media_root: Path) -> bytes:
    path = Path(str(message.get("image_path", ""))).resolve(strict=True)
    root = media_root.resolve(strict=True)
    if path != root and root not in path.parents:
        raise ValueError("AI evidence path is outside the configured media directory")
    data = path.read_bytes()
    if not data.startswith(b"\xff\xd8\xff") or not 1 <= len(data) <= MAX_EVIDENCE_BYTES:
        raise ValueError("AI evidence is not a supported JPEG")
    return data


def manifest_for(message: dict, evidence: bytes, grant: dict, edge_id: str, camera_id: str,
                 model_version: str) -> dict:
    text = "".join(str(message.get("text", "")).upper().split())
    if not message.get("valid") or not PLATE_PATTERN.fullmatch(text):
        raise ValueError("AI result is not a valid plate read")
    confidence = float(message["det_conf"])
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("plate detector confidence is outside [0,1]")
    event_id = uuid7_for_event(message)
    timestamp = captured_at(message)
    return {
        "schema_version": "0.1.0-draft.4",
        "event_id": event_id,
        "event_type": "PLATE_RECOGNIZED",
        "occurred_at": timestamp,
        "camera_id": camera_id,
        "edge_id": edge_id,
        "ai": {"model_id": "edge-cloud-plate-pipeline", "model_version": model_version},
        "detections": [{
            "detection_id": f"plate-{event_id}",
            "detection_type": "LICENSE_PLATE",
            "confidence": confidence,
            "plate_raw": text,
            "plate_normalized": text,
            "track_id": str(message["track_id"]),
        }],
        "violations": [],
        "evidence": [{
            "image_id": f"plate-crop-{event_id}",
            "evidence_type": "PLATE_CROP",
            "bucket": grant["bucket"],
            "object_key": grant["object_key"],
            "content_type": "image/jpeg",
            "size_bytes": len(evidence),
            "sha256": hashlib.sha256(evidence).hexdigest(),
            "captured_at": timestamp,
        }],
        "correlation_id": f"{message['node_id']}:{message['camera_id']}:{message['stream_epoch']}:{message['track_id']}:{message['read_no']}",
    }


class VMSClient:
    def __init__(self, origin: str, ca_file: str, cert_file: str, key_file: str,
                 timeout: float = 15.0, connect_ip: str = ""):
        self.origin = origin.rstrip("/")
        self.context = ssl.create_default_context(cafile=ca_file)
        self.context.load_cert_chain(certfile=cert_file, keyfile=key_file)
        self.timeout = timeout
        handlers = [_PinnedAddressHTTPSHandler(self.context, connect_ip)] if connect_ip else [HTTPSHandler(context=self.context)]
        self.opener = build_opener(*handlers)

    def request(self, method: str, url: str, body: bytes | None = None, content_type: str = "application/json"):
        request = Request(url, data=body, method=method, headers={"Content-Type": content_type})
        try:
            with self.opener.open(request, timeout=self.timeout) as response:
                return response.status, response.read()
        except HTTPError as exc:
            try:
                payload = json.loads(exc.read())
                code = str(payload.get("code", ""))
            except (ValueError, TypeError):
                code = ""
            raise VMSResponseError(exc.code, code) from exc
        except URLError as exc:
            raise ConnectionError("VMS ingestion endpoint is unavailable") from exc

    def submit(self, manifest: dict, image: bytes) -> None:
        expected = manifest["evidence"][0]
        event_id = manifest["event_id"]
        request = {
            "event_id": event_id,
            "image_id": expected["image_id"],
            "content_type": "image/jpeg",
            "size_bytes": len(image),
            "sha256": expected["sha256"],
        }
        _, payload = self.request("POST", self.origin + "/api/v1/ai/evidence/upload-authorizations",
                                  json.dumps(request, separators=(",", ":")).encode())
        grant = json.loads(payload)
        self.request("PUT", grant["upload_url"], image, "image/jpeg")
        expected["bucket"] = grant["bucket"]
        expected["object_key"] = grant["object_key"]
        status, _ = self.request("POST", self.origin + "/api/v1/ai/events",
                                 json.dumps(manifest, separators=(",", ":")).encode())
        if status != 202:
            raise VMSResponseError(status, "unexpected_ack")


async def publish(bootstrap: str, args) -> None:
    from aiokafka import AIOKafkaConsumer

    consumer = AIOKafkaConsumer(TOPIC, bootstrap_servers=bootstrap, group_id=args.group,
                                auto_offset_reset="latest", enable_auto_commit=False,
                                max_partition_fetch_bytes=4 << 20)
    client = VMSClient(args.vms_origin, args.vms_ca, args.vms_cert, args.vms_key,
                       args.timeout, connect_ip=args.vms_connect_ip)
    media_root = Path(args.media_dir)
    await consumer.start()
    print(f"[vms-ai-publisher] consuming {TOPIC}; camera={args.edge_id}/{args.camera_id}", flush=True)
    try:
        async for record in consumer:
            try:
                message = json.loads(record.value)
                if message.get("node_id") != args.edge_id or message.get("camera_id") != args.camera_id:
                    await consumer.commit()
                    continue
                text = "".join(str(message.get("text", "")).upper().split())
                if not message.get("valid") or not PLATE_PATTERN.fullmatch(text):
                    await consumer.commit()
                    continue
                image = evidence_bytes(message, media_root)
                manifest = manifest_for(message, image, {"bucket": "traffic-evidence", "object_key": "committed/pending"},
                                        args.edge_id, args.camera_id, args.model_version)
                await asyncio.to_thread(client.submit, manifest, image)
                await consumer.commit()
                print(f"[vms-ai-publisher] CLOUD_ACCEPTED event={manifest['event_id']} source_offset={record.offset}", flush=True)
            except Exception as exc:
                # Do not commit past an unaccepted VMS event. Restart/retry from the
                # same Kafka offset after the cause is corrected.
                print(f"[vms-ai-publisher] stopping without offset commit: {type(exc).__name__}: {exc}", flush=True)
                raise
    finally:
        await consumer.stop()


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bootstrap", default=os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:29092"))
    p.add_argument("--group", default=os.getenv("VMS_AI_CONSUMER_GROUP", "vms-ai-plate-publisher"))
    p.add_argument("--vms-origin", default=os.getenv("VMS_INGESTION_ORIGIN", ""))
    p.add_argument("--vms-connect-ip", default=os.getenv("VMS_INGESTION_CONNECT_IP", ""),
                   help="Optional address override while preserving the origin hostname for TLS/SNI")
    p.add_argument("--vms-ca", default=os.getenv("VMS_INGESTION_CA_FILE", ""))
    p.add_argument("--vms-cert", default=os.getenv("VMS_AI_CLIENT_CERT_FILE", ""))
    p.add_argument("--vms-key", default=os.getenv("VMS_AI_CLIENT_KEY_FILE", ""))
    p.add_argument("--edge-id", default=os.getenv("VMS_AI_EDGE_ID", ""))
    p.add_argument("--camera-id", default=os.getenv("VMS_AI_CAMERA_ID", ""))
    p.add_argument("--media-dir", default=os.getenv("V4_MEDIA_DIR", MEDIA_DIR))
    p.add_argument("--model-version", default=os.getenv("VMS_AI_MODEL_VERSION", "edge-cloud-architecture@1460469f"))
    p.add_argument("--timeout", type=float, default=15.0)
    return p


def main():
    args = parser().parse_args()
    if not all((args.vms_origin, args.vms_ca, args.vms_cert, args.vms_key, args.edge_id, args.camera_id)):
        raise SystemExit("VMS origin, mTLS files and scoped Edge/camera IDs are required")
    asyncio.run(publish(args.bootstrap, args))


if __name__ == "__main__":
    main()
