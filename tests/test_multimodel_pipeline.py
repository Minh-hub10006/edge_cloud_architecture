"""New model wire/dispatch/tracking checks; no claim of real model capacity."""
from __future__ import annotations

import asyncio
from dataclasses import replace
import json
from pathlib import Path
import tempfile
from time import time_ns
import unittest
from unittest.mock import patch
import grpc

from benchmark.cloud import configure
from cloud_core import CloudFrameCache, DecodedMediaFrame, ResultHub
from cloud_core.grpc_service import CloudDataPlaneServicer
from contracts import Box, CloudInferenceTicket, DetectionResult, FrameIdentity, MediaFrameRef, ModelMode
from contracts.grpc_codec import ticket_to_proto, ticket_from_proto, result_to_proto, result_from_proto
from contracts.kafka_transport import KafkaCloudWorker, CLOUD_TICKETS_TOPIC
from contracts.proto import edge_video_platform_pb2 as pb, edge_video_platform_pb2_grpc as pb_grpc


class Tracker:
    def update(self, result):
        return replace(result, boxes=[replace(box, track_id=42) for box in result.boxes])
    def reset(self, key):
        pass


class Adapter:
    runtime = "test_only"
    def __init__(self, model_id):
        self.model_id = model_id
        self.seen = []
    def infer(self, pixels):
        self.seen.append(pixels)
        return [Box(.1,.2,.4,.5,.9,2,"car")], 1
    def infer_detailed(self, pixels):
        boxes, ms = self.infer(pixels)
        return boxes, ms, {"cloud_preprocess_ms": .2, "cloud_session_call_ms": .5, "cloud_postprocess_ms": .3}


class MultimodelPipelineTests(unittest.IsolatedAsyncioTestCase):
    async def setup_service(self):
        self.cache = CloudFrameCache()
        self.adapters = {ModelMode.RTDETR_CLOUD: Adapter("rtdetrv2_r18vd_sp2_120e_coco"),
                         ModelMode.LWDETR_CLOUD: Adapter("lwdetr_large_coco")}
        self.service = CloudDataPlaneServicer(hub=ResultHub(tracker=Tracker()), frame_cache=self.cache,
            frame_taps=object(), cloud_adapters=self.adapters, benchmark_enabled=True, max_cloud_inflight=2)
        self.service.benchmark_identities = {mode: {"model_id": a.model_id, "artifact_sha256": "test-hash", "runtime": "test_only"} for mode,a in self.adapters.items()}

    async def ticket(self, mode, *, index=0, node="edge"):
        now = time_ns()
        frame = FrameIdentity(node,"cam",mode.value,1,index,now,index*33333333)
        ref = MediaFrameRef(frame,index,index*33333333,now)
        await self.cache.announce(ref)
        await self.cache.ingest(DecodedMediaFrame(frame.stream_key,index,index*33333333,"pixels-"+mode.value,now))
        return CloudInferenceTicket(frame,mode,self.adapters[mode].model_id,"benchmark/c/r/measure",
            ref.media_pts_ns, now+10_000_000_000, request_id=f"bp1-{index}-"+"a"*32)

    async def test_real_local_grpc_new_modes_preserve_frame_and_track_endpoint(self):
        await self.setup_service()
        server = grpc.aio.server()
        pb_grpc.add_EdgeCloudDataPlaneServicer_to_server(self.service, server)
        port = server.add_insecure_port("127.0.0.1:0")
        await server.start()
        channel = grpc.aio.insecure_channel(f"127.0.0.1:{port}")
        try:
            stub = pb_grpc.EdgeCloudDataPlaneStub(channel)
            for mode in self.adapters:
                for index in range(2):
                    ticket = await self.ticket(mode,index=index)
                    reply = await stub.SubmitCloudInference(ticket_to_proto(ticket), timeout=2)
                    self.assertEqual(reply.status,"accepted")
                    result = result_from_proto(reply.result)
                    self.assertEqual(result.frame,ticket.frame)
                    self.assertEqual(result.model_mode,mode)
                    self.assertEqual(result.model_id,ticket.model_id)
                    self.assertEqual(result.boxes[0].track_id,42)
                    self.assertEqual(result.attributes["benchmark_completion"],"bytetrack_tracked")
                    self.assertGreater(result.attributes["benchmark_tracking_completed_mono_ns"],0)
                    self.assertIn("cloud_bytetrack_update_ms",result.attributes["benchmark_stages_ms"])
        finally:
            await channel.close()
            await server.stop(0)

    async def test_kafka_worker_new_modes_reply_after_tracking_with_same_codec(self):
        await self.setup_service()
        worker = KafkaCloudWorker(self.service,"unused",topic_namespace="benchmark")
        sent = []
        class Producer:
            async def send_and_wait(self, topic, *, key, value):
                sent.append((topic,json.loads(value)))
        worker._producer = Producer()
        for mode in self.adapters:
            ticket = await self.ticket(mode)
            # Kafka deserializer uses the same protobuf/domain conversion as gRPC.
            decoded = ticket_from_proto(pb.CloudInferenceTicket.FromString(ticket_to_proto(ticket).SerializeToString()))
            await worker._handle(CLOUD_TICKETS_TOPIC,decoded)
            # the shared topic (results writer) and the Edge's own copy carry the same reply (8/11)
            self.assertEqual(sorted(t for t, _ in sent[-2:]),
                             ["benchmark.cloud.detection-result.v1", f"benchmark.cloud.detection-result.v1.{ticket.frame.node_id}"])
            self.assertEqual(sent[-1][1], sent[-2][1])
            topic, envelope = sent[-1]
            self.assertEqual(envelope["status"],"accepted")
            result = result_from_proto(pb.DetectionResult.FromString(bytes.fromhex(envelope["result"])))
            self.assertEqual(result.frame,ticket.frame)
            self.assertEqual(result.model_mode,mode)
            self.assertEqual(result.boxes[0].track_id,42)
            self.assertEqual(result.attributes["benchmark_completion"],"bytetrack_tracked")

    async def test_unloaded_or_wrong_identity_rejected_before_inference(self):
        await self.setup_service()
        ticket = await self.ticket(ModelMode.RTDETR_CLOUD)
        status, _, _ = await self.service.handle_cloud_ticket(replace(ticket,model_id="wrong"))
        self.assertEqual(status,"model_identity_mismatch")
        self.assertEqual(self.adapters[ModelMode.RTDETR_CLOUD].seen,[])
        status, _, _ = await self.service.handle_cloud_ticket(replace(ticket,model_mode=ModelMode.DFINE_CLOUD,
            frame=replace(ticket.frame,stream_id="unloaded")))
        self.assertEqual(status,"unsupported_model")

    async def test_imported_candidate_id_survives_tracking_and_rejects_sp2_alias(self):
        await self.setup_service()
        mode = ModelMode.RTDETR_CLOUD
        candidate_id = "rtdetrv2_s_candidate_265508cc3dc2"
        self.adapters[mode].model_id = candidate_id
        self.service.benchmark_identities[mode]["model_id"] = candidate_id
        ticket = await self.ticket(mode)
        status, result, _ = await self.service.handle_cloud_ticket(ticket)
        self.assertEqual(status, "accepted")
        self.assertEqual(result.model_id, candidate_id)
        self.assertEqual(result.boxes[0].track_id, 42)
        self.assertEqual(result.attributes["benchmark_completion"], "bytetrack_tracked")
        next_ticket = await self.ticket(mode, index=1)
        status, _, _ = await self.service.handle_cloud_ticket(replace(next_ticket, model_id="rtdetrv2_r18vd_sp2_120e_coco"))
        self.assertEqual(status, "model_identity_mismatch")

    def test_wire_enums_are_additive_and_all_modes_roundtrip(self):
        self.assertEqual((pb.YOLO_EDGE,pb.RTMDET_EDGE,pb.DFINE_CLOUD,pb.RTDETR_CLOUD,pb.LWDETR_CLOUD),(1,2,3,4,5))
        frame = FrameIdentity("edge","cam","stream",1,1,time_ns(),33)
        for mode in ModelMode:
            value = DetectionResult(frame,mode,"model","decision",(),1,time_ns())
            self.assertEqual(result_from_proto(result_to_proto(value)),value)
            if mode.placement.value == "cloud":
                tensor = b"CUT1" if mode is ModelMode.RTDETR_SPLIT_CLOUD else b""  # path B tickets carry a cut tensor
                ticket = CloudInferenceTicket(frame,mode,"model","decision",33,time_ns()+10**9,tensor=tensor)
                self.assertEqual(ticket_from_proto(ticket_to_proto(ticket)),ticket)

    async def test_lwdetr_imported_identity_survives_cloud_bytetrack(self):
        await self.setup_service()
        mode = ModelMode.LWDETR_CLOUD
        candidate_id = "lwdetr_large_candidate_4f2bb0afdd7f"
        self.adapters[mode].model_id = candidate_id
        self.service.benchmark_identities[mode]["model_id"] = candidate_id
        ticket = await self.ticket(mode)
        status, result, _ = await self.service.handle_cloud_ticket(ticket)
        self.assertEqual(status, "accepted")
        self.assertEqual(result.model_id, candidate_id)
        self.assertEqual(result.boxes[0].track_id, 42)
        self.assertEqual(result.attributes["benchmark_completion"], "bytetrack_tracked")
        wrong = replace(await self.ticket(mode, index=1), model_id="lwdetr_large_coco")
        status, _, _ = await self.service.handle_cloud_ticket(wrong)
        self.assertEqual(status, "model_identity_mismatch")

    def test_benchmark_identity_uses_actual_new_adapter_not_dfine_environment(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifact = Path(tmp)/"candidate.onnx"
            artifact.write_bytes(b"test")
            adapter = Adapter("candidate")
            adapter.artifact_path = artifact
            adapter.session = type("Session",(),{"get_providers":lambda self: ["test_only"]})()
            adapter.settings = {"kind":"lwdetr_large","top_k":300,"confidence":.1}
            service = type("Service",(),{"cloud_adapters":{ModelMode.LWDETR_CLOUD:adapter},"benchmark_identities":{},"max_cloud_inflight":2})()
            with patch("benchmark.model.source_revision",return_value={"commit":"test"}):
                configure(service)
            identity = service.benchmark_identities[ModelMode.LWDETR_CLOUD]
            self.assertEqual(identity["settings"],adapter.settings)
            self.assertTrue(identity["artifact_sha256"])
            self.assertNotIn("DFINE_TOP_K",identity["settings"])


if __name__ == "__main__":
    unittest.main()
