"""Path B (v4 step 3): cut-tensor wire format, ticket contract, Cloud service routing and the split adapter."""
from __future__ import annotations

from dataclasses import replace
from time import time_ns
import unittest

import numpy as np

from cloud_core import ResultHub
from cloud_core.grpc_service import CloudDataPlaneServicer
from contracts import Box, CloudInferenceTicket, FrameIdentity, ModelMode
from contracts.cut_tensor import pack, unpack
from contracts.grpc_codec import frame_to_proto, result_from_proto, ticket_from_proto, ticket_to_proto
from contracts.proto import edge_video_platform_pb2 as pb
from inference.triton_detr import RtdetrSplitAdapter

FRAME = FrameIdentity(node_id="edge-246", camera_id="cam-1", stream_id="s1", stream_epoch=0, source_frame_id=9,
                      capture_utc_ns=1_790_000_000_000_000_000, source_pts=600_000_000)


def cut(seed=0):
    rng = np.random.default_rng(seed)
    t = np.maximum(rng.standard_normal((1, 128, 80, 80), dtype=np.float32), 0)  # ReLU-like, ~50 % zeros
    return t


class CutTensorTests(unittest.TestCase):
    def test_round_trip_is_within_half_a_step_and_compresses(self):
        t = cut()
        data = pack(t, width=960, height=540, head_id="rt-l2-w8a8pc")
        back, w, h = unpack(data, expected_head_id="rt-l2-w8a8pc")
        step = (t.max() - t.min()) / 255
        self.assertEqual((back.shape, back.dtype, w, h), (t.shape, np.float32, 960, 540))
        self.assertLessEqual(float(np.abs(back - t).max()), step / 2 + 1e-6)
        self.assertLess(len(data), t.size)  # zeros compress below one byte per value

    def test_rejects_foreign_head_truncated_and_garbage(self):
        data = pack(cut(), width=960, height=540, head_id="rt-l2-w8a8pc")
        with self.assertRaisesRegex(ValueError, "expected"):
            unpack(data, expected_head_id="rt-l2-w8a16pc")
        with self.assertRaises(Exception):
            unpack(data[:-100], expected_head_id="rt-l2-w8a8pc")
        with self.assertRaisesRegex(ValueError, "not a cut tensor"):
            unpack(b"JUNK" + data[4:], expected_head_id="rt-l2-w8a8pc")

    def test_ticket_contract_and_protobuf_round_trip(self):
        base = dict(frame=FRAME, model_id="split", decision_id="w", preview_pts=1, deadline_utc_ns=1_790_000_000_500_000_000)
        data = pack(cut(), width=960, height=540, head_id="h")
        ticket = CloudInferenceTicket(**base, model_mode=ModelMode.RTDETR_SPLIT_CLOUD, tensor=data)
        self.assertEqual(ticket_from_proto(ticket_to_proto(ticket)), ticket)
        for bad in (dict(model_mode=ModelMode.RTDETR_SPLIT_CLOUD),                       # split without tensor
                    dict(model_mode=ModelMode.RTDETR_CLOUD, tensor=data),                 # tensor on a full model
                    dict(model_mode=ModelMode.RTDETR_SPLIT_CLOUD, tensor=data, jpeg=b"\xff\xd8x"),
                    dict(model_mode=ModelMode.RTDETR_SPLIT_CLOUD, tensor=b"x" * 2_000_001)):
            with self.assertRaises(ValueError):
                CloudInferenceTicket(**base, **bad)


class SplitAdapterTests(unittest.TestCase):
    def test_tail_gets_the_tensor_and_source_size_and_boxes_are_decoded(self):
        seen = {}

        class FakeTail:
            def run(self, names, feed):
                seen.update(feed)
                scores = np.zeros((1, 300), np.float32)
                scores[0, 0] = .9
                boxes = np.zeros((1, 300, 4), np.float32)
                boxes[0, 0] = [96, 54, 480, 270]  # pixels of the 960x540 source
                return [dict(labels=np.full((1, 300), 2, np.int64), boxes=boxes, scores=scores)[n] for n in names]

        t = cut(1)
        adapter = RtdetrSplitAdapter(head_id="h", session=FakeTail(), confidence=.5)
        boxes, ms = adapter.infer(pack(t, width=960, height=540, head_id="h"))
        self.assertEqual(seen["orig_target_sizes"].tolist(), [[960, 540]])
        lo, hi = seen["cut_range"][0]
        deq = seen["cut_u8"].astype(np.float32) * ((hi - lo) / 255) + lo  # what the uint8 tail computes on the GPU
        self.assertEqual(seen["cut_u8"].dtype, np.uint8)
        self.assertLess(float(np.abs(deq - t).max()), (t.max() - t.min()) / 255)
        self.assertEqual(len(boxes), 1)
        self.assertAlmostEqual(boxes[0].left, .1, places=4)
        self.assertAlmostEqual(boxes[0].bottom, .5, places=4)
        with self.assertRaisesRegex(ValueError, "expected"):
            adapter.infer(pack(t, width=960, height=540, head_id="other"))
        with self.assertRaisesRegex(ValueError, "HEAD_ID"):
            RtdetrSplitAdapter(head_id="", session=FakeTail())


class SplitServiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_tensor_ticket_reaches_the_split_adapter_without_rtsp_or_jpeg(self):
        seen = []

        class FakeSplit:
            model_id = "split"
            runtime = "triton"

            def infer(self, payload):
                seen.append(payload)
                return [Box(0.1, 0.2, 0.4, 0.5, 0.9, 2, "car")], 2.0

        class NoCache:
            async def wait_for(self, ticket):
                raise AssertionError("a tensor ticket must not wait for the RTSP frame cache")

        servicer = CloudDataPlaneServicer(hub=ResultHub(), frame_cache=NoCache(), frame_taps=object(),
                                          cloud_adapters={ModelMode.RTDETR_SPLIT_CLOUD: FakeSplit()}, preview_fps=15)
        now = time_ns()
        data = pack(cut(), width=960, height=540, head_id="h")
        reply = await servicer.SubmitCloudInference(pb.CloudInferenceTicket(
            schema_version=2, frame=frame_to_proto(replace(FRAME, capture_utc_ns=now)), model_mode=pb.RTDETR_SPLIT_CLOUD,
            model_id="split", decision_id="w", preview_pts=600_000_000, deadline_utc_ns=now + 30_000_000_000,  # cold ByteTrack import can take > 1 s
            request_id="r9", tensor=data), None)
        self.assertEqual(reply.status, "accepted", reply.message)
        self.assertEqual(seen, [data])
        result = result_from_proto(reply.result)
        self.assertEqual((result.attributes["frame_source"], result.attributes["media_encoded_seq"]), ("tensor", 9))


if __name__ == "__main__":
    unittest.main()


class EdgeHeadTests(unittest.TestCase):
    def test_npu_nhwc_output_is_packed_as_nchw_with_the_source_size(self):
        from inference.rtdetr_head_snpe import HEAD_ID, OUTPUT, RtdetrHeadSnpe
        nchw = cut(2)

        class FakeDaemon:
            def infer_raw(self, x):
                assert x.shape == (1, 640, 640, 3) and x.dtype == np.float32 and float(x.max()) <= 1.0
                return 0.0, {OUTPUT: nchw.transpose(0, 2, 3, 1).ravel()}  # the daemon returns flat NHWC floats

        head = RtdetrHeadSnpe(__file__, session=FakeDaemon())
        data, stages = head.encode(np.full((540, 960, 3), 128, np.uint8))
        back, w, h = unpack(data, expected_head_id=HEAD_ID)
        self.assertEqual((w, h), (960, 540))
        self.assertLessEqual(float(np.abs(back - nchw).max()), (nchw.max() - nchw.min()) / 255)
        self.assertEqual(set(stages), {"prepare_ms", "npu_ms", "pack_ms"})

    def test_concurrent_frames_never_read_each_others_shared_memory_output(self):
        from concurrent.futures import ThreadPoolExecutor
        import time as _time
        from inference.rtdetr_head_snpe import HEAD_ID, OUTPUT, RtdetrHeadSnpe
        shared = np.zeros(80 * 80 * 128, np.float32)  # one output buffer, like the daemon's shared memory

        class FakeDaemon:
            def infer_raw(self, x):
                shared[:] = float(x.max()) * 100 + np.arange(shared.size, dtype=np.float32) % 7  # value tied to this frame
                return 0.0, {OUTPUT: shared}

        head = RtdetrHeadSnpe(__file__, session=FakeDaemon())

        def one(level):
            data, _ = head.encode(np.full((54, 96, 3), level, np.uint8))
            _time.sleep(0)
            back, _, _ = unpack(data, expected_head_id=HEAD_ID)
            return abs(float(back.min()) - level / 255 * 100) < 1e-3

        with ThreadPoolExecutor(8) as ex:
            self.assertTrue(all(ex.map(one, [10, 200, 60, 250, 120, 30, 180, 90] * 5)))



class GpuHeadTests(unittest.TestCase):
    def test_gpu_head_packs_nchw_with_its_own_id_and_the_cloud_accepts_both_heads(self):
        from types import SimpleNamespace
        from inference.rtdetr_head_snpe import HEAD_ID, ONNX_HEAD_ID, OUTPUT, RtdetrHeadOnnx, open_head
        nchw = cut(3)
        seen = []

        class FakeOrt:
            def infer_raw(self, x):
                seen.append(x.shape)
                return 0.0, {OUTPUT: nchw}
        head = RtdetrHeadOnnx(__file__, session=FakeOrt())
        data, _ = head.encode(np.full((540, 960, 3), 50, np.uint8))
        self.assertEqual(seen, [(1, 3, 640, 640)])
        back, w, h = unpack(data, expected_head_id=(HEAD_ID, ONNX_HEAD_ID))
        self.assertEqual((w, h), (960, 540))
        self.assertLessEqual(float(np.abs(back - nchw).max()), (nchw.max() - nchw.min()) / 255)
        with self.assertRaisesRegex(ValueError, "expected"):
            unpack(data, expected_head_id=HEAD_ID)          # an NPU-only Cloud refuses the GPU head
        adapter = RtdetrSplitAdapter(head_id=f"{HEAD_ID}, {ONNX_HEAD_ID}", session=SimpleNamespace(run=None))
        self.assertEqual(adapter.head_id, (HEAD_ID, ONNX_HEAD_ID))
        with self.assertRaises(FileNotFoundError):
            open_head(__file__ + ".onnx")                   # a missing file is refused, never guessed


class EdgeHeadCloseTests(unittest.TestCase):
    def test_close_stops_only_this_dlcs_daemon(self):
        from pathlib import Path
        from types import SimpleNamespace
        from unittest import mock
        from inference.rtdetr_head_snpe import RtdetrHeadSnpe
        import legacy.adapters.npu.snpe_daemon_manager as manager
        stopped = []
        head = RtdetrHeadSnpe(__file__, session=SimpleNamespace())
        mine = SimpleNamespace(dlc_path=str(Path(__file__).resolve()), close=lambda: stopped.append("mine"))
        other = SimpleNamespace(dlc_path="/other.dlc", close=lambda: stopped.append("other"))
        with mock.patch.dict(manager._DAEMONS, {"a": mine, "b": other}, clear=True):
            head.close()
        self.assertEqual(stopped, ["mine"])
