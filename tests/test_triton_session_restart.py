"""TritonSession survives a Triton restart (11/11: registered shared-memory regions are forgotten -> NOT_FOUND)."""
from __future__ import annotations

import sys
import types
import unittest
from unittest import mock

import numpy as np


class FakeServer:
    def __init__(self):
        self.registered, self.calls = set(), 0

    def client(self, url):
        server = self

        class Client:
            def is_model_ready(self, model):
                return True

            def close(self):
                pass

            def register_system_shared_memory(self, key, path, size):
                server.registered.add(key)

            def unregister_system_shared_memory(self, key):
                server.registered.discard(key)

            def infer(self, model, inputs, outputs=None):
                server.calls += 1
                for t in inputs:
                    if t.region and t.region not in server.registered:
                        raise RuntimeError(f"[StatusCode.NOT_FOUND] Unable to find shared memory region: '{t.region}'")
                return types.SimpleNamespace(as_numpy=lambda name: np.ones(2, np.float32))
        return Client()


def fake_modules(server):
    class InferInput:
        def __init__(self, name, shape, dtype):
            self.region = None

        def set_shared_memory(self, key, size):
            self.region = key

        def set_data_from_numpy(self, x):
            pass
    grpc = types.SimpleNamespace(InferenceServerClient=server.client, InferInput=InferInput,
                                 InferRequestedOutput=lambda name: name)
    shm = types.SimpleNamespace(create_shared_memory_region=lambda key, path, size: key,
                                set_shared_memory_region=lambda handle, xs: None,
                                destroy_shared_memory_region=lambda handle: None)
    utils = types.SimpleNamespace(shared_memory=shm)
    root = types.SimpleNamespace(grpc=grpc, utils=utils)
    return {"tritonclient": root, "tritonclient.grpc": grpc, "tritonclient.utils": utils,
            "tritonclient.utils.shared_memory": shm}


class RestartTests(unittest.TestCase):
    def test_a_restarted_server_gets_the_region_again_and_the_call_succeeds(self):
        server = FakeServer()
        with mock.patch.dict(sys.modules, fake_modules(server)):
            from inference.triton_detr import TritonSession
            s = TritonSession("dfinex_full", "x:1")
            big = {"images": np.zeros((1, 3, 640, 640), np.float32)}     # > 64 KB: goes through shared memory
            s.run(["out"], big)
            server.registered.clear()                                     # Triton restarted
            out = s.run(["out"], big)
            self.assertEqual(out[0].tolist(), [1.0, 1.0])
            self.assertEqual((server.calls, len(server.registered), len(s._regions)), (3, 1, 1))

    def test_batches_of_different_sizes_keep_one_region_per_input(self):
        # 14/11: UC4 pose batches of 1-4 crops made a new region per size and never freed the old (~12 GB)
        server = FakeServer()
        with mock.patch.dict(sys.modules, fake_modules(server)):
            from inference.triton_detr import TritonSession
            s = TritonSession("pose_rtmo_s", "x:1")
            for n in (1, 2, 4, 3, 1, 4, 2):
                s.run(["out"], {"input": np.zeros((n, 3, 640, 640), np.float32)})
            self.assertEqual((len(server.registered), len(s._regions)), (1, 1))   # grown to the batch of 4, reused

    def test_other_errors_are_not_retried(self):
        server = FakeServer()
        with mock.patch.dict(sys.modules, fake_modules(server)):
            from inference.triton_detr import TritonSession
            s = TritonSession("m", "x:1")
            with mock.patch.object(s, "_run", side_effect=RuntimeError("[StatusCode.UNAVAILABLE] queue full")) as run:
                with self.assertRaises(RuntimeError):
                    s.run(["out"], {"x": np.zeros(4, np.float32)})
            self.assertEqual(run.call_count, 1)


if __name__ == "__main__":
    unittest.main()
