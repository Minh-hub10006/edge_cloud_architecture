"""Cloud models served by Triton (v4 main lane, step 3): concurrent calls from many cameras are batched by Triton
(max 4, 8 ms; triton_repo config.pbtxt). CLOUD_RUNTIME=triton switches the DETR adapters (inference/cloud_detr.py)
to this session; the split adapter (path B) only runs here.

Big inputs go through system shared memory (the Cloud service and the v4-triton container share 249; the container
runs with --ipc=host), one region per input per worker thread.
"""
from __future__ import annotations

import os
import threading
from time import perf_counter
from uuid import uuid4

import numpy as np

from contracts.cut_tensor import unpack_u8
from .detection_decode import rtdetr_decode
from .model_artifact import OUTPUT_NAMES

TRITON_MODELS = {"rtdetrv2_small": "rtdetr_full", "dfine_x": "dfinex_full"}
_DTYPES = {np.dtype(np.float32): "FP32", np.dtype(np.int64): "INT64", np.dtype(np.uint8): "UINT8"}
_SHM_MIN_BYTES = 1 << 16


class TritonSession:
    """Drop-in for the onnxruntime session's run(names, feed) used by CloudDetrAdapter."""

    def __init__(self, model: str, url: str | None = None):
        self.model, self.url = model, url or os.getenv("TRITON_URL", "localhost:18001")
        self._local = threading.local()
        self._regions: list[tuple] = []
        self._lock = threading.Lock()
        import tritonclient.grpc as grpcclient
        client = grpcclient.InferenceServerClient(self.url)
        if not client.is_model_ready(model):
            raise RuntimeError(f"Triton model {model} is not ready at {self.url}")
        client.close()

    def _thread_state(self):
        st = getattr(self._local, "state", None)
        if st is None:
            import tritonclient.grpc as grpcclient
            st = self._local.state = {"client": grpcclient.InferenceServerClient(self.url), "shm": {}}
        return st

    def run(self, names, feed):
        try:
            return self._run(names, feed)
        except Exception as exc:
            # 11/11: a Triton restart (adding a model; the container restarts itself too) forgets every registered
            # region: 715/1770 D-FINE-X frames failed with NOT_FOUND until the workers restarted. Register again, once.
            if "Unable to find shared memory region" not in str(exc):
                raise
            self._forget_thread_regions()
            return self._run(names, feed)

    def _forget_thread_regions(self):
        import tritonclient.utils.shared_memory as shm
        st = self._thread_state()
        mine = {key for key, _, _ in st["shm"].values()}
        with self._lock:
            for client, key, handle in [r for r in self._regions if r[1] in mine]:
                shm.destroy_shared_memory_region(handle)
            self._regions = [r for r in self._regions if r[1] not in mine]
        st["shm"].clear()

    def _run(self, names, feed):
        import tritonclient.grpc as grpcclient
        import tritonclient.utils.shared_memory as shm
        st = self._thread_state()
        inputs = []
        for name, x in feed.items():
            x = np.ascontiguousarray(x)
            t = grpcclient.InferInput(name, list(x.shape), _DTYPES[x.dtype])
            if x.nbytes >= _SHM_MIN_BYTES:
                region = st["shm"].get(name)
                if region is None or region[2] != x.nbytes:
                    key = f"v4_{self.model}_{uuid4().hex[:12]}"
                    handle = shm.create_shared_memory_region(key, f"/{key}", x.nbytes)
                    st["client"].register_system_shared_memory(key, f"/{key}", x.nbytes)
                    region = st["shm"][name] = (key, handle, x.nbytes)
                    with self._lock:
                        self._regions.append((st["client"], key, handle))
                shm.set_shared_memory_region(region[1], [x])
                t.set_shared_memory(region[0], x.nbytes)
            else:
                t.set_data_from_numpy(x)
            inputs.append(t)
        reply = st["client"].infer(self.model, inputs, outputs=[grpcclient.InferRequestedOutput(n) for n in names])
        return [reply.as_numpy(n) for n in names]

    def close(self):
        import tritonclient.utils.shared_memory as shm
        with self._lock:
            for client, key, handle in self._regions:
                try:
                    client.unregister_system_shared_memory(key)
                finally:
                    shm.destroy_shared_memory_region(handle)
            self._regions.clear()


class RtdetrSplitAdapter:
    """Path B Cloud half: cut tensor (contracts/cut_tensor.py) -> Triton rtdetr_tail_u8 -> boxes, decoded exactly like
    the full RT-DETRv2-S adapter. The tail takes the wire uint8 + (min, max) and dequantizes on the GPU
    (scripts/tail_uint8.py, bit-identical to the float tail), so the CPU only runs zstd. Refuses tensors from any
    head other than RTDETR_SPLIT_HEAD_ID."""

    kind = "rtdetrv2_small"
    runtime = "triton"

    def __init__(self, *, head_id: str, model_id: str = "rtdetrv2_r18vd_sp2_120e_coco+layer2_split",
                 confidence: float = 0.10, top_k: int = 300, session=None):
        heads = tuple(h.strip() for h in head_id.split(",") if h.strip()) if isinstance(head_id, str) else tuple(head_id)
        if not heads:
            raise ValueError("RTDETR_SPLIT_HEAD_ID required (comma list of the Edge heads cut from this tail's model)")
        if not 0 <= confidence <= 1 or type(top_k) is not int or not 1 <= top_k <= 300:
            raise ValueError("confidence must be 0..1 and top_k 1..300")
        self.head_id, self.model_id, self.confidence, self.top_k = heads, model_id, confidence, top_k
        self.session = session or TritonSession("rtdetr_tail_u8")

    @property
    def settings(self):
        return {"kind": "rtdetrv2_small_split", "head_id": ",".join(self.head_id), "cut": "layer2 (res_layers.1) uint8 in",
                "confidence": self.confidence, "top_k": self.top_k, "wire": "uint8+zstd (contracts/cut_tensor.py)",
                "label_map": "coco80_contiguous", "runtime": self.runtime}

    def infer(self, payload: bytes):
        boxes, elapsed, _ = self.infer_detailed(payload)
        return boxes, elapsed

    def infer_detailed(self, payload: bytes):
        started = perf_counter()
        q, lo, hi, width, height = unpack_u8(payload, expected_head_id=self.head_id)
        prepared = perf_counter()
        names = list(OUTPUT_NAMES[self.kind])
        out = dict(zip(names, self.session.run(names, {
            "cut_u8": q, "cut_range": np.array([[lo, hi]], dtype=np.float32),
            "orig_target_sizes": np.array([[width, height]], dtype=np.int64)})))
        inferred = perf_counter()
        boxes = rtdetr_decode(out, width, height, self.confidence, self.top_k)
        finished = perf_counter()
        return boxes, (finished - started) * 1000, {
            "cloud_preprocess_ms": (prepared - started) * 1000,
            "cloud_session_call_ms": (inferred - prepared) * 1000,
            "cloud_postprocess_ms": (finished - inferred) * 1000,
        }
