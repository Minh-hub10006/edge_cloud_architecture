"""v4 results store: Kafka result -> table rows, and the read-only check page (no database needed)."""
from __future__ import annotations

import json
import unittest

from contracts import Box, DetectionResult, FrameIdentity, ModelMode
from contracts.grpc_codec import result_to_proto

FRAME = FrameIdentity(node_id="edge-246", camera_id="cam-1", stream_id="s1", stream_epoch=5, source_frame_id=9,
                      capture_utc_ns=1_790_000_000_000_000_000, source_pts=600_000_000)


def envelope(boxes, decision="w", status="accepted"):
    result = DetectionResult(frame=FRAME, model_mode=ModelMode.RTDETR_SPLIT_CLOUD, model_id="split", decision_id=decision,
                             boxes=boxes, inference_ms=1.0, produced_utc_ns=FRAME.capture_utc_ns + 50_000_000,
                             request_id="r", preview_pts=600_000_000)
    return json.dumps({"status": status, "request_id": "r",
                       "result": result_to_proto(result).SerializeToString().hex()}).encode()


class WriterRowsTests(unittest.TestCase):
    def test_rows_match_the_table_columns_and_measurements_are_skipped(self):
        from cloud_core.results_writer import BOX_COLUMNS, FRAME_COLUMNS, rows_from_message
        box = Box(.1, .2, .3, .4, .9, 2, "car", track_id=7)
        frame, boxes = rows_from_message(envelope((box,)))
        self.assertEqual((len(frame), len(boxes[0])), (len(FRAME_COLUMNS), len(BOX_COLUMNS)))
        row = dict(zip(BOX_COLUMNS, boxes[0]))
        self.assertEqual((row["camera_id"], row["track_id"], row["class_name"], row["media_pts_ns"]),
                         ("cam-1", 7, "car", 600_000_000))
        self.assertAlmostEqual(row["x2"], .3, places=5)
        saved = dict(zip(FRAME_COLUMNS, frame))
        self.assertEqual(saved["box_count"], 1)
        self.assertAlmostEqual(saved["latency_ms"], 50.0)
        empty, none = rows_from_message(envelope(()))  # a frame without boxes still gets its frame row
        self.assertEqual((dict(zip(FRAME_COLUMNS, empty))["box_count"], none), (0, []))
        self.assertIsNone(rows_from_message(envelope((box,), decision="load/v4")))
        self.assertIsNone(rows_from_message(envelope((box,), decision="probe/x")))
        self.assertIsNone(rows_from_message(envelope((box,), status="deadline_exceeded")))

    def test_writer_reads_the_edge_tracked_topic(self):
        from cloud_core.results_writer import TOPICS
        from contracts.kafka_transport import EDGE_TRACKED_TOPIC
        self.assertIn(EDGE_TRACKED_TOPIC, TOPICS)

    def test_columns_exist_in_the_schema(self):
        from pathlib import Path
        from cloud_core.results_writer import BOX_COLUMNS, FRAME_COLUMNS
        schema = (Path(__file__).resolve().parents[1] / "scripts/db/schema.sql").read_text(encoding="utf-8")
        frames, detections = schema.split("CREATE TABLE IF NOT EXISTS detections")
        for column in FRAME_COLUMNS:
            self.assertIn(column, frames)
        for column in BOX_COLUMNS:
            self.assertIn(column, detections)


class FakeCursor:
    def __init__(self, seen):
        self.seen, self.rows, self.description = seen, [], []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    async def execute(self, sql, params=None):
        self.seen.append((sql, params))
        if "count(*)" in sql:
            self._set(["node_id", "camera_id", "stream_id", "stream_epoch", "frames"], [("edge-246", "cam-1", "s1", 5, 140)])
        else:
            self._set(["pts_ms", "track_id", "class_name"], [(900, 7, "car"), (900, 8, "bus"), (960, 7, "car")])

    def _set(self, names, rows):
        from types import SimpleNamespace
        self.description, self.rows = [SimpleNamespace(name=n) for n in names], rows

    async def fetchall(self):
        return self.rows

    async def fetchone(self):
        return self.rows[0] if self.rows else None


class CheckPageTests(unittest.TestCase):
    def client(self):
        from starlette.testclient import TestClient
        from web.check_app import create_app
        self.seen = []
        seen = self.seen

        class Conn:
            def cursor(self):
                return FakeCursor(seen)

            async def close(self):
                pass

        async def connect():
            return Conn()
        return TestClient(create_app(connect=connect, node_hosts={"edge-246": "192.168.180.246"}))

    def test_streams_carry_the_edge_video_url_and_recent_returns_boxes_per_frame(self):
        c = self.client()
        self.assertIn("Kiểm tra", c.get("/").text)
        self.assertNotIn("__TAG__", c.get("/").text)
        self.assertIn("decodeIdentity", c.get("/live_sync.js").text)
        (s,) = c.get("/api/streams").json()
        self.assertIn("192.168.180.246:8889/edge/edge-246/cam-1/s1/epoch-5", s["video_url"])
        r = c.get("/api/recent", params={"node": "edge-246", "camera": "cam-1", "stream": "s1", "epoch": 5}).json()
        self.assertEqual([f["pts_ms"] for f in r["frames"]], [900, 960])       # boxes grouped per frame, oldest first
        self.assertEqual(r["frames"][0]["boxes"][1], {"track_id": 8, "class_name": "bus"})
        self.assertEqual(self.seen[-1][1], ("edge-246", "cam-1", "s1", 5, -1000000))   # bound parameters, never pasted SQL
        c.get("/api/recent", params={"node": "edge-246", "camera": "cam-1", "stream": "s1", "epoch": 5, "after_ms": 960})
        self.assertEqual(self.seen[-1][1][-1], 960000000)                    # only frames newer than the page already has

    def test_bad_query_is_refused_before_touching_the_database(self):
        c = self.client()
        self.assertEqual(c.get("/api/recent", params={"node": "a", "camera": "b", "stream": "c", "epoch": "x;drop"}).status_code, 400)
        self.assertEqual(self.seen, [])


if __name__ == "__main__":
    unittest.main()


class CopyPublisherTests(unittest.IsolatedAsyncioTestCase):
    def publisher(self, seen):
        from types import SimpleNamespace
        from media.copy_publisher import CopyPublisher
        async def sink(reference):
            seen.append(reference)
        pub = CopyPublisher(profile=SimpleNamespace(fps=15), output="rtsp://x/y", stream_key=FRAME.stream_key,
                            main_video="main.mp4", on_frame_submitted=sink)
        pub._process = SimpleNamespace(returncode=None)
        return pub

    def test_command_forwards_without_encoding(self):
        from media.copy_publisher import build_copy_command
        cmd = build_copy_command("main.mp4", "rtsp://x/y", start_s=10)
        self.assertEqual(cmd[cmd.index("-c") + 1], "copy")
        self.assertIn("-re", cmd)
        self.assertEqual(cmd[cmd.index("-ss") + 1], "10.000")
        self.assertNotIn("libx264", cmd)

    async def test_publish_numbers_frames_and_a_dead_ffmpeg_is_an_error(self):
        from types import SimpleNamespace
        seen = []
        pub = self.publisher(seen)
        frame = SimpleNamespace(identity=FRAME)
        await pub.publish(frame)
        await pub.publish(frame)
        self.assertEqual([r.encoded_seq for r in seen], [0, 1])
        self.assertEqual(seen[1].media_pts_ns, round(1e9 / 15))
        pub.start_s = 10.0  # a run started at file frame 150: its boxes carry the number in the video's band
        await pub.publish(frame)
        self.assertEqual(seen[2].media_pts_ns, round(152 * 1e9 / 15))
        self.assertEqual(pub.frames_submitted, 3)
        pub._process.returncode = 1
        with self.assertRaisesRegex(RuntimeError, "FFmpeg exited"):
            await pub.publish(frame)

    async def test_a_looping_run_loops_the_file_and_wraps_its_frame_numbers(self):
        # 11/11: 180 s runs over 140 s files ended every agent when the main-stream FFmpeg reached the end
        from types import SimpleNamespace
        from media.copy_publisher import build_copy_command
        self.assertEqual(build_copy_command("m.mp4", "rtsp://x/y", loop=True)[4:7], ["-re", "-stream_loop", "-1"])
        self.assertNotIn("-stream_loop", build_copy_command("m.mp4", "rtsp://x/y"))
        seen = []
        pub = self.publisher(seen)
        pub.loop_frames, pub.start_s = 4, 2 / 15   # a 4-frame file, started at its frame 2
        for _ in range(4):
            await pub.publish(SimpleNamespace(identity=FRAME))
        self.assertEqual([round(r.media_pts_ns * 15 / 1e9) for r in seen], [2, 3, 0, 1])


class MainBandTests(unittest.TestCase):
    def test_band_written_for_a_main_file_decodes_to_its_frame_number(self):
        import numpy as np
        from media.copy_publisher import MAIN_KEY
        from media.frame_identity import append_identity, read_identity
        out = append_identity(np.zeros((540, 960, 3), np.uint8), key=MAIN_KEY, seq=1234, fps=15)
        self.assertEqual(out.shape[0], 604)
        self.assertEqual(read_identity(out, key=MAIN_KEY).seq, 1234)


class HwCaptureTests(unittest.TestCase):
    def test_stride_is_found_from_whole_frames(self):
        from media.hw_capture import find_stride
        line = '/GstPipeline:pipeline0/GstIdentity:identity0: last-message = chain   ******* (identity0:sink) ({} bytes, dts: none)'
        self.assertEqual(find_stride(chr(10).join([line.format(3317760)] * 4), 1080, 1920), 1152)   # 246: 1080 px rows -> 1152
        self.assertEqual(find_stride(line.format(3110400), 1080, 1920), 1080)
        for bad in (line.format(12345), line.format(3317760) + chr(10) + line.format(3110400), "no sizes"):
            with self.assertRaises(ValueError):
                find_stride(bad, 1080, 1920)

    def test_row_padding_never_reaches_the_picture(self):
        try:
            import av  # noqa: F401
        except ImportError:
            self.skipTest("PyAV not installed")
        import numpy as np
        from media.hw_capture import nv12_to_bgr
        rng = np.random.default_rng(0)
        w, h, stride = 96, 64, 128
        tight = rng.integers(16, 235, (h * 3 // 2, w), dtype=np.uint8)
        padded = np.full((h * 3 // 2, stride), 255, np.uint8)
        padded[:, :w] = tight
        a = nv12_to_bgr(padded.tobytes(), w, h, stride)
        b = nv12_to_bgr(tight.tobytes(), w, h, w)
        self.assertEqual(a.shape, (h, w, 3))
        self.assertTrue((a == b).all())

    def test_gpu_resized_frames_read_in_both_buffer_layouts(self):
        import numpy as np
        from media.hw_capture import small_bgr, transform_chain
        s, nbytes = 64, 64 * 256   # 246 for 640: 1638400 bytes = 640 rows of 2560
        img = np.random.default_rng(1).integers(1, 255, (s, s, 3), np.uint8)
        packed = np.zeros(nbytes, np.uint8)
        packed[:s * s * 3] = img.ravel()              # engine gles: pixels first, zero tail
        padded = np.zeros((s, nbytes // s), np.uint8)
        padded[:, :s * 3] = img.reshape(s, -1)        # fcv/ocv: every row padded
        for buf in (packed.tobytes(), padded.tobytes()):
            self.assertTrue((small_bgr(buf, s) == img).all())
        self.assertIn("engine=gles", transform_chain(640, engine_property=True))       # 246
        self.assertNotIn("engine=gles", transform_chain(640, engine_property=False))   # 252: older plugin
        self.assertIn("video/x-raw,format=BGR,width=640,height=640", transform_chain(640, engine_property=True))

    def test_a_decoder_that_stops_delivering_is_restarted_at_the_frame_it_was_on(self):
        # 12/11 on 252: a decoder gave no frame from ~110 s on, read() waited forever, the agent hit its run timeout
        import socket
        from collections import deque
        from types import SimpleNamespace
        from unittest import mock
        import numpy as np
        from media import hw_capture
        from media.hw_capture import HwCapture

        def fake_pipeline(frames):
            a, b = socket.socketpair()   # select() works on sockets everywhere (pipes: Linux only, like production)
            for f in frames:
                b.sendall(f)
            proc = SimpleNamespace(stdout=a.makefile("rb", buffering=0), poll=lambda: None, ends=(a, b))
            return proc
        img = np.full((2, 2, 3), 7, np.uint8)
        cap = HwCapture.__new__(HwCapture)
        cap.size, cap.frame_bytes, cap.hires_every, cap.index, cap.stalls = 2, 12, 0, 5, 0
        cap._err, cap._buf, cap.last_hires = deque(["qtic2vdec: no buffer"], maxlen=20), bytearray(12), None
        cap._proc = fake_pipeline([])                                     # stalled: nothing ever comes
        restarted_at = []

        def restart(prop, value):
            restarted_at.append(value)
            cap._proc, cap._buf = fake_pipeline([img.tobytes()]), bytearray(12)
            return True
        with mock.patch.object(hw_capture, "STALL_S", 0.05), mock.patch.object(cap, "set", side_effect=restart):
            ok, frame = cap.read()
        self.assertTrue(ok)
        self.assertTrue((frame == img).all())
        self.assertEqual((restarted_at, cap.stalls, cap.index), ([5], 1, 6))

    def test_a_hung_probe_is_tried_once_more_and_a_failed_one_is_not(self):
        # 12/11 on 246: one of 4 agents' probe hung 60 s while the others opened their decoders; that camera never ran
        from unittest import mock
        from media.hw_capture import HwCapture
        cap = HwCapture.__new__(HwCapture)
        hung = TimeoutError("hardware decoder probe hung for 60 s")
        with mock.patch("time.sleep"):
            with mock.patch.object(cap, "_probe_once", side_effect=[hung, "log"]) as once:
                self.assertEqual(cap._probe(["chain"]), "log")
            self.assertEqual(once.call_count, 2)
            with mock.patch.object(cap, "_probe_once", side_effect=[hung, hung]):
                with self.assertRaisesRegex(RuntimeError, "after one retry"):
                    cap._probe(["chain"])
            with mock.patch.object(cap, "_probe_once", side_effect=RuntimeError("probe failed (exit 1)")) as once:
                with self.assertRaises(RuntimeError):
                    cap._probe(["chain"])
            self.assertEqual(once.call_count, 1)

    def test_a_stopped_pipelines_hires_reader_never_marks_the_next_one_ended(self):
        # 12/11: the old reader wrote its end marker into the new pipeline's dict (restart, loop rewind)
        import io
        import threading
        from media.hw_capture import HwCapture
        cap = HwCapture.__new__(HwCapture)
        cap.hires_bytes, cap.width, cap.height = 12, 2, 2
        old, new = {}, {}
        cap._hires, cap._hires_cv = new, threading.Condition()
        cap._read_hires(io.BytesIO(b""), old, threading.Condition())   # the old pipeline ends
        self.assertIn(None, old)
        self.assertNotIn(None, new)

    def test_edge_yolo_gets_the_camera_aspect_back(self):
        import numpy as np
        from types import SimpleNamespace
        from edge_agent.control_v1 import unstretch
        square = np.zeros((640, 640, 3), np.uint8)
        self.assertEqual(unstretch(SimpleNamespace(pixels=square, source_size=(1080, 1920))).shape, (640, 360, 3))
        self.assertEqual(unstretch(SimpleNamespace(pixels=square, source_size=(1920, 1080))).shape, (360, 640, 3))
        self.assertIs(unstretch(SimpleNamespace(pixels=square, source_size=None)), square)

    def test_closing_an_own_npu_daemon_removes_its_shared_memory_folder(self):
        import os
        import tempfile
        from types import SimpleNamespace
        from legacy.adapters.npu.snpe_daemon_manager import SnpeDaemonHandle
        with tempfile.TemporaryDirectory() as base:
            shm = os.path.join(base, "shm_x")
            os.makedirs(shm)
            open(os.path.join(shm, "input"), "wb").write(b"0" * 1024)
            h = object.__new__(SnpeDaemonHandle)
            h.proc, h.socket_path, h.shm_dir = SimpleNamespace(poll=lambda: 0), os.path.join(base, "sock"), shm
            h.client = SimpleNamespace(shutdown=lambda: None)
            h.close()
            self.assertFalse(os.path.exists(shm))
            os.makedirs(shm)                 # a reused daemon of another process: its folder is left alone
            h.proc = None
            h.close()
            self.assertTrue(os.path.exists(shm))

    def test_gpu_resize_needs_a_hardware_decoder(self):
        from media.opencv_source import OpenCVFrameSource
        with self.assertRaises(ValueError):
            OpenCVFrameSource("v.mp4", node_id="n", camera_id="c", stream_id="s", hw_size=640)

    def test_gst_source_for_files_and_cameras(self):
        from media.hw_capture import gst_source
        self.assertEqual(gst_source("/v.mp4", "v4l2h264dec")[:3], ["filesrc", "location=/v.mp4", "!"])
        self.assertIn("rtph264depay", gst_source("rtsp://cam/1", "qtic2vdec"))
        self.assertIn("qtic2vdec", gst_source("rtsp://cam/1", "qtic2vdec"))
        self.assertIn("capture-io-mode=userptr", gst_source("/v.mp4", "v4l2h264dec", "userptr"))
        self.assertNotIn("capture-io-mode=userptr", gst_source("/v.mp4", "qtic2vdec", "userptr"))  # V4L2 only


@unittest.skipUnless(hasattr(__import__("os"), "set_blocking") and __import__("sys").platform != "win32", "POSIX signals")
class HwStopTests(unittest.TestCase):
    def test_stop_sends_sigint_and_drains_so_the_child_can_finish(self):
        import subprocess
        import sys
        from media.hw_capture import stop_gracefully
        child = chr(10).join([
            'import signal, sys, time',
            'done = []',
            'signal.signal(signal.SIGINT, lambda *a: done.append(1))',
            'while not done:',
            "    sys.stdout.buffer.write(b'x' * 65536); sys.stdout.flush()",
            'sys.exit(7)',
        ])
        p = subprocess.Popen([sys.executable, "-c", child], stdout=subprocess.PIPE, bufsize=0)
        stop_gracefully(p, timeout_s=5)
        self.assertEqual(p.returncode, 7)   # it ended by itself after SIGINT, not killed


class NodeResultsTopicTests(unittest.IsolatedAsyncioTestCase):
    async def test_each_edge_gets_its_own_copy_even_for_rejections(self):
        from types import SimpleNamespace
        from contracts.kafka_transport import CLOUD_RESULTS_TOPIC, KafkaCloudWorker, node_results_topic
        sent = []

        class Producer:
            async def send_and_wait(self, topic, *, key, value):
                sent.append((topic, key))
        worker = KafkaCloudWorker(service=None, bootstrap_servers="fake")
        worker._producer = Producer()
        ticket = SimpleNamespace(frame=FRAME, request_id="r")
        await worker._publish_result("r", "deadline_exceeded", None, "late", source=ticket)
        key = b"edge-246/cam-1/s1/5"
        self.assertEqual(sorted(sent), [(CLOUD_RESULTS_TOPIC, key), (CLOUD_RESULTS_TOPIC + ".edge-246", key)])
        with self.assertRaises(ValueError):
            node_results_topic(CLOUD_RESULTS_TOPIC, "bad node/x")
