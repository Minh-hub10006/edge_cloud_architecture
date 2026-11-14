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
        from dataclasses import replace
        frame = lambda k: SimpleNamespace(identity=replace(FRAME, source_frame_id=k))
        await pub.publish(frame(0))
        await pub.publish(frame(1))
        self.assertEqual([r.encoded_seq for r in seen], [0, 1])
        self.assertEqual(seen[1].media_pts_ns, round(1e9 / 15))
        pub.start_s = 10.0  # a run started at file frame 150: its boxes carry the number in the video's band
        await pub.publish(frame(5))   # frames 2-4 were dropped by a late source: the video's frame 155, not 152
        self.assertEqual(seen[2].media_pts_ns, round(155 * 1e9 / 15))
        self.assertEqual(pub.frames_submitted, 3)
        pub._process.returncode = 1
        with self.assertRaisesRegex(RuntimeError, "FFmpeg exited"):
            await pub.publish(frame(6))

    async def test_a_looping_run_loops_the_file_and_wraps_its_frame_numbers(self):
        # 11/11: 180 s runs over 140 s files ended every agent when the main-stream FFmpeg reached the end
        from types import SimpleNamespace
        from media.copy_publisher import build_copy_command
        self.assertEqual(build_copy_command("m.mp4", "rtsp://x/y", loop=True)[4:7], ["-re", "-stream_loop", "-1"])
        self.assertNotIn("-stream_loop", build_copy_command("m.mp4", "rtsp://x/y"))
        seen = []
        pub = self.publisher(seen)
        pub.loop_frames, pub.start_s = 4, 2 / 15   # a 4-frame file, started at its frame 2
        from dataclasses import replace
        for k in range(4):
            await pub.publish(SimpleNamespace(identity=replace(FRAME, source_frame_id=k)))
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
        cap.failures = cap.restarts = 0
        cap.dead, cap.frame_count = False, 100
        cap._err, cap._buf, cap.last_hires = deque(["qtic2vdec: no buffer"], maxlen=20), bytearray(12), None
        cap._proc = fake_pipeline([])                                     # stalled: nothing ever comes
        restarted_at = []

        def restart(prop, value):
            restarted_at.append(value)
            cap._proc, cap._buf = fake_pipeline([img.tobytes()]), bytearray(12)
            return True
        with mock.patch.object(hw_capture, "STALL_S", 0.05), mock.patch.object(cap, "set", side_effect=restart), \
                mock.patch.object(hw_capture, "stop_gracefully", return_value=True) as stop, mock.patch("time.sleep"):
            ok, frame = cap.read()
        self.assertTrue(ok)
        self.assertTrue((frame == img).all())
        self.assertEqual((restarted_at, cap.stalls, cap.index, cap.restarts), ([5], 1, 6, 1))
        self.assertEqual(stop.call_args.kwargs, {"timeout_s": 15, "force": False})   # never killed (ghost sessions)

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
            busy = RuntimeError("probe failed (exit 1): Device or resource busy. Check the device (an ENOMEM ...)")
            with mock.patch.object(cap, "_probe_once", side_effect=[busy, "log"]) as once:   # 12/11 on 246 and 252
                self.assertEqual(cap._probe(["chain"]), "log")
            self.assertEqual(once.call_count, 2)
            oom = RuntimeError("probe failed (exit 1): vb2_streamon failed, -12. Check the device (an ENOMEM ...)")
            with mock.patch.object(cap, "_probe_once", side_effect=oom) as once:
                with self.assertRaises(RuntimeError):
                    cap._probe(["chain"])
            self.assertEqual(once.call_count, 1)   # out of memory: only a reboot helps

    def test_restart_at_a_frame_skips_whole_frames_even_when_the_pipe_gives_small_pieces(self):
        # 12/11: one read(frame_bytes) on the unbuffered pipe returns <= 64 KB, so a restart at frame > 0 stopped at once,
        # index back at 0, stream cut mid-frame -> shifted garbage pictures, no boxes (246 and 252)
        from types import SimpleNamespace
        from unittest import mock
        from media.hw_capture import CAP_PROP_POS_FRAMES, HwCapture
        frames = b"".join(bytes([k]) * 12 for k in range(4))

        class Pipe:   # like a raw pipe: never more than 5 bytes per read
            def __init__(self):
                self.pos = 0

            def read(self, n):
                piece = frames[self.pos:self.pos + min(n, 5)]
                self.pos += len(piece)
                return piece
        cap = HwCapture.__new__(HwCapture)
        cap.frame_bytes, cap.index, cap.dead = 12, 0, False

        def start(seek_frame=0):
            cap._proc, cap.index = SimpleNamespace(stdout=Pipe()), 0
        with mock.patch.object(cap, "release"), mock.patch.object(cap, "_start", side_effect=start), \
                mock.patch.object(cap, "_readable", return_value=True):
            self.assertTrue(cap.set(CAP_PROP_POS_FRAMES, 3))
            self.assertEqual((cap.index, cap._proc.stdout.pos), (3, 36))   # exactly at the start of frame 3
            self.assertFalse(cap.set(CAP_PROP_POS_FRAMES, 9))              # the file has only 4 frames

    def test_a_decoder_that_exits_before_the_end_of_the_file_restarts_where_it_was(self):
        # 12/11: decoders exited at 22 s / 91 s; the source took it for the end of the file and the camera stopped
        from collections import deque
        from types import SimpleNamespace
        from unittest import mock
        from media.hw_capture import HwCapture
        from media import hw_capture
        cap = HwCapture.__new__(HwCapture)
        cap.frame_count, cap.index, cap.failures, cap.restarts, cap.dead = 2100, 329, 0, 0, False
        cap._err, cap._proc = deque(["poll error"], maxlen=20), SimpleNamespace(poll=lambda: 1)
        with mock.patch("time.sleep") as sleep, mock.patch.object(cap, "set", return_value=True) as seek, \
                mock.patch.object(hw_capture, "stop_gracefully", return_value=True):
            self.assertTrue(cap._restart_after_failure())
            seek.assert_called_once_with(1, 329)
            sleep.assert_called_once_with(2)                  # the driver closes the old session first
            cap.index = 2099                                  # the real end of the file: the source rewinds itself
            self.assertFalse(cap._restart_after_failure())
            cap.index = 500
            self.assertTrue(cap._restart_after_failure())
            self.assertTrue(cap._restart_after_failure())
            self.assertFalse(cap._restart_after_failure())    # 4th restart: over budget, the camera stops
        self.assertTrue(cap.dead)
        self.assertFalse(cap.set(1, 0))                       # and the loop rewind opens nothing either
        self.assertEqual(cap.read(), (False, None))

    def test_a_decoder_that_will_not_stop_is_never_replaced_by_a_new_session(self):
        # 12/11 on 246: msm_vidc "max limit 16 already running 16 sessions" - killed / unstopped decoders leaked them
        from collections import deque
        from types import SimpleNamespace
        from unittest import mock
        from media import hw_capture
        from media.hw_capture import HwCapture
        cap = HwCapture.__new__(HwCapture)
        cap.frame_count, cap.index, cap.failures, cap.restarts, cap.dead = 2100, 50, 0, 0, False
        cap._err, cap._proc = deque([], maxlen=20), SimpleNamespace(poll=lambda: None)
        with mock.patch.object(hw_capture, "stop_gracefully", return_value=False), \
                mock.patch.object(cap, "set") as seek, mock.patch("time.sleep"):
            self.assertFalse(cap._restart_after_failure())
        seek.assert_not_called()
        self.assertTrue(cap.dead)

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


class KeyframeTests(unittest.TestCase):
    def test_keyframes_in_display_order_from_ffprobe_csv(self):
        from media.hw_capture import parse_keyframes
        csv = "0.000000,K__\n0.066667,___\n0.133333,___\n2.000000,K__\n0.200000,___\nbad line\n2.066667,___\n"
        self.assertEqual(parse_keyframes(csv), [(0, 0.0), (4, 2.0)])   # sorted by time: the 2 s keyframe is frame 4
