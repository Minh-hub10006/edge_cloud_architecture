from __future__ import annotations

import importlib.util
from pathlib import Path
import shutil
import subprocess
from unittest.mock import patch
import unittest

from web.frame_sync import frame_pts_tolerance_ns, select_frame_metadata
from web.overlay_page import render_overlay_page


KEY = ("edge-01", "cam-01", "preview", 2)


class FrameSyncTests(unittest.TestCase):
    def test_matches_only_nearest_metadata_within_half_frame(self) -> None:
        event = {"source_frame_id": 12}
        self.assertEqual(
            select_frame_metadata(1_000_000_000, {1_000_000_000: event}, fps=15),
            (1_000_000_000, event),
        )
        self.assertEqual(frame_pts_tolerance_ns(15), 33_333_333)

    def test_rejects_stale_or_future_metadata_outside_display_frame(self) -> None:
        event = {"source_frame_id": 12}
        self.assertIsNone(select_frame_metadata(1_100_000_000, {1_000_000_000: event}, fps=15))
        self.assertIsNone(select_frame_metadata(1_000_000_000, {1_100_000_000: event}, fps=15))

    def test_rejects_invalid_fps_and_pts(self) -> None:
        for fps in (0, -1, 241):
            with self.assertRaises(ValueError):
                frame_pts_tolerance_ns(fps)
        with self.assertRaises(ValueError):
            select_frame_metadata(-1, {}, fps=15)


class OverlayPageTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("node"), "Node unavailable")
    def test_track_smoothing_and_raw_review(self):
        root = Path(__file__).resolve().parents[1]
        result = subprocess.run(["node", str(root / "tests/test_box_smoothing.cjs")],
                                capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_display_smoothing_is_opt_in_and_bounded(self):
        html = render_overlay_page(KEY, media_base="http://localhost", websocket_path="/ws")
        self.assertIn('"smoothMs":0', html)
        self.assertIn('"review":false', html)
        enabled = render_overlay_page(KEY, media_base="http://localhost", websocket_path="/ws", smooth_ms=120, review=True)
        self.assertIn('"smoothMs":120', enabled)
        self.assertIn('"review":true', enabled)
        for value in (-1, 201):
            with self.assertRaises(ValueError):
                render_overlay_page(KEY, media_base="http://localhost", websocket_path="/ws", smooth_ms=value)

    @unittest.skipUnless(shutil.which("node"), "Node unavailable")
    def test_javascript_timeline_regressions(self) -> None:
        root = Path(__file__).resolve().parents[1]
        result = subprocess.run(["node", str(root / "tests/test_live_sync.cjs")],
                                capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    @unittest.skipUnless(shutil.which("node"), "Node unavailable")
    def test_rendered_page_pool_and_connection_lifecycle(self) -> None:
        html = render_overlay_page(KEY, media_base="http://localhost:8889", fps=30, websocket_path="/ws")
        root = Path(__file__).resolve().parents[1]
        result = subprocess.run(["node", str(root / "tests/test_overlay_runtime.cjs")],
                                input=html, encoding="utf-8", capture_output=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_page_uses_whep_websocket_pts_join_and_normalized_box_canvas(self) -> None:
        html = render_overlay_page(
            KEY, media_base="http://192.0.2.1:8889", fps=15,
            websocket_path="/v1/streams/edge-01/cam-01/preview/2/metadata",
        )
        self.assertIn("http://192.0.2.1:8889/edge/edge-01/cam-01/preview/epoch-2/whep", html)
        self.assertIn('"websocketPath":"/v1/streams/edge-01/cam-01/preview/2/metadata"', html)
        self.assertNotIn("meta.mediaTime*1e9", html)
        self.assertIn("decodeIdentity", html)
        self.assertIn("timeline.present(now)", html)
        self.assertIn("event.preview_pts", html)
        self.assertIn("b.left*display.width", html)
        self.assertIn("Video PTS:", html)
        self.assertIn("Metadata PTS:", html)
        self.assertIn("No detection for this frame", html)
        self.assertIn('"allowSmoke":false', html)

    @unittest.skipUnless(importlib.util.find_spec("starlette"), "Starlette unavailable")
    def test_synthetic_overlay_smoke_requires_server_opt_in(self) -> None:
        from starlette.testclient import TestClient
        from web.cloud_app import create_app

        path = "/v1/streams/edge-01/cam-01/preview/2/live?smoke=1"
        with patch.dict("os.environ", {}, clear=True), TestClient(create_app()) as client:
            self.assertIn('"allowSmoke":false', client.get(path).text)
        with patch.dict("os.environ", {"CLOUD_ENABLE_OVERLAY_SMOKE": "1"}), TestClient(create_app()) as client:
            # Fake browser-clock PTS cannot validate real synchronization.
            self.assertIn('"allowSmoke":false', client.get(path).text)
            self.assertIn('"allowSmoke":false', client.get(path[:-8]).text)

    def test_embed_mode_only_adds_compact_body_class(self) -> None:
        plain = render_overlay_page(KEY, media_base="http://cloud:8889", websocket_path="/ws")
        embed = render_overlay_page(KEY, media_base="http://cloud:8889", websocket_path="/ws", embed=True)
        self.assertIn("<body>", plain)
        self.assertNotIn('<body class="embed">', plain)
        self.assertEqual(embed, plain.replace("<body>", '<body class="embed">', 1))

    def test_page_rejects_invalid_media_url_and_fps(self) -> None:
        for base, fps in (("javascript:alert(1)", 15), ("http://localhost", 0)):
            with self.assertRaises(ValueError):
                render_overlay_page(KEY, media_base=base, fps=fps, websocket_path="/ws")

    @unittest.skipUnless(importlib.util.find_spec("starlette"), "Starlette unavailable")
    def test_cloud_app_exposes_live_page_for_stream(self) -> None:
        from starlette.testclient import TestClient
        from web.cloud_app import create_app

        with TestClient(create_app()) as client:
            response = client.get("/v1/streams/edge-01/cam-01/preview/2/live")
        self.assertEqual(response.status_code, 200)
        self.assertIn("/edge/edge-01/cam-01/preview/epoch-2/whep", response.text)
        self.assertIn("/v1/streams/edge-01/cam-01/preview/2/metadata", response.text)

    @unittest.skipUnless(importlib.util.find_spec("starlette"), "Starlette unavailable")
    def test_live_page_takes_video_from_the_streams_own_edge(self) -> None:
        from unittest.mock import patch
        from starlette.testclient import TestClient
        from web.cloud_app import create_app

        bases = '{"edge-01": "http://192.168.180.246:8889"}'
        with patch.dict("os.environ", {"CLOUD_EDGE_WEBRTC_BASES": bases,
                                       "CLOUD_MEDIAMTX_WEBRTC_BASE": "http://192.168.180.249:8889"}), \
                TestClient(create_app()) as client:
            own = client.get("/v1/streams/edge-01/cam-01/preview/2/live").text
            other = client.get("/v1/streams/edge-02/cam-01/preview/2/live").text
        self.assertIn("http://192.168.180.246:8889/edge/edge-01/cam-01/preview/epoch-2/whep", own)
        self.assertIn("http://192.168.180.249:8889/edge/edge-02/cam-01/preview/epoch-2/whep", other)  # fallback

    @unittest.skipUnless(importlib.util.find_spec("starlette"), "Starlette unavailable")
    def test_stream_status_endpoint_reports_hub_streams(self) -> None:
        from starlette.testclient import TestClient
        from cloud_core import ResultHub
        from web.cloud_app import create_app

        hub = ResultHub()
        hub.stream_status = lambda: [{"node_id": "edge-252", "stream_epoch": 3}]
        with TestClient(create_app(hub=hub)) as client:
            response = client.get("/v1/streams/status")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["cache-control"], "no-store")
        self.assertEqual(response.json(), {"streams": [{"node_id": "edge-252", "stream_epoch": 3}]})

    @unittest.skipUnless(importlib.util.find_spec("starlette"), "Starlette unavailable")
    def test_stable_live_url_resolves_most_recent_active_epoch(self) -> None:
        from starlette.testclient import TestClient
        from web.cloud_app import create_app

        class ActiveStreams:
            def active_streams(self):
                return {
                    ("edge-01", "cam-01", "preview", 2): 100.0,
                    ("edge-01", "cam-01", "preview", 9): 200.0,
                    ("edge-02", "cam-01", "preview", 11): 300.0,
                }

        app = create_app()
        app.state.frame_taps = ActiveStreams()
        with patch.dict("os.environ", {}, clear=True), TestClient(app) as client:
            response = client.get("/v1/streams/edge-01/cam-01/preview/live?fps=30")
        self.assertEqual(response.status_code, 200)
        self.assertIn("/edge/edge-01/cam-01/preview/epoch-9/whep", response.text)
        self.assertIn('"websocketPath":"/v1/streams/edge-01/cam-01/preview/9/metadata"', response.text)

    @unittest.skipUnless(importlib.util.find_spec("starlette"), "Starlette unavailable")
    def test_stable_live_url_says_reload_when_no_edge_stream_is_active(self) -> None:
        from starlette.testclient import TestClient
        from web.cloud_app import create_app

        class NoActiveStreams:
            def active_streams(self):
                return {}

        app = create_app()
        app.state.frame_taps = NoActiveStreams()
        with TestClient(app) as client:
            response = client.get("/v1/streams/edge-01/cam-01/preview/live")
        self.assertEqual(response.status_code, 404)
        self.assertIn("start Edge publishing", response.json()["error"])
