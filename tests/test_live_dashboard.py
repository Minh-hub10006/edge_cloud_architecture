from __future__ import annotations

import importlib.util
from pathlib import Path
import shutil
import subprocess
import unittest
from unittest.mock import patch

from web.live_dashboard import render_dashboard


class LiveDashboardTests(unittest.TestCase):
    def test_renders_one_tile_config_per_node_with_logic_inlined(self) -> None:
        html = render_dashboard(["edge-246", "edge-252", "edge-247"])
        self.assertIn('"nodes":["edge-246","edge-252","edge-247"]', html)
        self.assertIn("LiveDashboard", html)
        self.assertIn("/v1/streams/status", html)
        self.assertNotIn("__LOGIC__", html)
        self.assertNotIn("__CONFIG__", html)

    def test_rejects_unsafe_or_duplicate_nodes_and_bad_timing(self) -> None:
        for nodes in ([], ["edge-246", "edge-246"], ["<script>"], ["a b"]):
            with self.assertRaises(ValueError):
                render_dashboard(nodes)
        for kwargs in ({"fps": 0}, {"buffer_ms": 50}, {"smooth_ms": 0}, {"poll_ms": 100}):
            with self.assertRaises(ValueError):
                render_dashboard(["edge-246"], **kwargs)

    @unittest.skipUnless(shutil.which("node"), "Node.js unavailable")
    def test_javascript_tile_logic(self) -> None:
        root = Path(__file__).resolve().parents[1]
        result = subprocess.run(["node", str(root / "tests/test_live_dashboard.cjs")],
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)

    @unittest.skipUnless(shutil.which("node"), "Node.js unavailable")
    def test_page_script_swaps_reloads_and_survives_cloud_loss(self) -> None:
        import re
        import tempfile
        root = Path(__file__).resolve().parents[1]
        scripts = re.findall(r"<script>(.*?)</script>", render_dashboard(["edge-246", "edge-252", "edge-247"]), re.S)
        self.assertEqual(len(scripts), 2)
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "dash_logic.js").write_text(scripts[0], encoding="utf-8")
            Path(tmp, "dash_ui.js").write_text(scripts[1], encoding="utf-8")
            result = subprocess.run(["node", str(root / "tests/test_live_dashboard_ui.cjs"), tmp],
                                    capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("DASHBOARD_SIM_PASS", result.stdout)

    @unittest.skipUnless(importlib.util.find_spec("starlette"), "Starlette unavailable")
    def test_cloud_app_serves_dashboard_from_env_nodes(self) -> None:
        from starlette.testclient import TestClient
        from web.cloud_app import create_app

        with patch.dict("os.environ", {"CLOUD_LIVE_NODES": "edge-246, edge-252"}), TestClient(create_app()) as client:
            response = client.get("/live")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["cache-control"], "no-store")
        self.assertIn('"nodes":["edge-246","edge-252"]', response.text)
        with patch.dict("os.environ", {"CLOUD_LIVE_NODES": "bad node"}), TestClient(create_app()) as client:
            response = client.get("/live")
        self.assertEqual(response.status_code, 500)
        self.assertIn("CLOUD_LIVE_NODES", response.json()["error"])


if __name__ == "__main__":
    unittest.main()
