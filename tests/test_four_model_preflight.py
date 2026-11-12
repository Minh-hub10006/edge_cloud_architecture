import importlib.util
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location(
    "preflight", Path(__file__).resolve().parents[1] / "scripts/four_model_preflight.py")
preflight = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preflight)


class PreflightTests(unittest.TestCase):
    def test_missing_and_unpinned_do_not_pass(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.dlc"
            self.assertEqual(preflight.inventory(path)["gate"], "UNAVAILABLE")
            path.write_bytes(b"test model")
            self.assertEqual(preflight.inventory(path)["gate"], "UNPINNED")
            self.assertEqual(preflight.inventory(path, "bad")["gate"], "MISMATCH")
            self.assertEqual(preflight.inventory(path, preflight.digest(path))["gate"], "MATCH")

    def test_source_inventory_excludes_env_and_results(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "inference").mkdir()
            (root / "inference/catalog.py").write_text("# example\n")
            (root / "inference/.env").write_text("secret=not collected")
            (root / "inference/results").mkdir()
            (root / "inference/results/debug.py").write_text("# disposable\n")
            report = preflight.source_inventory(root)
            self.assertEqual(set(report["files"]), {"inference/catalog.py"})
            self.assertEqual(report["gate"], "RECORDED_NOT_ACCEPTED")
            self.assertEqual(preflight.source_inventory(root / "missing")["gate"], "UNKNOWN")
