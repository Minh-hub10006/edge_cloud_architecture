from __future__ import annotations

import unittest

from contracts import ModelMode, Placement
from inference import ModelAdapterCatalog, phase7_cloud_catalog, phase7_edge_catalog


class ModelCatalogTests(unittest.TestCase):
    def test_phase7_registration_is_lazy_and_placement_aware(self) -> None:
        self.assertEqual(phase7_edge_catalog().modes(Placement.EDGE), (ModelMode.YOLO_EDGE,))
        self.assertEqual(phase7_cloud_catalog().modes(Placement.CLOUD), (ModelMode.DFINE_CLOUD,))

    def test_explicit_edge_receiver_loads_no_cloud_models(self):
        self.assertEqual(phase7_cloud_catalog(enabled=("none",)).modes(), ())
        with self.assertRaises(ValueError):
            phase7_cloud_catalog(enabled=("none", "dfine"))

    def test_new_adapter_can_be_registered_without_changing_transport(self) -> None:
        class FakeModel:
            model_id = "survey-candidate"

            def infer(self, bgr):
                return [], 1.0

        catalog = ModelAdapterCatalog()
        catalog.register(ModelMode.RTDETR_CLOUD, FakeModel)
        self.assertEqual(catalog.load(ModelMode.RTDETR_CLOUD).model_id, "survey-candidate")
        with self.assertRaisesRegex(ValueError, "already registered"):
            catalog.register(ModelMode.RTDETR_CLOUD, FakeModel)
        with self.assertRaisesRegex(ValueError, "no adapter"):
            catalog.load(ModelMode.DFINE_CLOUD)
