from __future__ import annotations

from types import SimpleNamespace
import unittest

from edge_agent.phase7 import _box_change, _tracked_boxes


def box(track_id, left=0.1, top=0.1, right=0.4, bottom=0.4):
    return SimpleNamespace(
        track_id=track_id, left=left, top=top, right=right, bottom=bottom,
    )


class Phase7MonitorBoxChangeTests(unittest.TestCase):
    def test_uses_available_track_ids_when_other_detections_are_untracked(self):
        result = SimpleNamespace(boxes=(box(7), box(None, 0.5, 0.5, 0.7, 0.7)))
        tracks = _tracked_boxes(result)

        self.assertEqual(tuple(tracks), (7,))
        self.assertEqual(_box_change(tracks, {7: (0.1, 0.1, 0.4, 0.4)}), 0.0)

    def test_no_tracked_boxes_breaks_comparison(self):
        self.assertIsNone(_tracked_boxes(SimpleNamespace(boxes=(box(None),))))
        self.assertIsNone(_box_change({7: (0.1, 0.1, 0.4, 0.4)}, None))


if __name__ == "__main__":
    unittest.main()
