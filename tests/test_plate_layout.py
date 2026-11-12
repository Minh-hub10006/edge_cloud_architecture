"""11/11: the plate DLC takes NHWC (fed NCHW since 8/11 it gave only garbage crops); its ONNX stays NCHW."""
import unittest


class PlateLayoutTests(unittest.TestCase):
    def test_plate_dlc_is_fed_nhwc_and_the_vehicle_yolo_keeps_its_layout(self):
        from legacy.adapters.yolo.yolo_snpe import YOLOSnpeDetector
        plate = YOLOSnpeDetector(dlc_path="/missing/plate_yolov8n_320_int8_split.dlc", variant="plate", imgsz=320)
        vehicle = YOLOSnpeDetector(dlc_path="/missing/yolov8n_320_int8_split_traffic100.dlc", variant="yolov8",
                                   imgsz=320)
        self.assertEqual((plate.input_layout, vehicle.input_layout), ("nhwc", "nchw"))
        tensor, _ = plate._preprocess(__import__("numpy").zeros((90, 160, 3), "uint8"))
        self.assertEqual(tensor.shape, (1, 320, 320, 3))


if __name__ == "__main__":
    unittest.main()
