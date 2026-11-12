# Edge247: Windows 11 workstation (hostname ws570-249), RTX 5070 Ti, no admin rights.
# Runs under PortableGit bash with an embeddable Python whose python312._pth must list this
# release directory (PYTHONPATH is ignored). YOLO runs the float ONNX the NPU DLCs were
# converted from, on ONNX Runtime CUDA (FP32), so the graph and decode match 246/252.
NODE_ID=edge-247
EDGE_IP=192.168.180.247  # v50: browsers fetch this Edge's video from its own MediaMTX
EDGE_UCLAMP_MIN=
EDGE_HOME=/c/Users/user1
EDGE_PYTHON_DEFAULT=$EDGE_HOME/edge_py/python.exe
VIDEO=$EDGE_HOME/edge_artifacts/traffic_1min.mp4
YOLOV8N_DLC=$EDGE_HOME/edge_artifacts/onnx/yolov8n_320_split.onnx
YOLO11S_DLC=$EDGE_HOME/edge_artifacts/onnx/yolo11s320_float.onnx
YOLOV8N_DLC_SHA=a4509a87da33bcbb3561414a9fba602f4092a6a722a685ad5090ef6276885cab
YOLO11S_DLC_SHA=62ff9e6d7803d30b741d3933845bc23e4404a2e4d37273db540dd6248c1300db
YOLOV8N_KEY=yolov8n_320_split
YOLO11S_KEY=yolo11s320_float_ort
YOLO11S_LAYOUT=nchw
# DRL capability profile (results/BENCHMARK_RESULTS.md, 2026-10-11): pipeline FPS of YOLOv8n/YOLO11s alone
# (inflight 2, 10k x 3) and x264 cap = 1000 / ffmpeg_write_ms p50 of the Cloud-model pipeline.
EDGE_CAPABILITY='{"yolov8n_fps":201.4,"yolo11s_fps":165.2,"encode_fps":268.8}'
EDGE_MEDIAMTX_BIN=$EDGE_HOME/edge_artifacts/mediamtx/mediamtx.exe  # v50 (run_test.sh starts it once per run)
# Plate path (CONTROL_PLATE=1): the same plate_yolov8n_320 as the NPU Edges, split ONNX on ORT CUDA (scripts/export_plate_onnx.py).
PLATE_DLC=$EDGE_HOME/plate_ocr_home/edge/plate_yolov8n_320_split.onnx
