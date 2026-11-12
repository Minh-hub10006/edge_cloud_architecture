# Edge246: QCS6490 RB3 Gen2, Ubuntu 24.04 (glibc 2.39), user ubuntu, 8 cores.
NODE_ID=edge-246
EDGE_IP=192.168.180.246  # v50: browsers fetch this Edge's video from its own MediaMTX
# Scheduler placement hint for every Edge run (uclampset -m): default EAS keeps preprocess on little
# cores (YOLOv8n model capacity 61.9 -> 84.9 FPS with 512, max 49 C). Empty = OS default.
EDGE_UCLAMP_MIN=512
EDGE_HOME=/home/ubuntu
EDGE_PYTHON_DEFAULT=$EDGE_HOME/miniconda3/envs/tracking/bin/python
QAIRT_ROOT_DEFAULT=$EDGE_HOME/sigm/qairt/2.43.0.260128
QAIRT_ABI=aarch64-oe-linux-gcc11.2
VIDEO=$EDGE_HOME/edge_multimodel/videos/clips_1min/traffic_1min.mp4
YOLOV8N_DLC=$EDGE_HOME/edge_artifacts/npu243/yolov8n_320_int8_split_traffic100.dlc
YOLO11S_DLC=$EDGE_HOME/edge_video_platform_benchmark_overlap_20261004/yolo11s320_split_diag.NHBFPo/npu_run/edge_packet/yolo11s320_w8a8_qcs6490.dlc
YOLOV8N_DLC_SHA=0da7101a4f75716bd4260e12499805b3e90f5943f6ddb71b345b005cca281d32
YOLO11S_DLC_SHA=1bb6a516a06e23bb9a6022c0b537574745aee603adc2539ec8a885d78231d8b3
SNPE_DAEMON_BIN_DEFAULT=$EDGE_HOME/edge_artifacts/bin/snpe-daemon
YOLO11S_DAEMON_BIN=$EDGE_HOME/edge_video_platform_benchmark_overlap_20261004/yolo11s_integration_v3.9dxyacpe/build/snpe-daemon-yolo11s-v3
# DRL capability profile (results/BENCHMARK_RESULTS.md, 2026-10-11): pipeline FPS of YOLOv8n/YOLO11s alone
# (inflight 2, 10k x 3) and x264 cap = 1000 / ffmpeg_write_ms p50 of the Cloud-model pipeline.
EDGE_CAPABILITY='{"yolov8n_fps":101.6,"yolo11s_fps":86.3,"encode_fps":66.6}'
EDGE_MEDIAMTX_BIN=$EDGE_HOME/edge_artifacts/mediamtx/mediamtx  # v50 (run_test.sh starts it once per run)
# Plate path (CONTROL_PLATE=1): our own copy of the platform's plate_yolov8n_320 split DLC, so our SNPE daemon never
# shares the other project's daemon (keyed by DLC path) that already runs ~/edge_artifacts/npu243/plate_yolov8n_320_int8_split.dlc.
PLATE_DLC=$EDGE_HOME/plate_ocr_home/edge/plate_yolov8n_320_int8_split.dlc
