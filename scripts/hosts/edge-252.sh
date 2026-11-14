# Edge252: QCS6490 ODK (Yupik), Ubuntu 20.04 (glibc 2.31), user root, 6 cores.
# Daemons are built on this host (scripts/setup_edge_node.sh): 246 binaries need glibc 2.39.
NODE_ID=edge-252
EDGE_IP=192.168.180.252  # v50: browsers fetch this Edge's video from its own MediaMTX
# OS default scheduler: sched_boost moves work to big cores (+37%) but this board has no heatsink and
# reaches 95 C within ~75 s of sustained load, so 252 is measured as delivered.
EDGE_UCLAMP_MIN=
EDGE_HOME=/root
EDGE_PYTHON_DEFAULT=$EDGE_HOME/miniconda3/envs/tracking/bin/python
QAIRT_ROOT_DEFAULT=/opt/qairt/2.43.0.260128
QAIRT_ABI=aarch64-ubuntu-gcc9.4
VIDEO=$EDGE_HOME/edge_multimodel/videos/clips_1min/traffic_1min.mp4
# 252 segfaults in on-device HTP graph prepare, so it needs DLCs with a QCS6490 cache for the
# exact daemon outputs boxes,scores (snpe-dlc-graph-prepare on Cloud249, SDK 2.43; weights unchanged).
# YOLOv8n model_id is the DLC file stem, so the re-prepared file keeps the original name.
YOLOV8N_DLC=$EDGE_HOME/edge_artifacts/npu243_prepared/yolov8n_320_int8_split_traffic100.dlc
# The prepared DLC is fine (v3 daemon --build-check PASS on DSP, Oct 28); the legacy init-cache daemon ran it on the
# CPU (~300% CPU, 25x slower, F1 .463). Graph-scoped strict v3 daemon instead, as YOLO11s:
YOLOV8N_SNPE_GRAPH=yolov8n_320_split
YOLO11S_DLC=$EDGE_HOME/edge_artifacts/npu243/yolo11s320_w8a8_qcs6490_dual.dlc
YOLOV8N_DLC_SHA=00a027898d3e83146b9c8df4d558363272151d2220a13a03fa26bc976bd57591
YOLO11S_DLC_SHA=4e1e5246d4e027bf66fd82a52e0a59e5cd23654fbbc2f3f8a4b3c5f7e6850316
SNPE_DAEMON_BIN_DEFAULT=$EDGE_HOME/edge_artifacts/bin/snpe-daemon
YOLO11S_DAEMON_BIN=$EDGE_HOME/edge_artifacts/bin/snpe-daemon-yolo11s-v3
# DRL capability profile (results/BENCHMARK_RESULTS.md, 2026-10-11): pipeline FPS of YOLOv8n/YOLO11s alone
# (inflight 2, 10k x 3) and x264 cap = 1000 / ffmpeg_write_ms p50 of the Cloud-model pipeline.
EDGE_CAPABILITY='{"yolov8n_fps":45.6,"yolo11s_fps":39.0,"encode_fps":20.9}'
EDGE_MEDIAMTX_BIN=$EDGE_HOME/edge_artifacts/mediamtx/mediamtx  # v50 (run_test.sh starts it once per run)
# Plate path (CONTROL_PLATE=1): like YOLOv8n here it needs a QCS6490-prepared DLC (snpe-dlc-graph-prepare on Cloud249)
# and the strict v3 daemon with the DLC's graph name (snpe-dlc-info: plate_320_split_fp32, outputs boxes,scores).
PLATE_DLC=$EDGE_HOME/plate_ocr_home/edge/plate_yolov8n_320_int8_split_qcs6490.dlc
PLATE_SNPE_GRAPH=plate_320_split_fp32
# 14/11: full-size frames (slow lane) converted on the CPU: 252's old qtivtransform faults the GPU with 4 streams
# (media/hw_capture.py, scripts/hw_stress.py). Exported: the camera agents read it.
export V4_HIRES_CONVERT=cpu
