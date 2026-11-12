# Runbook

Mọi lệnh chạy trên máy đó (SSH). Code đặt ở `~/edge_video_platform_v4` trên mỗi máy. Lần chạy dài nên đặt trong
`tmux new -s v4run` (SSH đứt không giết agent). Trên 247 mở Git Bash trước: `%USERPROFILE%\PortableGit\bin\bash.exe -l`.

## 1. Cập nhật code

```bash
# trên máy có repo: đóng gói rồi chép (scp) gói .tgz sang từng máy, kiểm sha256 hai đầu
tar -xzf ~/edge_v4_code_<ngày>.tgz -C ~/edge_video_platform_v4 --strip-components=1
```
- Sửa file trên Windows rồi chép sang Linux: giữ xuống dòng LF (CRLF làm bash báo `$'\r': command not found`).
- 247: đường dẫn code phải là 1 dòng trong `python312._pth` của Python nhúng (script chạy tự báo nếu thiếu).

## 2. Cloud 249

```bash
cd ~/edge_video_platform_v4
bash scripts/run_results_v4.sh db                 # lần đầu: tạo mật khẩu (~/.v4_db.env, mode 600), DB, schema
bash scripts/triton_config_v4.sh trt              # Triton: cấu hình làn nhanh + TensorRT FP32 (biên dịch 5-15 phút lần đầu)
RTDETR_SPLIT_HEAD_ID=rtl2-w8a8pc,rtl2-onnx-fp32 bash scripts/run_cloud_v4.sh start 4   # 4 worker làn nhanh
bash scripts/run_slow_v4.sh start                 # worker làn sau (UC3, UC4)
bash scripts/run_plate_v4.sh start                # OCR biển số: 16 luồng, Triton plate_ocr gom lô (PLATE_OCR_BACKEND=easyocr: cách cũ)
bash scripts/run_results_v4.sh start              # ghi DB
```
Dừng: `run_cloud_v4.sh stop`, `run_slow_v4.sh stop`, `run_plate_v4.sh stop`, `run_results_v4.sh stop` (Kafka, Triton, DB để chạy).

Kiểm tra nhanh:
```bash
echo "fast $(pgrep -fc 'v4-cloud|cloud_core.worker') slow $(pgrep -fc v4-slow-worker) ocr $(pgrep -fc v4-plate-ocr) writer $(pgrep -fc v4-results-writer)"   # 4 1 1 1
docker ps --format '{{.Names}} {{.Status}}' | grep v4-                     # kafka, triton, tsdb: Up
curl -s -X POST localhost:18000/v2/repository/index | grep -o '"state":"READY"' | wc -l   # 8 (7 + plate_ocr)
```

Triton (`scripts/triton_config_v4.sh`): `show` | `trt [MODELS]` | `trt-off` | `ratelimit` | `add NAME ONNX` | `restore DIR`.
OCR biển số (`plate_ocr`, ONNX từ `scripts/plate_ocr_parity.py export`): `add plate_ocr plate_ocr.onnx`, rồi kiểm chữ giống EasyOCR:
`scripts/plate_ocr_parity.py parity --model-home ~/plate_ocr_home --crops <thư mục ảnh biển> --backend triton` (cổng ≥ 99,4 %:
chính EasyOCR chạy CPU so với GPU cũng chỉ khớp 521/524 vì sai số phần cứng).
Mỗi lệnh sao lưu cấu hình cũ (`~/triton_config_backup_*`). Bản biên dịch TensorRT ở `~/triton_trt_cache` (giữ qua lần tạo lại container).
Kiểm độ khớp sau mỗi thay đổi model / độ chính xác (cổng: "Triton GPU vs ORT CPU" ≥ 0,99):
```bash
PYTHONPATH=.:$HOME/v4_pydeps ~/miniconda3/envs/qnn/bin/python -B scripts/triton_check.py --frames ~/triton_check_20261105/frames --head ~/triton_check_20261105/head.onnx --conc 1 4 --seconds 5
```

## 3. Edge

```bash
# 246
export V4_DECODER=v4l2h264dec V4_HW_SIZE=640 V4_SLOW_TASKS=plates,count,congestion,crowd,gathering,fight,crash,loiter
bash scripts/run_edge_v4.sh 246 "vn9:N vn6:A vn9:B vn6:C vn9:D" 180
# 252
export V4_DECODER=qtic2vdec V4_HW_SIZE=640 V4_SLOW_TASKS=...   (như trên)
bash scripts/run_edge_v4.sh 252 "vn9:N vn6:A vn9:B vn6:C vn9:D" 180
# 247 (Git Bash; giải mã CPU, không cần V4_DECODER)
export V4_SLOW_TASKS=...
bash scripts/run_edge_v4.sh 247 "vn9:N vn6:A vn9:B vn6:C vn9:D" 180
```
- Dạng camera: `video:CÁCH[:KHUNG_BẮT_ĐẦU]`, ví dụ `vn6:C:700`. Video: `~/test_videos/<video>.mp4` + `<video>_main.mp4`.
- Tham số 4 (tuỳ chọn) `HH:MM:SS`: giờ bắt đầu theo **đồng hồ của chính máy** (252 chạy giờ UTC, 246/247 giờ VN).
- Biến khác: `V4_SLOW_CONFIG='{"congestion_vehicles": 30, ...}'`, `V4_PROFILES=<hồ sơ camera json>`, `V4_HEAD=<file nửa đầu B>`.
- Cuối lần chạy script in bảng: khung xử lý / tổng, bỏ, lỗi, độ trễ p50/p95, CPU, nhiệt độ; log ở `~/v4_runs_edge/run_<máy>_<giờ>/`.
  In lại bảng: `python -B scripts/v4_run_summary.py <thư mục run>`.

## 4. File model và dữ liệu (không có trong repo)

| Máy | File | Ghi chú |
|---|---|---|
| 249 | `~/triton_repo_20261105/<model>/1/model.onnx`: `rtdetr_full`, `rtdetr_tail_u8`, `rtdetr_tail`, `dfinex_full`, `crowd_density` (DM-Count, ONNX sha `ac90db17…`), `pose_rtmo_s` (RTMO-s, sha `d0703d40…`), `action_stgcnpp` (ST-GCN++ NTU60, sha `f966b4d4…`) | `add NAME ONNX` để thêm model làn sau |
| 249 | `~/plate_ocr_home` (EasyOCR + mạng `plate`), `~/triton_repo_20261105/plate_ocr/1/model.onnx` (cùng trọng số, sha `0e972566…`), `~/.v4_db.env` | mật khẩu DB chỉ ở file này |
| 246 | `~/edge_artifacts/npu243/yolov8n_320_int8_split_traffic100.dlc`; YOLO11s `yolo11s320_w8a8_qcs6490.dlc` + daemon `snpe-daemon-yolo11s-v3` (đường dẫn trong `scripts/hosts/edge-246.sh`); `~/edge_b_20261106/head_w8a8_pc.dlc` (nửa đầu B); `~/plate_ocr_home/edge/plate_yolov8n_320_int8_split.dlc`; QAIRT `~/sigm/qairt/2.43.0.260128` | |
| 252 | `~/edge_artifacts/npu243_prepared/yolov8n_320_int8_split_traffic100.dlc` (graph `yolov8n_320_split`); `~/edge_artifacts/npu243/yolo11s320_w8a8_qcs6490_dual.dlc`; `/root/edge_b_20261106/head_w8a8_pc.dlc` (cùng file 246, đã có cache QCS6490); `~/plate_ocr_home/edge/plate_yolov8n_320_int8_split_qcs6490.dlc` (graph `plate_320_split_fp32`); `/root/v4_pydeps` (torch, ultralytics, scipy, lap) | daemon v3 theo tên đồ thị |
| 247 | `~/edge_artifacts/onnx/yolov8n_320_split.onnx`, `yolo11s320_float.onnx`; `~/edge_b_20261106/head.onnx`; `~/plate_ocr_home/edge/plate_yolov8n_320_split.onnx` | ONNX Runtime CUDA |

Mọi Edge: `~/test_videos/vn6*.mp4`, `vn9*.mp4`; MediaMTX trong `~/edge_artifacts/mediamtx/`.

## 5. Kiểm tra kết quả trong DB (249)

```bash
q() { docker exec v4-tsdb psql -U postgres -d v4 -Atc "$1"; }
q "SELECT node_id, camera_id, count(*), round(avg(latency_ms)) FROM frames WHERE capture_time > now() - interval '10 minutes' GROUP BY 1,2 ORDER BY 1,2"
q "SELECT event_type, count(*) FROM events WHERE started_at > now() - interval '1 hour' GROUP BY 1"
q "SELECT camera_id, track_id, plate, valid, reads FROM plate_tracks WHERE last_seen > now() - interval '10 minutes' ORDER BY reads DESC LIMIT 10"
```
Giờ trong DB là UTC.

## 6. Lỗi đã gặp và cách xử lý

| Triệu chứng | Nguyên nhân | Xử lý |
|---|---|---|
| `vb2_streamon failed, -12`, `inst in error state` (dmesg), mọi camera không có khung | hết CMA (246 ~125 MB, giảm dần qua nhiều lần chạy) | khởi động lại Edge; `sync; echo 1 > /proc/sys/vm/drop_caches` trước khi chạy giúp có thêm CMA |
| `hardware decoder probe hung for 60 s` | nhiều phiên giải mã mở cùng lúc | đã tự thử lại 1 lần; lặp lại → khởi động lại Edge |
| `[HwCapture] no frame for 10 s ... restarting` | bộ giải mã đứng (252 lỗi -5 khi nóng / quá tải) | tự khởi động lại; nhiều lần liên tiếp → giảm số camera, làm mát 252 |
| `control v1 TIMEOUT_INCOMPLETE` | agent không xong trong `giây + 120` | xem `[HwCapture]` trong log camera đó |
| `no control-v1 feedback` lúc khởi động | worker Cloud chưa sẵn sàng | chờ ~30 s sau khi bật worker rồi chạy Edge |
| Ảnh biển số là rác | đưa NCHW vào DLC biển số (DLC cần NHWC) | đã sửa trong `legacy/adapters/yolo/yolo_snpe.py`; kiểm bằng `scripts/plate_dlc_check.py` |
| 252 cách N rất nóng / CPU cao | YOLOv8n chạy daemon cũ (CPU) khi thiếu `YOLOV8N_SNPE_GRAPH` | script chạy đã truyền biến này |
| `Unable to find shared memory region` | Triton khởi động lại | worker tự đăng ký lại (bản hiện tại) |
| Cách D chỉ ~50 % khung khi 3 Edge cùng chạy | D-FINE-X 1 bản, ONNX Runtime | 2 bản + 6 suất + TensorRT FP32 (đã áp) |
| Deadline / mọi khung bị coi là quá hạn (247) | đồng hồ lệch | bật `w32time` trên 247 |
| Đọc chữ biển đúng 2–3/19 | model EasyOCR | việc của nhóm CV (bộ thử 40 ảnh có sẵn) |

## 7. Giới hạn hiện tại

- 252 không gánh được 4–5 camera đủ 8 việc (85–97 °C, cách B 35–62 %); để DRL học chọn cách.
- `rtdetr_tail_u8` (cách B) chưa TensorRT (chưa có công cụ kiểm độ khớp cho nó).
- ST-GCN++ dùng trọng số NTU60, chưa huấn luyện lại; ngưỡng đánh nhau 0,3 tạm.
- DRL chưa cắm vào v4.
