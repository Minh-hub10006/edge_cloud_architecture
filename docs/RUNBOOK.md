# Runbook

Mọi lệnh chạy trên máy đó (SSH). Code đặt ở `~/edge_video_platform_v4` trên mỗi máy. Lần chạy dài nên đặt trong
`tmux new -s v4run` (SSH đứt không giết agent). Trên 247 mở Git Bash trước: `%USERPROFILE%\PortableGit\bin\bash.exe -l`.

## 1. Cập nhật code

```bash
# trên máy có repo: đóng gói (gói chứa sẵn thư mục edge_video_platform_v4/), chép (scp) sang từng máy, kiểm sha256 hai đầu
git archive --format=tar.gz --prefix=edge_video_platform_v4/ -o edge_v4_code_<ngày>.tgz HEAD
tar -xzf ~/edge_v4_code_<ngày>.tgz -C ~          # trên từng máy
```
- Sửa file trên Windows rồi chép sang Linux: giữ xuống dòng LF (CRLF làm bash báo `$'\r': command not found`).
- 247: đường dẫn code phải là 1 dòng trong `python312._pth` của Python nhúng (script chạy tự báo nếu thiếu).

## 2. Cloud 249

```bash
cd ~/edge_video_platform_v4
bash scripts/run_results_v4.sh db                 # lần đầu: tạo mật khẩu (~/.v4_db.env, mode 600), DB, schema
bash scripts/triton_config_v4.sh trt "dfinex_full rtdetr_full crowd_density rtdetr_tail_q"   # Triton + TensorRT FP32 (biên dịch 5-15 phút lần đầu)
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
curl -s -X POST localhost:18000/v2/repository/index | grep -o '"state":"READY"' | wc -l   # 9
```

Triton (`scripts/triton_config_v4.sh`): `show` | `trt [MODELS]` | `trt-off` | `ratelimit` | `add NAME ONNX` | `restore DIR`.
`add` nhận cả model làn nhanh lẫn làn sau; các model đang chạy TensorRT được giữ nguyên khi viết lại cấu hình.
Nửa sau cách B (`rtdetr_tail_q`, ONNX từ `scripts/tail_quant.py TAIL.onnx OUT.onnx --check HEAD.onnx ẢNH...`):
`add rtdetr_tail_q rtdetr_tail_q.onnx`, rồi khởi động lại worker làn nhanh.
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
# 252 (hosts/edge-252.sh đặt sẵn V4_HIRES_CONVERT=cpu: ảnh lớn đổi màu bằng CPU)
export V4_DECODER=qtic2vdec V4_HW_SIZE=640 V4_SLOW_TASKS=...   (như trên)
bash scripts/run_edge_v4.sh 252 "vn6:N vn9:A vn6:B vn9:D" 180
# 247 (Git Bash; giải mã CPU, không cần V4_DECODER)
export V4_SLOW_TASKS=...
bash scripts/run_edge_v4.sh 247 "vn9:N vn6:A vn9:B vn6:C vn9:D" 180
```
- Dạng camera: `video:CÁCH[:KHUNG_BẮT_ĐẦU]`, ví dụ `vn6:C:700`. Video: `~/test_videos/<video>.mp4` + `<video>_main.mp4`.
- Tham số 4 (tuỳ chọn) `HH:MM:SS`: giờ bắt đầu theo **đồng hồ của chính máy** (252 chạy giờ UTC, 246/247 giờ VN).
- Biến khác: `V4_SLOW_CONFIG='{"congestion_vehicles": 30, ...}'`, `V4_PROFILES=<hồ sơ camera json>`, `V4_HEAD=<file nửa đầu B>`,
  `V4_CUT_BITS` (cách B: 2 mặc định, 8 = uint8 cũ).
- Video thử: `<video>.mp4` (phân tích) phải trùng từng khung với `<video>_main.mp4` (luồng chính, có dải số khung):
  chạy một lần `bash scripts/fix_test_videos.sh <máy>` (tạo lại từ luồng chính nếu số khung khác, giữ bản cũ).
- Lần chạy đầu trên mỗi Edge thử bộ giải mã rồi lưu ở `~/.cache/v4_hw_probe.json`; xoá file này sau khi cập nhật hệ điều hành.
- Script tự dọn đồ sót (agent, daemon NPU, ffmpeg) trước và sau mỗi lần chạy; in `leftovers of a run ...` khi có.
- Cuối lần chạy script in bảng: khung xử lý / tổng, bỏ, lỗi, độ trễ p50/p95, CPU, nhiệt độ; log ở `~/v4_runs_edge/run_<máy>_<giờ>/`.
  In lại bảng: `python -B scripts/v4_run_summary.py <thư mục run>`.

## 4. File model và dữ liệu (không có trong repo)

| Máy | File | Ghi chú |
|---|---|---|
| 249 | `~/triton_repo_20261105/<model>/1/model.onnx`: `rtdetr_full`, `rtdetr_tail_q` (cách B, sha `2935debc…`), `rtdetr_tail_u8`, `rtdetr_tail`, `dfinex_full`, `crowd_density` (DM-Count, ONNX sha `ac90db17…`), `pose_rtmo_s` (RTMO-s, sha `d0703d40…`), `action_stgcnpp` (ST-GCN++ NTU60, sha `f966b4d4…`) | `add NAME ONNX` để thêm model làn sau |
| 249 | `~/plate_ocr_home` (EasyOCR + mạng `plate`), `~/triton_repo_20261105/plate_ocr/1/model.onnx` (cùng trọng số, sha `0e972566…`), `~/.v4_db.env` | mật khẩu DB chỉ ở file này |
| 246 | `~/edge_artifacts/npu243/yolov8n_320_int8_split_traffic100.dlc`; YOLO11s `yolo11s320_w8a8_qcs6490.dlc` + daemon `snpe-daemon-yolo11s-v3` (đường dẫn trong `scripts/hosts/edge-246.sh`); `~/edge_b_20261106/head_w8a8_pc.dlc` (nửa đầu B); `~/plate_ocr_home/edge/plate_yolov8n_320_int8_split.dlc`; QAIRT `~/sigm/qairt/2.43.0.260128` | |
| 252 | `~/edge_artifacts/npu243_prepared/yolov8n_320_int8_split_traffic100.dlc` (graph `yolov8n_320_split`); `~/edge_artifacts/npu243/yolo11s320_w8a8_qcs6490_dual.dlc`; `/root/edge_b_20261106/head_w8a8_pc.dlc` (cùng file 246, đã có cache QCS6490); `~/plate_ocr_home/edge/plate_yolov8n_320_int8_split_qcs6490.dlc` (graph `plate_320_split_fp32`); `/root/v4_pydeps` (torch, ultralytics, scipy, lap) | daemon v3 theo tên đồ thị |
| 247 | `~/edge_artifacts/onnx/yolov8n_320_split.onnx`, `yolo11s320_float.onnx`; `~/edge_b_20261106/head.onnx`; `~/plate_ocr_home/edge/plate_yolov8n_320_split.onnx` | ONNX Runtime CUDA |

Mọi Edge: `~/test_videos/vn6*.mp4`, `vn9*.mp4`; MediaMTX trong `~/edge_artifacts/mediamtx/`.

## 5. Kiểm nội dung Edge → DB và tài nguyên

```bash
# 249: xuất một Edge (N phút gần nhất) -> ~/run_export_<node>.tgz; trên laptop giải nén rồi kiểm
bash scripts/export_run.sh edge-246 15
python -B scripts/verify_db_export.py <thư mục> --videos <thư mục video phân tích> --map cam-1=vn9 cam-2=vn6 ...
python -B scripts/review_page.py <thư mục> --videos <...> --map ...      # trang xem lại theo UC (mở index.html)
# tài nguyên trong lúc chạy (Linux): máy, nhóm tiến trình, GPU + lô Triton (Cloud)
python3 -B scripts/resource_log.py --role edge|cloud --seconds 300
# bộ giải mã: ảnh đưa cho model = ảnh gốc? nhảy khung đúng? N luồng cùng lúc có lỗi GPU?
python -B scripts/hw_frame_dump.py ~/test_videos/vn9.mp4 --decoder <dec> --frames 0 300 --seek 1234
python -B scripts/hw_stress.py ~/test_videos/vn9.mp4 --decoder <dec> --streams 4 --seconds 60
```
Cổng `verify_db_export.py`: số khung khớp 100 %; box nằm trên vật thể của model tham chiếu (YOLO11m) ≥ 50 %, cùng lớp
≥ 80 % (YOLO11s hay lẫn người / xe máy, xe tải / ô tô: 78–81 %); ảnh biển nằm trong xe của nó trên đúng khung ≥ 90 %.

## 6. Kiểm tra kết quả trong DB (249)

```bash
q() { docker exec v4-tsdb psql -U postgres -d v4 -Atc "$1"; }
q "SELECT node_id, camera_id, count(*), round(avg(latency_ms)) FROM frames WHERE capture_time > now() - interval '10 minutes' GROUP BY 1,2 ORDER BY 1,2"
q "SELECT event_type, count(*) FROM events WHERE started_at > now() - interval '1 hour' GROUP BY 1"
q "SELECT camera_id, track_id, plate, valid, reads FROM plate_tracks WHERE last_seen > now() - interval '10 minutes' ORDER BY reads DESC LIMIT 10"
```
Giờ trong DB là UTC.

## 7. Lỗi đã gặp và cách xử lý

| Triệu chứng | Nguyên nhân | Xử lý |
|---|---|---|
| `vb2_streamon failed, -12`, `inst in error state` (dmesg), mọi camera không có khung | hết CMA (246 ~125 MB, giảm dần qua nhiều lần chạy) | khởi động lại Edge; `sync; echo 1 > /proc/sys/vm/drop_caches` trước khi chạy giúp có thêm CMA |
| `hardware decoder probe hung for 60 s` | nhiều agent thử bộ giải mã cùng lúc (camera khởi động muộn 90–110 s); GStreamer cũ của 252 bỏ qua `eos-after` | đã sửa: lưu kết quả thử, mỗi lúc một camera, tự dừng sau 30 khung; lặp lại → khởi động lại Edge |
| `[HwCapture] no frame for 10 s ... restart` + dmesg `kgsl GPU PAGE FAULT ... gst-launch` | `qtivtransform` cũ của 252 đổi màu ảnh lớn 1080x1920 cho nhiều luồng → GPU đọc ra ngoài bộ đệm → bộ giải mã lỗi -5 (không phải do nhiệt) | `V4_HIRES_CONVERT=cpu` (đã đặt trong `hosts/edge-252.sh`); kiểm bằng `scripts/hw_stress.py` |
| `control v1 TIMEOUT_INCOMPLETE` | agent treo quá `giây + 120` | xem `[HwCapture]` trong log camera đó và dmesg; gom: `tar` thư mục run + `dmesg -T` |
| Box lệch vị trí trên video | video phân tích ngắn hơn / khác luồng chính | `scripts/fix_test_videos.sh <máy>` |
| Mọi box trên 252 sai, ảnh lệch 64 KiB | thư viện `qtic2vdec` in log vào stdout lẫn với ảnh | đã sửa: ảnh đi đường ống riêng (`media/hw_capture.py`) |
| `/dev/shm` đầy vùng `v4_*`, worker làn sau ~12 GB | vùng nhớ dùng chung không được giải phóng khi đổi cỡ | đã sửa; dọn vùng cũ: dừng worker, `curl -X POST localhost:18000/v2/systemsharedmemory/unregister`, `rm /dev/shm/v4_*` |
| `no control-v1 feedback` lúc khởi động | worker Cloud chưa sẵn sàng | chờ ~30 s sau khi bật worker rồi chạy Edge |
| Ảnh biển số là rác | đưa NCHW vào DLC biển số (DLC cần NHWC) | đã sửa trong `legacy/adapters/yolo/yolo_snpe.py`; kiểm bằng `scripts/plate_dlc_check.py` |
| 252 cách N rất nóng / CPU cao | YOLOv8n chạy daemon cũ (CPU) khi thiếu `YOLOV8N_SNPE_GRAPH` | script chạy đã truyền biến này |
| `Unable to find shared memory region` | Triton khởi động lại | worker tự đăng ký lại (bản hiện tại) |
| Cách D chỉ ~50 % khung khi 3 Edge cùng chạy | D-FINE-X 1 bản, ONNX Runtime | 2 bản + 6 suất + TensorRT FP32 (đã áp) |
| Deadline / mọi khung bị coi là quá hạn (247) | đồng hồ lệch | bật `w32time` trên 247 |
| Đọc chữ biển đúng 2–3/19 | model EasyOCR | việc của nhóm CV (bộ thử 40 ảnh có sẵn) |

## 8. Giới hạn hiện tại

- 252 chạy đủ 4 camera × 8 việc nhưng nóng tới ~96 °C (không tản nhiệt; cách B ~85 %, D ~89 % khung); cần tản nhiệt
  cho chạy dài; 5 camera chưa thử lại sau các bản sửa.
- `rtdetr_tail_q` (cách B) chạy TensorRT FP32 từ 15/11 (trước đó ONNX Runtime, chậm hơn cả model đầy đủ).
- Cách B nén 2 bit (~90 KB) vẫn lớn hơn JPEG của C (~76 KB); xuống thấp hơn cần huấn luyện lớp nén.
- ST-GCN++ dùng trọng số NTU60, chưa huấn luyện lại; ngưỡng đánh nhau 0,3 tạm.
- DRL chưa cắm vào v4.
