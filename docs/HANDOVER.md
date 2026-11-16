# Bàn giao kỹ thuật (16/11/2026)

Tài liệu này dành cho người nhận lại hệ để tích hợp tiếp (web sản phẩm, DRL, model mới). Đọc theo thứ tự:
[ARCHITECTURE](ARCHITECTURE.md) → tài liệu này → [RUNBOOK](RUNBOOK.md) (lệnh chạy, lỗi đã gặp) →
[EDGE_NPU](EDGE_NPU.md) (phần cứng Edge, chuyển model sang NPU, model chia đôi) → [FLOWS](FLOWS.md).

## 1. Repo có gì và không có gì

- **Có**: toàn bộ code Edge, Cloud, script chạy / đo / kiểm, schema DB, cấu hình từng Edge, mã nguồn daemon NPU, ~640 test.
- **Không có** (bị `.gitignore` chặn, xem §3): file model (`.onnx`, `.dlc`, `.pt`), bản TensorRT đã biên dịch, video thử,
  SDK Qualcomm, kết quả đo, mật khẩu DB.
- Chưa có trong hệ: web sản phẩm (đã gỡ trang kiểm tra cũ; điểm nối ở §4), DRL (khung ở `drl/`, agent đang chạy cách cố định).

## 2. Dựng lại từ đầu

### 2.1 Cloud (máy có GPU NVIDIA + Docker; hiện là 249, RTX 4090)

Ba container, tên và cổng cố định trong script:

| Container | Ảnh | Cổng | Tạo bằng |
|---|---|---|---|
| `v4-kafka` | `apache/kafka:3.7.0` (KRaft, 1 nút) | 29092 | lệnh dưới (chưa có script) |
| `v4-triton` | `nvcr.io/nvidia/tritonserver:25.08-py3` | 18000 HTTP / 18001 gRPC / 18002 số đo | `scripts/triton_config_v4.sh` (tự tạo lại container) |
| `v4-tsdb` | `timescale/timescaledb:2.17.2-pg16` | 25432 | `scripts/run_results_v4.sh db` (tạo mật khẩu ở `~/.v4_db.env`, DB, schema) |

```bash
# Kafka: đổi 192.168.180.249 thành IP LAN của máy Cloud (Edge kết nối theo địa chỉ này)
docker run -d --name v4-kafka --restart unless-stopped -p 29092:29092 -e KAFKA_NODE_ID=1 \
  -e KAFKA_PROCESS_ROLES=broker,controller -e KAFKA_LISTENERS=PLAINTEXT://:29092,CONTROLLER://:29093 \
  -e KAFKA_ADVERTISED_LISTENERS=PLAINTEXT://192.168.180.249:29092 -e KAFKA_CONTROLLER_LISTENER_NAMES=CONTROLLER \
  -e KAFKA_LISTENER_SECURITY_PROTOCOL_MAP=CONTROLLER:PLAINTEXT,PLAINTEXT:PLAINTEXT \
  -e KAFKA_CONTROLLER_QUORUM_VOTERS=1@localhost:29093 -e KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR=1 \
  -e KAFKA_TRANSACTION_STATE_LOG_REPLICATION_FACTOR=1 -e KAFKA_TRANSACTION_STATE_LOG_MIN_ISR=1 \
  -e KAFKA_MESSAGE_MAX_BYTES=2097152 -e KAFKA_REPLICA_FETCH_MAX_BYTES=2097152 -e KAFKA_LOG_RETENTION_MINUTES=10 \
  apache/kafka:3.7.0
```
- Kafka **không tự tạo topic**: các script `run_*_v4.sh start` tạo topic của mình. Topic kết quả riêng từng Edge
  (`cloud.detection-result.v1.<node_id>`) được tạo cho `edge-246 edge-252 edge-247`; **thêm Edge mới**:
  `V4_NODES="edge-246 ... edge-x" bash scripts/run_cloud_v4.sh start`.
- Python Cloud: env conda `qnn` (Python 3.10: onnxruntime-gpu, tritonclient, aiokafka, zstandard, opencv, ultralytics cho
  ByteTrack, lap) + `~/v4_pydeps` (setuptools < 81); OCR dùng `~/miniconda3/bin/python` + `~/plate_ocr_home`. Nhóm phụ thuộc
  theo vai trò ở `pyproject.toml`. `bash scripts/run_plate_v4.sh check` báo thư viện OCR còn thiếu.
- Kho model Triton: `~/triton_repo_20261105/<model>/1/model.onnx` (danh sách ở §3); cấu hình do `triton_config_v4.sh` viết.
- Thứ tự bật và lệnh kiểm: [RUNBOOK §2](RUNBOOK.md#2-cloud-249).

### 2.2 Edge Qualcomm (246, 252)

1. QAIRT / SNPE 2.43 trên máy (đường dẫn trong `scripts/hosts/edge-<máy>.sh`); GStreamer với phần tử của Qualcomm.
2. Python: env conda `tracking` + `~/v4_pydeps` (aiokafka, zstandard, PyAV, torch + ultralytics cho ByteTrack, lap).
3. `bash scripts/setup_edge_node.sh check` (xem thiếu gì) rồi `build` (biên dịch 2 daemon NPU theo ABI của máy).
4. Chép file model của máy đó (§3), MediaMTX vào `~/edge_artifacts/mediamtx/`, video thử vào `~/test_videos/`.
5. **Thêm một Edge mới**: tạo `scripts/hosts/edge-<tên>.sh` theo mẫu 246 (bo mới, chuẩn bị đồ thị được) hoặc 252 (cần DLC đã
   chuẩn bị sẵn), thêm nhánh `case` trong `scripts/run_edge_v4.sh`, tạo topic kết quả cho node đó (trên).
Chi tiết phần cứng, daemon và cách chuyển model: [EDGE_NPU](EDGE_NPU.md).

### 2.3 Edge GPU (247, Windows)

Git Bash (PortableGit), Python nhúng 3.12 có onnxruntime-gpu, opencv, aiokafka, zstandard, ultralytics; đường dẫn code là
một dòng trong `python312._pth`. Model là ONNX (§3). Giải mã bằng OpenCV trên CPU.

## 3. File không có trong repo: ở đâu, kiểm bằng gì, tạo lại thế nào

Code **từ chối chạy** file sai mã SHA-256 ở những dòng có ghi "ghim trong code". Mọi file dưới đây hiện có trên máy chạy;
các file có dấu ● còn một bản sao trong thư mục `v50_deploy/` trên laptop của nhóm phát triển.

| File | Dùng cho | Nơi đặt | SHA-256 (đầu) | Nguồn / cách tạo lại |
|---|---|---|---|---|
| `rtdetrv2_s_640.onnx` ● | C; gốc để cắt B | 249 `~/cloud_artifacts/rtdetrv2_s_dyn_20261105/`, Triton `rtdetr_full` | `6b5b2ca4fb32…` (= model id `rtdetrv2_s_candidate_6b5b2ca4fb32`) | RT-DETRv2-S (r18vd, COCO) xuất lô động: `scripts/candidate_artifact.py` |
| `dfine_x_obj2coco.onnx` ● | D | 249 `~/cloud_artifacts/dfine_x_candidate_20261021/`, Triton `dfinex_full` | `9461641566be…` | D-FINE-X (Objects365 → COCO), cùng script |
| `head.onnx` ● | nửa đầu B trên 247; mốc so NPU | 247 `~/edge_b_20261106/` | `5e72f7c91386…` (ghim trong `inference/rtdetr_head_snpe.py`) | `scripts/cut_points.py rtdetrv2_s_640.onnx ẢNH --cut rtdetr_layer2 --out DIR` |
| `head_w8a8_pc.dlc` ● | nửa đầu B trên NPU | 246, 252 `~/edge_b_20261106/` | `2e0c4b09df23…` (ghim, như trên) | `scripts/cut_head_raw.py` + `scripts/convert_cut_head_snpe.sh … quantize_pc` trên 249 |
| `rtdetr_tail_q.onnx` ● | nửa sau B | Triton `rtdetr_tail_q` | `2935debc8987…` | `scripts/tail_quant.py tail.onnx OUT.onnx --check head.onnx ẢNH…` |
| `plate_ocr.onnx` ● | UC1 đọc chữ | Triton `plate_ocr` | `0e9725664532…` | `scripts/plate_ocr_parity.py export` từ trọng số EasyOCR trong `~/plate_ocr_home` |
| `crowd_density` (DM-Count) | UC3 | Triton | `ac90db17…` | xuất ONNX từ DM-Count; kiểm bằng `scripts/crowd_triton_check.py` |
| `pose_rtmo_s`, `action_stgcnpp` | UC4 | Triton | `d0703d40…`, `f966b4d4…` | RTMO-s; ST-GCN++ trọng số NTU60 (chưa huấn luyện lại) |
| `yolov8n_320_int8_split_traffic100.dlc` | N | 246 `~/edge_artifacts/npu243/`; 252 `npu243_prepared/` (bản đã chuẩn bị đồ thị) | 246 `0da7101a…`, 252 `00a02789…` (`hosts/edge-*.sh`) | YOLOv8n 320, đầu ra tách `boxes`/`scores`, INT8 |
| `yolo11s320_w8a8_qcs6490.dlc` (246), `…_dual.dlc` (252) | A | đường dẫn trong `hosts/edge-*.sh` | `1bb6a516…`, `4e1e5246…` (ghim trong `inference/yolo11s_snpe.py`) | `scripts/yolo11s_snpe_candidate.py` |
| `yolov8n_320_split.onnx`, `yolo11s320_float.onnx` | N, A trên 247 | 247 `~/edge_artifacts/onnx/` | YOLO11s `62ff9e6d…` (ghim) | ONNX gốc của hai DLC trên |
| model tìm biển `plate_yolov8n_320_*` (DLC / ONNX) | UC1 trên Edge | `~/plate_ocr_home/edge/` mỗi Edge | theo `hosts/edge-*.sh` | của nền tảng biển số; ONNX: `scripts/export_plate_onnx.py` |
| `~/plate_ocr_home` | EasyOCR + mạng `plate` | 249 | — | bộ model OCR của nhóm CV |
| bản TensorRT đã biên dịch | — | 249 `~/triton_trt_cache/` | — | tự dựng lại khi `triton_config_v4.sh trt …` (5–15 phút) |
| `vn6_main.mp4`, `vn9_main.mp4` ● | luồng chính thử nghiệm (có dải số khung) | mọi Edge `~/test_videos/` | `c7c002676687…`, `0d3710fecc11…` | `scripts/make_main_video.py` từ video gốc |
| `vn6.mp4`, `vn9.mp4` ● | luồng phân tích (trùng từng khung với luồng chính) | mọi Edge `~/test_videos/` | `fcd22b749af7…`, `89e25c2f1aef…` | `scripts/fix_test_videos.sh <máy>` tạo từ `<video>_main.mp4` |
| QAIRT / SNPE 2.43.0.260128 | chạy và chuyển model NPU | 246 `~/sigm/qairt/`, 252 `/opt/qairt/`, 249 (bản x86) | — | tải từ Qualcomm (cần tài khoản) |
| MediaMTX | phát luồng chính | mọi Edge `~/edge_artifacts/mediamtx/` | — | bản phát hành của MediaMTX; cấu hình `media/mediamtx.yml` |
| `~/.v4_db.env` | mật khẩu DB | 249 (mode 600) | — | `run_results_v4.sh db` tự sinh; **không chép đi nơi khác** |
| kết quả đo, bản xuất DB | — | laptop nhóm `v50_deploy/meas/` | — | tạo lại bằng `scripts/scenario.sh` + `export_run.sh` |

Đường dẫn chính xác trên từng máy: [RUNBOOK §4](RUNBOOK.md#4-file-model-và-dữ-liệu-không-có-trong-repo) và `scripts/hosts/edge-*.sh`.

## 4. Điểm nối cho web sản phẩm

**Dữ liệu**: TimescaleDB trên Cloud, cổng 25432, DB `v4`; schema và chú thích từng cột ở `scripts/db/schema.sql`. Giờ là UTC.

| Cần hiển thị | Lấy từ |
|---|---|
| box + mã bám trên video | `detections` (toạ độ 0–1 so với khung, `track_id`, `class_name`) nối `frames` theo `(node_id, camera_id, stream_id, stream_epoch, frame_id)` |
| số đếm | `counts` (cửa sổ 5 s: trung bình và lớn nhất từng loại) |
| sự kiện + ảnh | `events` (`event_type`, `started_at`, `ended_at`, `detail`, `snapshot_path`) |
| biển số | view `plate_tracks` (biển tốt nhất mỗi xe) hoặc `plates` (từng lần đọc, `image_path`) |
| đám đông | `crowd` (`persons_est`, `grid` 8×8, `frame_id`) |
| UC4 | `actions` (điểm từng lần kiểm) + `events` loại `fight` / `crash` |

- **Một lần chạy của một camera** = `(node_id, camera_id, stream_id, stream_epoch)`; `stream_epoch` đổi mỗi lần agent khởi động.
- **Video**: mỗi Edge phát luồng chính (H.264 gốc, không mã hoá lại) qua MediaMTX của chính nó:
  `rtsp://<IP Edge>:8554/edge/<node>/<camera>/<stream>/epoch-<epoch>` (WebRTC cổng 8889, cùng đường dẫn).
- **Ghép box với khung hình**: `frames.media_pts_ns` là thời điểm của khung trong luồng chính; `frame_id` là số khung. Với video
  thử, 64 dòng điểm ảnh cuối mỗi khung luồng chính mang số khung (`media/frame_identity.py`, có mã kiểm tra) — web đọc dải đó
  để biết đang hiện khung nào rồi vẽ box của đúng `frame_id`, không dựa vào đồng hồ. `scripts/verify_db_export.py` (phép
  kiểm `band`) làm đúng phép này và là mẫu tham khảo.
- **Ảnh** sự kiện / biển số là file trên Cloud (`~/v4_media/`, xoá sau 30 ngày); DB lưu đường dẫn. Web cần một dịch vụ đọc file.
- Cần dữ liệu trực tiếp thay vì hỏi DB: đọc các topic Kafka kết quả (danh sách ở [ARCHITECTURE §6](ARCHITECTURE.md#6-kafka));
  `cloud_core/results_writer.py` là ví dụ đọc và giải mã từng loại tin.
- **Chưa dùng được**: cột `frames.latency_ms` so đồng hồ Edge với đồng hồ Cloud (247 lệch ~0,36 s) — cần đồng bộ giờ (NTP)
  trước khi hiển thị.

## 5. Điểm nối khác

| Việc | Chỗ cắm |
|---|---|
| DRL chọn cách chạy / FPS | `edge_agent/control_v1.py` (`--policy`; hiện `observe` = cách cố định), `edge_agent/drl_policy.py`, `drl/`; trạng thái mỗi giây ghi ở `--state-jsonl`; tải Cloud về qua `controlv1.cloud.feedback` |
| Đổi tracker (ví dụ OC-SORT) | `cloud_core/tracking.py` (`CloudByteTrack`), dùng ở cả Edge (`control_v1.py`, `--edge-tracking`) và Cloud (`result_hub.py`) |
| Thêm model Cloud làn sau | `triton_config_v4.sh add NAME ONNX`; worker ở `cloud_core/slow_worker.py`; luật Edge ở `edge_agent/*_task.py` |
| Thêm / đổi model trên NPU | [EDGE_NPU §5](EDGE_NPU.md) (quy trình + cổng kiểm), ghim SHA trong lớp adapter và `hosts/edge-*.sh` |
| Ngưỡng luật UC2–UC4, vùng | hồ sơ camera (`V4_PROFILES`, `scripts/profiles/`), `V4_SLOW_CONFIG` |

## 6. Đo và kiểm (đã dùng cho các số trong báo cáo)

| Việc | Lệnh |
|---|---|
| Một kịch bản trên một Edge (4 camera + ghi tài nguyên) | `bash scripts/scenario.sh S1b 246 180 HH:MM:SS` (S0 nghỉ, S1a 4×N, S1b 4×A, S2a 4×C, S2b 4×D, S3 4×B, S4 trộn) |
| Tài nguyên Cloud trong lúc chạy | `python -B scripts/resource_log.py --role cloud --seconds 1200 --out ~/v4_res/S1b` (Ctrl+C khi Edge xong: vẫn in bảng Triton) |
| Xuất DB một Edge + ảnh mẫu | `bash scripts/export_run.sh edge-246 120` (249) |
| Kiểm nội dung mọi bảng | `python -B scripts/verify_db_export.py DIR --videos V --main-videos M --map cam-1=vn9:N …` (laptop) |
| Bảng tổng hợp một đợt đo | `python -B scripts/meas_summary.py <thư mục đo> S1a S1b …` |
| Sức chứa Cloud bằng camera giả | `scripts/cloud_v4_load.py --mode rtdetr\|split\|dfinex --cameras 12 24 36 48` (249; cần topic `cloud.detection-result.v1.load`, lệnh tạo ở đầu file) |
| Độ khớp model trên Triton | `scripts/triton_check.py --models rtdetr_tail_q rtdetr_full …` (cổng ≥ 0,99) |
| F1 cách B / C trên ảnh có nhãn | `scripts/cut_g3_eval.py prep / prep-vn / score` |
| Mạng một Edge → Cloud | `scripts/kafka_ping.py echo` (249) + `ping` (Edge); ping gói lớn: `ping -c 300 -i 0.01 -s 1472 <IP Cloud>` phải mất 0 % |

Kết quả 15/11 (3 Edge × 4 camera): mọi bảng DB đạt ở cả 5 cách; Edge nặng nhất (4 × YOLO11s) ≥ 99,7 % khung; sức chứa Cloud
đo bằng camera giả: D ≈ 14 camera (trần GPU), C và B ≥ 24 camera; B (2 bit, nửa sau TensorRT) tốn ít GPU Cloud hơn C 20–32 %,
F1 0,8413 so với 0,8429 (UA-DETRAC), 0,619 so với 0,610 (video VN, nhãn tự động).

## 7. Việc còn mở khi bàn giao

1. **Mạng của 252**: đường gửi lên mất ≈ 11 % gói lớn (ping 1472 byte), TCP chỉ ~6–10 Mbit/s; 246 cùng phép thử 0 %. Cần
   đổi dây / cổng switch / bộ chuyển mạng. Cho tới lúc đó các cách gửi lên Cloud (B, C, D) trên 252 chỉ đạt 30–60 % khung;
   kịch bản S2a / S2b / S3 / S4 với đủ 3 Edge phải đo lại.
2. C và B trên 24 camera: nghẽn nằm ngoài GPU (worker hay bộ giả lập), chưa tách.
3. Luật UC2–UC4 chưa đo độ đúng với nhãn; ngưỡng là giá trị khởi đầu. Trần 4 lần tìm biển / s / camera chưa đo tối ưu.
4. DM-Count: lô 4 mới đo trên ONNX Runtime (không lợi); chưa đo lại trên TensorRT.
5. F1 cách B 2 bit với nửa đầu NPU trên video VN chưa đo (UA-DETRAC đã đo).
6. `resource_log.py --role cloud` đếm cả lưu lượng cầu nối Docker (số mạng vào của Cloud bị cao).
7. 246: CMA trống thấp dần qua nhiều lần chạy; tua bộ giải mã ~9 s. 252: không tản nhiệt (81–96 °C).
8. DRL chưa cắm; ST-GCN++ chưa huấn luyện lại; lớp nén có huấn luyện cho cách B chưa làm.
