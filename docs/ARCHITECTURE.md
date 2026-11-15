# Kiến trúc v4

## 1. Các máy

| Máy | Vai trò | Phần cứng / hệ điều hành | Ghi chú |
|---|---|---|---|
| **249** | Cloud | RTX 4090, Ubuntu; Docker: Kafka (`v4-kafka`, 29092), Triton (`v4-triton`, 18000/18001/18002), TimescaleDB (`v4-tsdb`, 25432) | worker làn nhanh, làn sau, OCR, ghi DB |
| **246** | Edge | Qualcomm RB3gen2 (QCS6490), Ubuntu 24 | NPU HTP, giải mã `v4l2h264dec`, CMA ~125 MB |
| **252** | Edge | QCS6490 ODK, Ubuntu 20, không tản nhiệt | giải mã `qtic2vdec`, `qtivtransform` bản cũ (ảnh lớn đổi màu bằng CPU, xem §2), daemon NPU "v3" theo tên đồ thị, nóng tới ~96 °C khi 4 camera đủ tải |
| **247** | Edge | PC Windows, RTX 5070 Ti, Git Bash | model ONNX trên GPU, giải mã OpenCV (CPU) |

Mạng LAN 1 Gbit. Edge tự quyết định cho từng camera; Cloud phục vụ và báo tải ngược lại.

## 2. Thành phần

```
Camera ─► Edge Agent (1 tiến trình / camera)
            ├─ nguồn khung: giải mã phần cứng (640x640 cho model + nhánh ảnh lớn mỗi 5 khung)
            │     ảnh lớn đổi màu trên GPU (qtivtransform) ở 246, trên CPU (videoconvert) ở 252
            ├─ luồng chính → MediaMTX trên Edge (RTSP/WebRTC cho web)
            ├─ LÀN NHANH: 1 trong 5 cách (bảng dưới) → box + mã bám
            │     └─ kết quả → Kafka → DB (frames, detections)
            └─ LÀN SAU: BUS + bộ đệm khung 3 s → 8 việc (luật) → Kafka
                                                     │
Cloud 249 ◄──────────────── Kafka ◄──────────────────┘
  ├─ worker làn nhanh (4 tiến trình, cloud_app + grpc_service): vé suy luận → Triton → ByteTrack → kết quả về Edge
  ├─ worker làn sau (slow_worker): UC3 DM-Count, UC4 RTMO-s + ST-GCN++ → bảng + sự kiện
  ├─ worker OCR biển số (16 luồng chuẩn bị dòng chữ → Triton plate_ocr gom lô) → bảng plates
  ├─ Triton: model làn nhanh (ưu tiên 1) + làn sau (ưu tiên 4) dưới bộ giới hạn GPU
  └─ results_writer: mọi topic kết quả → TimescaleDB
```

## 3. Làn nhanh: 5 cách chạy (mỗi camera chọn 1)

| Cách | Model | Chạy ở đâu | Gửi lên Cloud | Bám đối tượng |
|---|---|---|---|---|
| **N** | YOLOv8n 320 | Edge (NPU / GPU 247) | chỉ kết quả (cho DB) | Edge |
| **A** | YOLO11s 320 | Edge (NPU / GPU 247) | chỉ kết quả | Edge |
| **B** | RT-DETRv2-S **chia đôi** sau layer2 | nửa đầu Edge (NPU W8A8 / ONNX 247), nửa sau Cloud | tensor **2 bit theo kênh** + zstd (~90 KB; `V4_CUT_BITS=8`: uint8 ~450 KB) | Cloud |
| **C** | RT-DETRv2-S đầy đủ | Cloud (Triton) | JPEG | Cloud |
| **D** | D-FINE-X đầy đủ | Cloud (Triton) | JPEG | Cloud |

- Khung 640x640 bị kéo vuông bởi bộ giải mã; model chạy trên Edge (N, A) được ép lại đúng tỉ lệ camera.
- Mỗi camera tối đa 2 khung đang chờ Cloud; quá hạn thì bỏ (Edge không chờ).
- Kết quả của Cloud về hộp riêng của từng Edge (`<topic>.<node_id>`).
- Không chia D-FINE-X: không có điểm cắt vừa giao nhiều việc cho Edge vừa gửi ít dữ liệu (8 % việc hoặc 1,55 MB/khung).
- Gói tensor B (`contracts/cut_tensor.py`): `CUT2` = 2 bit mỗi giá trị, min/max riêng từng kênh, F1 0,839 so với
  0,8445 của uint8 trên UA-DETRAC (`scripts/compress_survey.py`, 40 cách nén không cần huấn luyện); Cloud vẫn đọc gói
  `CUT1` (uint8) cũ.

## 4. Làn sau

Edge chỉ chạy **luật** trên box + mã bám (riêng UC1 chạy thêm model tìm biển). Việc nặng chạy trên Cloud.

| Việc | UC | Edge | Cloud |
|---|---|---|---|
| `plates` | UC1 | YOLOv8n 320 tìm biển trong ảnh xe (ảnh lớn), cắt biển, tối đa 2 lần / xe | đọc chữ (mạng EasyOCR trên Triton, gom lô các dòng chữ) → `plates` |
| `count` | UC2/UC3 | đếm người, từng loại xe mỗi 5 s | → `counts` |
| `congestion` | UC2 | ≥ N xe và phần lớn đứng yên, giữ X s → sự kiện | → `events` |
| `crowd` | UC3 | đông người → 1 ảnh lớn / 5 s | DM-Count (mật độ, lưới 8x8) → `crowd`, sự kiện khi rất đông |
| `gathering` | UC3 | nhóm người đứng sát nhau lâu → sự kiện | → `events` |
| `fight` | UC4 | người đi bộ sát nhau + quẫy động → 10 ảnh cắt / 2 s | RTMO-s (khung xương) + ST-GCN++ → `actions`, sự kiện "fight" |
| `crash` | UC4 | xe máy đang chạy → dừng + nằm ngang 2 s → 1 ảnh cắt | RTMO-s: có người nằm → `actions`, sự kiện "crash" |
| `loiter` | UC4 | 1 người ở yên > 2 phút → sự kiện | → `events` |

- Hồ sơ camera (`--camera-profile`): bật việc nào, ngưỡng, vùng (`area`, `road`, `watch`).
- Sự kiện có bắt đầu/kết thúc, kèm ảnh minh hoạ (file trên 249, đường dẫn trong DB).
- Sự kiện qua **hộp thư gửi bù** (ghi file + fsync) trước khi gửi: mất mạng không mất sự kiện.
- Cloud bận → gửi phản hồi giãn nhịp, Edge gửi thưa lại. Tin quá hạn (ảnh đám đông 10 s, clip 15 s) bị bỏ, không tốn GPU.

## 5. Triton (249)

| Model | Dùng cho | Cấu hình |
|---|---|---|
| `rtdetr_full` | C | 2 bản, **TensorRT FP32**, lô ≤ 4, chờ gom ≤ 8 ms |
| `dfinex_full` | D | 2 bản, **TensorRT FP32**, lô ≤ 4 |
| `rtdetr_tail_q` | B | 2 bản, **TensorRT FP32** (15/11; box khớp 1,000, 6,5 ms/lô so với 10,6 ms trên ONNX Runtime); nhận mã lượng tử + min/bước theo kênh, giải lượng tử trên GPU (`scripts/tail_quant.py`, box giống hệt nửa sau số thực) |
| `rtdetr_tail_u8`, `rtdetr_tail` | (dự phòng) | nửa sau cũ (uint8 / số thực), không còn dùng |
| `crowd_density` | UC3 | lô 1 (không gom: ảnh 1280 chiếm trọn GPU), **TensorRT FP32**, ưu tiên thấp |
| `pose_rtmo_s` | UC4 | lô ≤ 4, ưu tiên thấp |
| `action_stgcnpp` | UC4 | lô ≤ 8, ưu tiên thấp |
| `plate_ocr` | UC1 | lô ≤ 64 dòng chữ, chờ gom ≤ 2 ms, ưu tiên thấp; dòng cùng bề rộng (bội số 64) của mọi camera chạy chung |

- Bộ giới hạn GPU 6 suất: làn nhanh 1 suất/lần chạy (ưu tiên 1); model làn sau chiếm cả 6 suất (chỉ chạy khi GPU rảnh làn nhanh).
- Hàng đợi làn nhanh tối đa 64 yêu cầu, chờ tối đa 1 s rồi từ chối (không phục vụ muộn).
- TensorRT FP16 nhanh hơn 2–3 lần nhưng lệch box (khớp 0,89–0,95 < 0,99) → không dùng; FP32 khớp 1,000.
- Worker dùng bộ nhớ dùng chung với Triton: một vùng cho mỗi luồng và mỗi đầu vào, lớn dần khi cần (14/11 sửa rò:
  trước đó 1255 vùng / 16 GB); tự đăng ký lại khi Triton khởi động lại. `plate_ocr` gửi thẳng trong yêu cầu.

## 6. Kafka

- Làn nhanh: vé suy luận (JPEG / tensor), kết quả (chung + riêng từng Edge), kết quả Edge tự bám (`edge.tracked-result.v1`), phản hồi điều khiển.
- Làn sau: `edge.count.v1`, `edge.event.v1`, `edge.plate-crop.v1`, `edge.crowd-frame.v1`, `edge.action-clip.v1`, `edge.crash-frame.v1` → `cloud.plate-read.v1`, `cloud.crowd.v1`, `cloud.action.v1`, `cloud.slow-feedback.v1`.
- Khoá tin = camera → giữ thứ tự từng camera, chia việc giữa các worker. Topic suy luận giữ tin ngắn (60 s).

## 7. Cơ sở dữ liệu (TimescaleDB, `scripts/db/schema.sql`)

| Bảng | Nội dung | Giữ |
|---|---|---|
| `frames` | mỗi khung: camera, cách chạy, model, số box, độ trễ | 7 ngày |
| `detections` | mỗi box: toạ độ, lớp, độ tin cậy, mã bám | 7 ngày |
| `counts` | đếm theo cửa sổ 5 s | 30 ngày |
| `events` | mọi sự kiện (bắt đầu/kết thúc, chi tiết, ảnh) | 90 ngày (dọn mỗi giờ) |
| `plates`, view `plate_tracks` | lần đọc biển, biển tốt nhất mỗi xe | 30 ngày |
| `crowd` | ước lượng đám đông + lưới 8x8 | 30 ngày |
| `actions` | điểm mỗi lần kiểm UC4 | 30 ngày |

Ảnh minh hoạ / ảnh biển: `~/v4_media/` trên 249, xoá sau 30 ngày.

## 8. DRL (chưa cắm vào v4)

Mỗi Edge một agent: quan sát trạng thái từng camera (độ trễ, hàng đợi, tải Cloud, CPU, nhiệt độ, độ mờ, số xe…) và
chọn cách chạy + FPS. Hiện các camera chạy cách cố định (`--policy observe`); khung DRL ở `drl/`, `edge_agent/drl_policy.py`.
Hướng đã thống nhất: chạy đủ tải (kể cả quá tải) để DRL học; kịch bản đa dạng (nội dung, số camera, mạng, tải Cloud, sự cố).
