# Edge 246 và 252: cấu tạo, các khối, cách code dùng chúng, và chuyển model sang NPU

*Viết ngày 15/11/2026 (cập nhật 16/11), theo code trong repo này và các số đo đã ghi trong dự án.
Thông số phần cứng chung của chip QCS6490 lấy theo công bố của Qualcomm, nhóm chưa tự đo; những chỗ như vậy đều
có ghi chú.*

---

## 1. Hai máy Edge

Cả hai máy dùng **cùng một chip Qualcomm QCS6490**, nhưng là hai bo mạch khác nhau với hệ điều hành và phần mềm đa
phương tiện khác nhau. Vì vậy code có nhiều nhánh riêng cho từng máy.

| | **246** | **252** |
|---|---|---|
| Bo mạch | Qualcomm RB3 Gen2 | QCS6490 ODK (Yupik) |
| Hệ điều hành | Ubuntu 24.04 (glibc 2.39) | Ubuntu 20.04 (glibc 2.31) |
| Người dùng | `ubuntu` | `root` (không có `sudo`, khởi động lại bằng `reboot`) |
| Số nhân CPU dùng được | 8 | 8 (đo 15/11; ghi chú cũ trong `scripts/hosts/edge-252.sh` ghi 6) |
| Tản nhiệt | có | **không**: lên tới ~96 °C khi 4 camera đủ tải |
| Bộ giải mã video (GStreamer) | `v4l2h264dec` | `qtic2vdec` |
| Đổi cỡ, đổi màu trên GPU (`qtivtransform`) | bản mới, có `engine=gles` | bản cũ, không chọn được engine |
| Ảnh độ phân giải đầy đủ (cho làn sau) | đổi màu trên GPU | **đổi màu trên CPU** (`V4_HIRES_CONVERT=cpu`), vì GPU bị lỗi trang khi 4 camera |
| Bộ SDK NPU | QAIRT/SNPE 2.43, `~/sigm/qairt/2.43.0.260128`, ABI `aarch64-oe-linux-gcc11.2` | QAIRT/SNPE 2.43, `/opt/qairt/2.43.0.260128`, ABI `aarch64-ubuntu-gcc9.4` |
| Chuẩn bị đồ thị NPU trên máy | được | **không được** (bị sập khi chuẩn bị đồ thị HTP), nên mọi DLC phải được chuẩn bị sẵn trên 249 |
| Daemon NPU | biên dịch trên 246 | phải biên dịch lại trên 252, vì bản của 246 cần glibc 2.39 |
| Cấu hình trong code | `scripts/hosts/edge-246.sh` | `scripts/hosts/edge-252.sh` |

Mỗi file cấu hình máy khai báo: đường dẫn Python, SDK, từng file model (DLC) kèm mã SHA-256, daemon NPU, MediaMTX,
và "hồ sơ năng lực" cho DRL (số khung/giây tối đa của từng model khi chạy một mình).

---

## 2. Các khối phần cứng của chip QCS6490 và hệ dùng khối nào

```
                ┌──────────────────────────── QCS6490 ────────────────────────────┐
  file / RTSP ─►│ KHỐI GIẢI MÃ VIDEO ──NV12──► GPU Adreno ──BGR 640x640──► CPU    │
   (H.264)      │ (v4l2h264dec /            (qtivtransform:        (Python:       │
                │  qtic2vdec)                đổi cỡ + đổi màu)      Edge Agent)   │
                │        │                                           │    ▲       │
                │        └──NV12 ảnh lớn──► GPU (246) / CPU (252) ───┘    │       │
                │                                                          ▼  │    │
                │                              NPU Hexagon HTP ◄── /dev/shm + socket
                │                              (snpe-daemon: YOLO, nửa đầu RT-DETR, │
                │                               tìm biển số)                       │
                │  Bộ nhớ chung (LPDDR) + vùng CMA cho bộ đệm video                │
                └──────────────────────────────────────────────────────────────────┘
                                            │ Kafka (LAN 1 Gbit)
                                            ▼
                                        Cloud 249
```

| Khối | Là gì | Hệ dùng nó để | Code |
|---|---|---|---|
| **CPU Kryo** (8 nhân, chia nhóm nhân lớn và nhân nhỏ)¹ | bộ xử lý chính | chạy Edge Agent (Python): chuẩn bị ảnh cho model, giải mã box, NMS, ByteTrack, luật làn sau, nén tensor, gửi Kafka | `edge_agent/` |
| **Khối giải mã video** | phần cứng giải mã H.264 riêng | giải mã luồng phân tích của từng camera ra ảnh NV12; giới hạn 16 phiên giải mã cùng lúc | `media/hw_capture.py` |
| **GPU Adreno**¹ | bộ xử lý đồ hoạ | đổi cỡ 1080x1920 → 640x640 và đổi NV12 → BGR (`qtivtransform`); trên 246 đổi màu cả ảnh lớn | `media/hw_capture.py` (`transform_chain`) |
| **NPU Hexagon, phần HTP** (đời V68)¹ | bộ tăng tốc mạng nơ-ron, tính số nguyên 8 bit là chính | chạy các model đã lượng tử: YOLOv8n (N), YOLO11s (A), nửa đầu RT-DETR (B), model tìm biển số (UC1) | `native/snpe_daemon*`, `legacy/adapters/npu/`, `inference/*_snpe.py` |
| **Bộ nhớ chung + CMA** | RAM dùng chung cho CPU, GPU, NPU và khối video; CMA là vùng liền mạch dành cho bộ đệm video | bộ đệm giải mã; tensor vào/ra NPU qua `/dev/shm` | 246 thường còn 70–125 MB CMA trống |
| **Cảm biến nhiệt** | `/sys/class/thermal` | ghi nhiệt độ vào trạng thái DRL và bộ ghi tài nguyên | `scripts/resource_log.py` |

¹ Thông số công bố của Qualcomm cho QCS6490: CPU Kryo 670 (1 nhân 2,7 GHz, 3 nhân 2,4 GHz, 4 nhân 1,9 GHz), GPU
Adreno 643, NPU Hexagon 770 khoảng 12 TOPS. Nhóm chưa tự đo các con số này. Đời HTP **V68** thì thấy trong code:
đường dẫn thư viện `hexagon-v68` của SDK.

**NPU không có bộ đếm mức bận để đọc**, nên ta chỉ đo được thời gian chạy của mỗi lần gọi, không có "% NPU" như GPU.

---

## 3. Một khung ảnh đi qua Edge như thế nào

Mỗi camera là **một tiến trình Edge Agent** (`edge_agent/control_v1.py`), chạy bằng `scripts/run_edge_v4.sh`.

```
video H.264 (1080x1920)
   │
   ├──► luồng chính: chuyển tiếp nguyên vẹn, không giải mã lại ──► MediaMTX trên Edge ──► web (RTSP/WebRTC)
   │        (media/copy_publisher.py; có dải số khung 64 dòng ở đáy để web tìm đúng khung)
   │
   └──► luồng phân tích: GStreamer
          filesrc/rtspsrc ─► h264parse ─► giải mã phần cứng ─► NV12 ─► tee
                                                                       ├─► nhánh 1: qtivtransform ─► BGR 640x640 ─► ống 1
                                                                       └─► nhánh 2: videorate (1/5 khung) ─► BGR 1080x1920 ─► ống 2
                                                                           (246: qtivtransform; 252: videoconvert trên CPU)
          ống 1 → Python: khung 640x640 (bị kéo vuông)
          ống 2 → Python: ảnh lớn của khung 0, 5, 10, ... (cho biển số, đám đông, UC4)
                │
                ▼
         LÀN NHANH, mỗi camera chọn 1 cách
           N: YOLOv8n 320 trên NPU ─► box ─► ByteTrack trên Edge ─► Kafka (kết quả)
           A: YOLO11s 320 trên NPU ─► box ─► ByteTrack trên Edge ─► Kafka (kết quả)
           B: nửa đầu RT-DETR trên NPU ─► tensor 128x80x80 ─► nén 2 bit ─► Kafka ─► Cloud chạy nửa sau
           C/D: JPEG 640 ─► Kafka ─► Cloud chạy cả model
                │
                ▼
         LÀN SAU: BUS + bộ đệm 3 giây ─► 8 việc (luật; riêng biển số chạy thêm model tìm biển trên NPU)
```

**Vì sao ảnh 640x640 bị kéo vuông**: GPU đổi thẳng 1080x1920 thành 640x640 để tiết kiệm CPU, khoảng 0,23 nhân mỗi
camera thay vì 2 nhân nếu làm bằng OpenCV. Model chạy trên Cloud (RT-DETR, D-FINE) vốn được chuẩn bị ảnh kiểu kéo
vuông nên dùng thẳng. Còn YOLO được huấn luyện trên ảnh giữ tỉ lệ: nếu cho nó ảnh kéo vuông thì F1 của YOLO11s
giảm từ 0,4536 xuống 0,3957. Vì vậy hàm `unstretch()` trong `edge_agent/control_v1.py` ép ảnh lại về tỉ lệ camera
trước khi đưa cho YOLO, và F1 trở lại 0,4512 (VN, 8/11).

---

## 4. Trong code: từng khối được dùng thế nào

### 4.1 Giải mã phần cứng: `media/hw_capture.py`

Lớp `HwCapture` đóng vai `cv2.VideoCapture`, nên phần còn lại của code (`media/opencv_source.py`) không cần biết
khung đến từ phần cứng hay từ OpenCV.

| Việc | Cách làm | Lý do (sự cố đã gặp) |
|---|---|---|
| Dựng pipeline | `gst_source()` + `transform_chain()`, chạy `gst-launch-1.0` như một tiến trình con | GStreamer chạy trong môi trường hệ thống (tắt conda) |
| Lấy khung ra | khung đi qua **ống riêng** (`fdsink fd=...`), không qua stdout | 13/11 trên 252: `qtic2vdec` in 64 KiB chữ ra stdout trước khung đầu, làm mọi ảnh lệch 65 536 byte |
| Hai nhánh | `tee`: nhánh 640 cho model, nhánh ảnh lớn qua `videorate drop-only` (giữ 1/5 khung, bỏ phần còn lại trước khi chép) | ảnh lớn chỉ cần cho làn sau |
| Ảnh lớn trên 252 | `videoconvert` (CPU) thay `qtivtransform` | 14/11: 4 camera đổi màu ảnh lớn trên GPU gây 59 lỗi "kgsl GPU PAGE FAULT" trong 60 giây, kéo theo lỗi giải mã -5. Đổi sang CPU thì 4 × 900 khung chỉ còn 0–1 lỗi |
| Dò kích thước bộ đệm | `_probe_once()` chạy thử 30 khung, đếm dòng `identity`, rồi tự dừng bằng SIGINT; kết quả lưu ở `~/.cache/v4_hw_probe.json`, mỗi lần chỉ một tiến trình dò (khoá file) | GStreamer của 252 bỏ qua `eos-after`; nhiều camera dò cùng lúc thì bị treo, camera khởi động trễ 90–110 giây |
| Dừng | `stop_gracefully()`: gửi SIGINT để GStreamer tự kết thúc và đóng phiên giải mã; `release()` không bao giờ giết cứng | giết cứng để lại "phiên ma" trong trình điều khiển video (`msm_vidc`, tối đa 16 phiên), chỉ khởi động lại máy mới hết |
| Agent chết | `prctl(PR_SET_PDEATHSIG, SIGINT)`: nhân Linux tự gửi SIGINT cho GStreamer | như trên |
| Kẹt khung | đợi tối đa 10 giây (`V4_CAPTURE_STALL_S`), rồi khởi động lại bộ giải mã đúng khung đang dở | 12/11 trên 252: bộ giải mã im lặng 55 giây không báo lỗi |
| Tua tới khung bất kỳ | tìm **khung khoá ngay trước** khung đích (`keyframe_before`, đọc bằng ffprobe), cắt luồng bằng ffmpeg sang MPEG-TS rồi cho `tsdemux` đọc | Matroska làm tròn mốc thời gian tới mili giây, làm `videorate` chọn sai khung |

Khi không có giải mã phần cứng (247, hoặc không đặt `V4_DECODER`), `media/opencv_source.py` giải mã bằng OpenCV trên
CPU. Nếu chậm hơn 2 giây so với thời gian thực thì nó tua tới, tối đa một lần mỗi 5 giây.

### 4.2 Gọi NPU: daemon C++ + Python

Python không gọi NPU trực tiếp mà qua **một tiến trình daemon C++** giữ model luôn nạp sẵn:

```
Edge Agent (Python)                                     snpe-daemon (C++, SNPE C API)
  SnpeDlcSession ── SnpeDaemonHandle ── SnpeDaemonClient ──Unix socket──► nạp DLC một lần, chạy trên DSP/HTP
                                         ghi đầu vào  ──► /dev/shm/... ◄── đọc đầu vào, ghi đầu ra
```

| Thành phần | File | Việc |
|---|---|---|
| Daemon | `native/snpe_daemon/snpe_daemon.cpp` (bản cũ), `native/snpe_daemon_candidate/snpe_daemon.cpp` (bản **v3**) | nạp DLC, mở phiên NPU, chờ lệnh `PING / INFER / INFER_SHM / SHUTDOWN` qua socket |
| Quản lý daemon | `legacy/adapters/npu/snpe_daemon_manager.py` | mỗi DLC một daemon trong mỗi tiến trình camera (tên socket có pid); dùng lại daemon còn sống; dọn thư mục `/dev/shm` khi đóng |
| Kết nối | `legacy/adapters/npu/snpe_daemon_client.py` | gửi và nhận khung qua bộ nhớ dùng chung (`SNPE_DAEMON_SHM=1`) |
| Phiên | `legacy/adapters/npu/snpe_dlc_session.py` | `infer_raw(tensor)` → (thời gian NPU tự báo, các tensor ra theo tên) |
| Môi trường | `legacy/adapters/npu/snpe_env.py` | `LD_LIBRARY_PATH` (thư viện SNPE cho CPU ARM) và `ADSP_LIBRARY_PATH` (thư viện `hexagon-v68` chạy trên NPU) |

**Daemon v3** (`SNPE_DAEMON_GRAPH_DSP_V3`) khác bản cũ ở 3 điểm:
1. Phải nêu **tên đồ thị** bên trong DLC (`--graph`) và **tên tensor ra** (`--outputs boxes,scores`).
2. **Cấm chạy dự phòng trên CPU**: NPU không sẵn sàng thì báo lỗi luôn. Sự cố 12/11 cho thấy lý do: trên 252, cách N lỡ
   chạy bằng daemon cũ, model rơi xuống CPU, tốn khoảng 300 % CPU, chậm 25 lần, F1 0,463, máy lên 96,6 °C.
3. Python kiểm "hợp đồng" trước khi dùng (`--contract-version` phải trả đúng chuỗi trên).

Daemon được biên dịch trên chính Edge (`scripts/setup_edge_node.sh` biên dịch cả hai bản daemon theo ABI của máy), vì 246
và 252 có glibc khác nhau.

**Dọn dẹp**: một daemon bị bỏ lại vẫn giữ phiên NPU, các lần sau báo "DSP unavailable". Vì vậy agent đóng mọi
daemon nó mở khi kết thúc (`finally` trong `control_v1.py`), và `run_edge_v4.sh` gọi `clean_leftovers` trước và sau
mỗi lần chạy (dọn agent, daemon NPU, ffmpeg còn sót, nhưng giữ MediaMTX).

### 4.3 Các model chạy trên NPU

| Dùng cho | Lớp Python | File DLC trên 246 | Trên 252 | Đầu vào | Đầu ra |
|---|---|---|---|---|---|
| **N**: YOLOv8n 320 | `inference/adapters.py` `YoloSnpeAdapter` → `legacy/adapters/yolo/yolo_snpe.py` | `~/edge_artifacts/npu243/yolov8n_320_int8_split_traffic100.dlc` | bản đã chuẩn bị đồ thị trên 249 (`npu243_prepared/`), đồ thị `yolov8n_320_split` | 320x320x3 float, NHWC, letterbox | `boxes`, `scores` |
| **A**: YOLO11s 320 | `inference/yolo11s_snpe.py` `Yolo11sSnpeAdapter` (+ `yolo11s_contract.py`) | `yolo11s320_w8a8_qcs6490.dlc` (sha `1bb6a516…`) | `yolo11s320_w8a8_qcs6490_dual.dlc` (sha `4e1e5246…`, cùng trọng số) | 320x320x3 float, NHWC | `boxes` [1, 2100, 4], `scores` [1, 2100, 80] |
| **B**: nửa đầu RT-DETRv2-S | `inference/rtdetr_head_snpe.py` `RtdetrHeadSnpe` | `edge_b_20261106/head_w8a8_pc.dlc` (sha `2e0c4b09…`), đồ thị `head_fp32` | **cùng file** (đã có sẵn bộ đệm đồ thị cho QCS6490) | 640x640x3 float, NHWC | 1 tensor [1, 80, 80, 128] |
| **UC1**: tìm biển số | `edge_agent/plate_branch.py` → `YOLOSnpeDetector(variant="plate")` | `~/plate_ocr_home/edge/plate_yolov8n_320_int8_split.dlc` | `..._qcs6490.dlc`, đồ thị `plate_320_split_fp32` | 320x320x3, **NHWC** | `boxes`, `scores` |

- Mọi lớp đều **kiểm SHA-256 của file model** trước khi chạy: sai file thì dừng, không âm thầm chạy model khác.
- Chữ **"split"** trong tên YOLO **không phải là chia model**. Nó nghĩa là đầu ra được tách thành hai tensor riêng (`boxes`
  và `scores`) thay vì một tensor gộp. Khi lượng tử 8 bit, mỗi tensor dùng chung một thang đo; toạ độ box (0–320) và
  điểm (0–1) mà nằm chung một tensor thì điểm sẽ mất gần hết độ chính xác. Đây là lý do thường gặp của cách tách
  này; nhóm chưa đo phương án gộp. Sau khi NPU trả ra, CPU làm phần còn lại: giải mã box, lọc ngưỡng và NMS
  (`yolo11s_contract.decode`, NMS dạng ma trận numpy).
- Bộ chọn model theo cách chạy nằm ở `inference/catalog.py` (`phase7_edge_catalog`). Agent nhận đường dẫn DLC qua tham số
  (`--yolo-dlc`, `--yolo11s-dlc`, `--rtdetr-head-dlc`, `--plate-detector`) do `run_edge_v4.sh` điền từ file cấu hình máy.
- Một NPU phục vụ nhiều phiên: 4 camera × (model làn nhanh + model biển số) + nửa đầu B. Ghi nhận cũ (29/10, v50.9):
  8 phiên chạy cùng lúc từng làm DSP của 246 sập (mã lỗi 911). Các lần chạy 4 camera ngày 15/11 thì không gặp lại.

---

## 5. Chuyển một model bình thường sang chạy được trên NPU

Model huấn luyện bằng PyTorch tính số thực 32 bit. NPU HTP của QCS6490 chạy nhanh nhất với **số nguyên 8 bit**. Việc
chuyển đổi làm trên **Cloud 249** (bộ công cụ SNPE 2.43 bản x86), rồi chép file DLC sang Edge.

```
PyTorch (.pt)
   │ 1. xuất ONNX: kích thước cố định, tách đầu ra (boxes / scores), bỏ phần NMS
   ▼
model.onnx ──────────────────────────────────────► (247: chạy thẳng ONNX trên GPU bằng ONNX Runtime CUDA)
   │ 2. snpe-onnx-to-dlc            → DLC số thực (đầu vào tự đổi thành NHWC)
   │    snpe-dlc-info               → KIỂM: tên, kích thước đầu vào/ra, tên đồ thị
   ▼
model_fp32.dlc
   │ 3. ảnh hiệu chỉnh: vài trăm ảnh thật, chuẩn bị Y HỆT lúc chạy (cỡ, RGB, /255, NHWC) → .raw + input_list.txt
   │ 4. snpe-dlc-quant(ize)         → W8A8: trọng số 8 bit theo từng kênh, kích hoạt 8 bit, bias 32 bit
   ▼
model_w8a8.dlc
   │ 5. snpe-dlc-graph-prepare --htp_socs qcs6490 --set_output_tensors "<đồ thị> boxes,scores"
   │                                → nhúng sẵn đồ thị đã biên dịch cho HTP của QCS6490 (252 bắt buộc cần)
   ▼
model_w8a8_qcs6490.dlc
   │ 6. CỔNG KIỂM trên Edge thật: snpe-net-run --use_dsp so với model số thực
   │      - tensor ra: tương quan ≥ 0,99
   │      - box: trùng với model số thực (IoU ≥ 0,5)
   │      - F1 trên ảnh có nhãn không tụt đáng kể
   │ 7. ghi SHA-256 vào code + file cấu hình máy
   ▼
dùng trong hệ (daemon v3, đúng tên đồ thị, đúng tên tensor ra)
```

### Các bước và lý do

| Bước | Công cụ / script | Điều cần đúng | Nếu sai thì |
|---|---|---|---|
| 1. ONNX | xuất từ PyTorch (YOLO: ultralytics; RT-DETR/D-FINE: `scripts/candidate_artifact.py`) | kích thước cố định; NMS để ngoài model (CPU làm) | toán tử động hoặc NMS trong đồ thị thường không chạy được trên HTP |
| 2. Đổi sang DLC | `snpe-onnx-to-dlc --input_network m.onnx --input_dim images 1,3,H,W` | xem lại bằng `snpe-dlc-info`: **đầu vào thành NHWC** [1, H, W, 3] | đưa nhầm NCHW thì NPU vẫn chạy nhưng ra rác. 11/11: model biển số cho box IoU 0/49 khi đưa NCHW, 49/49 khi đưa NHWC; lỗi này tồn tại từ 8/11 tới khi phát hiện |
| 3. Ảnh hiệu chỉnh | `scripts/cut_head_raw.py` (nửa đầu RT-DETR), các script `*_quant*` | **giống hệt** cách chuẩn bị ảnh lúc chạy (`INTER_AREA`, RGB, /255, NHWC); ảnh thật của bài toán | sai cách chuẩn bị thì thang đo lượng tử sai, độ chính xác tụt |
| 4. Lượng tử | `snpe-dlc-quant` (bản mới) hoặc `snpe-dlc-quantize` | `--use_per_channel_quantization --bias_bitwidth 32`; thử thêm `--act_bitwidth 16` (W8A16) và `--algorithms cle` | thang đo chung cho cả lớp làm mất độ chính xác ở các kênh nhỏ (ví dụ nửa đầu RT-DETR, mục 6) |
| 5. Chuẩn bị đồ thị | `snpe-dlc-graph-prepare --htp_socs qcs6490` | chỉ định đúng tensor ra | 252 sập khi tự chuẩn bị đồ thị trên máy; không có bước này thì 252 không chạy được |
| 6. Cổng kiểm | `snpe-net-run --use_dsp` trên Edge; `scripts/yolo11s_snpe_candidate.py`, `scripts/cut_g2_check.py`, `scripts/edge_head_check.py`, `scripts/cut_g3_eval.py` | so với model số thực trên cùng ảnh | không qua cổng thì không đưa vào hệ |
| 7. Ghim | hằng `DLC_SHA256` trong lớp Python + `*_DLC_SHA` trong `scripts/hosts/edge-*.sh` | | file khác bị từ chối khi khởi động |

### Bài học từ những lần chuyển đổi

- **Không phải model nào cũng lượng tử được.** RTMDet-s (10/2026) đã thử nhiều cách (W8A8, W8A16, theo kênh,
  MSE), lần nào cũng 0 box trùng với bản số thực, nên dừng hẳn. Vì vậy hệ chỉ giữ các model đã qua cổng.
- **Nửa đầu D-FINE trên NPU từng ra tensor toàn 0 hoặc hằng số.** Phải kiểm giá trị tensor trước khi đo tốc độ.
  Mã cũ ở `legacy/adapters/dfine/` có sẵn bộ chặn trường hợp này.
- **Một file DLC đã chuẩn bị đồ thị cho QCS6490 thì chạy được trên cả hai máy.** `head_w8a8_pc.dlc` đã có sẵn bộ đệm
  `HTP_V68_QCS6490_2MB`, nên 252 dùng chung file với 246.
- **Đúng tên đồ thị.** Mỗi DLC có tên đồ thị bên trong (xem bằng `snpe-dlc-info`). Daemon v3 cần đúng tên, nên file cấu
  hình 252 khai thêm `YOLOV8N_SNPE_GRAPH` và `PLATE_SNPE_GRAPH`.

---

## 6. Model chia đôi (cách B): RT-DETRv2-S cắt sau layer2

### 6.1 Ý tưởng
Chạy **phần đầu của model trên NPU của Edge**, gửi **tensor trung gian** lên Cloud, Cloud chạy **phần còn lại**. Edge
gánh bớt việc cho GPU Cloud. Cái giá là tensor trung gian phải đủ nhỏ để gửi qua mạng, và Edge phải chạy được phần
đầu đủ nhanh.

```
Edge (NPU)                                          Cloud 249 (Triton)
ảnh 640x640 ─► stem + layer1 + layer2 của backbone ─► tensor 128 x 80 x 80 ─► nén 2 bit ─► Kafka ─►
             (ResNet-18vd của RT-DETRv2-S)            (819 200 số)             (~90 KB)
                                                       ─► giải nén ─► layer3, layer4, encoder, decoder ─► box ─► ByteTrack
                                                          (rtdetr_tail_q, GPU)
```

### 6.2 Chọn điểm cắt: `scripts/cut_points.py`
- `--profile` liệt kê các **điểm cắt một tensor**, tức những chỗ mà mọi thông tin từ ảnh chỉ đi qua đúng một tensor.
  Với mỗi chỗ, nó cho biết số giá trị phải gửi và phần trăm phép tính nằm ở Edge.
- RT-DETRv2-S: chỉ **sau layer2** (`/model/backbone/res_layers.1/blocks.1/act/Relu`) là cắt một tensor gọn. Sau toàn bộ
  backbone thì phải gửi 3 tensor.
- **D-FINE-X không chia**: hai điểm cắt một tensor chỉ giao cho Edge 8 % hoặc 28 % phép tính, hoặc phải gửi 1,55 MB mỗi
  khung. Không có điểm nào vừa giao nhiều việc vừa gửi ít.

### 6.3 Tách model và cổng kiểm G1
`split()` trong `cut_points.py` dùng `onnx.utils.Extractor.extract_model`:
- `head.onnx`: `images` → tensor cắt;
- `tail.onnx`: tensor cắt + `orig_target_sizes` → `boxes / scores / labels`.

**G1**: nửa đầu nối nửa sau phải ra cùng box với model gốc. Đã đạt.

### 6.4 Chuyển nửa đầu sang NPU: `scripts/convert_cut_head_snpe.sh` (chạy trên 249)

| Giai đoạn | Lệnh | Kết quả |
|---|---|---|
| `convert` | `snpe-onnx-to-dlc ... --input_dim images 1,3,640,640` → `head_fp32.dlc` + `fp32_info.txt` | **dừng lại để người kiểm**: đầu vào phải là [1, 640, 640, 3], đầu ra [1, 80, 80, 128] (NHWC) |
| `quantize` | `CUT_IO_CONFIRMED=1 ... quantize`: W8A8 và W8A16 mặc định, `--enable_htp --htp_socs qcs6490` | **G2 không đạt**: tương quan khoảng 0,98; W8A16 cũng chỉ nhích lên chút ít, nên lỗi nằm ở **trọng số** |
| `quantize_pc` | thêm `--use_per_channel_quantization --bias_bitwidth 32`, ba bản: `pc 8`, `pc 16`, `pccle 8` | `head_w8a8_pc.dlc` **đạt G2** và được chọn |

Ảnh hiệu chỉnh tạo bằng `scripts/cut_head_raw.py`, chuẩn bị đúng như model Cloud (`INTER_AREA` 640, RGB, /255, NHWC).
Script từ chối ghi đè thư mục đã có, để không trộn hai bộ ảnh hiệu chỉnh.

### 6.5 Các cổng kiểm của nửa đầu

| Cổng | Kiểm gì | Script | Kết quả |
|---|---|---|---|
| **G1** | nửa đầu + nửa sau (số thực) = model gốc | `cut_points.py` | đạt |
| **G2** | tensor của NPU so với nửa đầu số thực, cùng ảnh: tương quan ≥ 0,99; box nửa sau giữ nguyên | `cut_g2_check.py`, `edge_head_check.py` (chạy qua daemon thật) | mặc định ~0,98 (không đạt) → theo kênh: **đạt** |
| **G3** | F1 trên UA-DETRAC có nhãn thật (xe, điểm ≥ 0,6, IoU 0,5) | `cut_g3_eval.py` (đầu ra NPU từ `snpe-net-run`) | xem bảng dưới |

Kết quả G3 (5/11, 2272 xe có nhãn):

| Phương án | F1 |
|---|---|
| Model gốc, ảnh không nén | 0,8447 |
| C: model gốc, ảnh JPEG q85 | 0,8429 |
| B: nửa đầu số thực + tensor uint8 | 0,8445 |
| **B: nửa đầu NPU W8A8 theo kênh + tensor uint8** | **0,8444** |
| B: nửa đầu NPU W8A16 theo kênh + tensor uint8 | 0,8436 |

Nghĩa là chạy nửa đầu trên NPU **không làm giảm độ chính xác** so với chạy cả model trên Cloud.

### 6.6 Nén tensor để gửi: `contracts/cut_tensor.py`

| Gói | Cách nén | Cỡ mỗi khung | F1 (UA-DETRAC, nửa đầu số thực) |
|---|---|---|---|
| `CUT1` (cũ, `V4_CUT_BITS=8`) | 8 bit, một min/max cho cả tensor, zstd | ~450 KB | 0,8445 |
| **`CUT2` (mặc định từ 14/11)** | **2 bit**, min/max **riêng từng kênh** (128 kênh), gói 4 giá trị vào 1 byte, zstd | **~90 KB** | 0,839 |
| (để so) JPEG q85 của cách C | | ~76 KB | 0,843 |

- Vì sao min/max theo kênh: ở 2 bit mà dùng một thang chung, các kênh có giá trị nhỏ bị về 0 hết, F1 chỉ còn 0,545
  (`scripts/compress_survey.py`, 40 cách nén thử không cần huấn luyện).
- Gói có phần đầu 33 byte: mã nhận dạng, **mã nửa đầu** (`rtl2-w8a8pc`, 247 là `rtl2-onnx-fp32`), số bit, cỡ ảnh gốc,
  kích thước tensor. Cloud **từ chối** gói có mã nửa đầu lạ (`RTDETR_SPLIT_HEAD_ID`), để không bao giờ ghép nhầm nửa đầu
  này với nửa sau khác.
- Trên Cloud, `rtdetr_tail_q` (`scripts/tail_quant.py`) nhận thẳng mã 2 bit cùng min/bước của từng kênh và giải lượng tử
  trên GPU. Box ra giống hệt nửa sau số thực. Từ 15/11 nó chạy TensorRT FP32 (6,5 ms/lô so với 8,6 ms của cả model).
- G3 cho 2 bit với nửa đầu chạy **thật trên NPU 246** (15/11, cùng 200 khung): **F1 0,8413, 89 KB** (C JPEG 0,8429). Trên
  video VN (nhãn tự động, nửa đầu số thực): 2 bit 0,619, 8 bit 0,628, JPEG 0,610; chưa đo với nửa đầu NPU.

### 6.7 Mỗi khung trên Edge (cách B): `RtdetrHeadSnpe.encode()`
1. `prepare`: ảnh → 640x640 `INTER_AREA`, RGB, /255, float32 NHWC.
2. `npu`: daemon v3 chạy `head_w8a8_pc.dlc` trên DSP. Kết quả là một vùng bộ nhớ dùng chung mà khung sau sẽ ghi đè, nên
   phải chép ra ngay trong lúc giữ khoá.
3. Đổi NHWC → NCHW, rồi `pack(bits=2)`.
4. Ghi thời gian từng bước (`prepare_ms`, `npu_ms`, `pack_ms`) vào kết quả để DRL và báo cáo dùng.

247 (GPU) dùng `RtdetrHeadOnnx`: cùng cách chuẩn bị ảnh, cùng gói, nửa đầu chạy bằng ONNX Runtime CUDA.

### 6.8 Số đo khi chạy thật
- Băng thông lên của 246 (4 camera trộn cách): **41,5 → 18,5 Mbit/s** khi chuyển cách B từ uint8 sang 2 bit; box cách B
  tốt ngang bản 8 bit.
- 15/11, 3 Edge × 4 camera: camera chạy B trên 252 đạt **85 %** thời gian thực, thấp nhất trong 4 cách. NPU của 252 yếu
  hơn và đang gánh cả model biển số.

---

## 7. Khác biệt cần nhớ khi vận hành 246 và 252

| Vấn đề | 246 | 252 | Code xử lý |
|---|---|---|---|
| Bộ giải mã | `v4l2h264dec`; dừng chậm (tua mất ~9 giây) | `qtic2vdec`; in nhiều chữ ra stdout | khung qua ống riêng; tua từ khung khoá |
| Đổi màu ảnh lớn | GPU (`qtivtransform engine=gles`) | CPU (`videoconvert`), tránh lỗi trang GPU | `V4_HIRES_CONVERT=cpu` trong `edge-252.sh` |
| Nhiệt | ~69–79 °C | lên ~96 °C với 4 camera, không có tản nhiệt | nhiệt độ có trong trạng thái DRL; nên lắp tản nhiệt trước khi chạy dài |
| Chuẩn bị đồ thị NPU | tự làm được | phải làm sẵn trên 249 | DLC riêng `*_qcs6490.dlc` / `npu243_prepared/` |
| Daemon NPU | biên dịch trên 246 | biên dịch trên 252 | `scripts/setup_edge_node.sh` |
| Phiên giải mã ma | có gặp (`msm_vidc` 16/16 phiên khi không còn bộ giải mã nào) | | không bao giờ giết cứng GStreamer |
| Bộ nhớ CMA | 70–125 MB trống, có trôi dần | | còn mở |
| Lập lịch CPU | `uclampset -m 512` (đặt việc tiền xử lý lên nhân lớn: YOLOv8n 61,9 → 84,9 khung/giây) | để mặc định (tăng tốc thì nóng nhanh hơn) | `EDGE_UCLAMP_MIN` |

Số đo 15/11 (3 Edge × 4 camera, 180 giây, đủ 8 việc làn sau):
- 246: các camera đạt 95–98 % thời gian thực;
- 252: N/A/B/D lần lượt 98/98/85/89 %, tối đa 96 °C, 0 lỗi GPU;
- biển số 60/60 ở mọi máy.

---

## 8. Còn mở
- Thử huấn luyện một lớp nén cho tensor cách B (để xuống ~76 KB mà F1 ≥ 0,83).
- 252: lắp tản nhiệt; thử lại 5 camera.
- 246: tua chậm (~9 giây); CMA trôi dần khi chạy lâu.
- Có thể thay model: D-FINE-L thay D-FINE-X trên Cloud, YOLO11n thay YOLOv8n trên Edge. Mỗi lần thay phải đi lại đủ
  các bước ở mục 5.

## Phụ lục: file cần biết

| Việc | File |
|---|---|
| Cấu hình từng máy | `scripts/hosts/edge-246.sh`, `scripts/hosts/edge-252.sh` |
| Chạy nhiều camera | `scripts/run_edge_v4.sh`, `scripts/scenario.sh` |
| Agent một camera | `edge_agent/control_v1.py` |
| Giải mã phần cứng | `media/hw_capture.py`, `scripts/hw_frame_dump.py`, `scripts/hw_stress.py` |
| Daemon NPU | `native/snpe_daemon*/snpe_daemon.cpp`, `legacy/adapters/npu/` |
| Model trên Edge | `inference/adapters.py`, `inference/yolo11s_snpe.py`, `yolo11s_contract.py`, `inference/rtdetr_head_snpe.py`, `edge_agent/plate_branch.py` |
| Chuyển YOLO11s sang NPU | `scripts/yolo11s_snpe_candidate.py` |
| Chia và chuyển RT-DETR | `scripts/cut_points.py`, `scripts/cut_head_raw.py`, `scripts/convert_cut_head_snpe.sh`, `scripts/cut_g2_check.py`, `scripts/edge_head_check.py`, `scripts/cut_g3_eval.py` |
| Nén tensor | `contracts/cut_tensor.py`, `scripts/compress_survey.py`, `scripts/tail_quant.py` |
