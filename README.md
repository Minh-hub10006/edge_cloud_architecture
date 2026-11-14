# Edge–Cloud video analytics (v4)

Hệ phân tích video giao thông chia việc giữa **Edge** (Qualcomm QCS6490 / PC GPU) và **Cloud** (RTX 4090 + Triton):
mỗi camera chạy **làn nhanh** (phát hiện + bám đối tượng, 1 trong 5 cách N/A/B/C/D) và **làn sau** (UC1–UC4:
biển số, đếm và ùn tắc, đám đông, đánh nhau / ngã xe / lảng vảng). Kết quả của cả hai làn vào TimescaleDB.

| Tài liệu | Nội dung |
|---|---|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | kiến trúc: máy, thành phần, 5 cách chạy, làn sau, Triton, Kafka, DB |
| [docs/FLOWS.md](docs/FLOWS.md) | các luồng dữ liệu và luồng xử lý sự cố |
| [docs/RUNBOOK.md](docs/RUNBOOK.md) | cài, chạy, kiểm tra trên từng máy; file model; lỗi đã gặp và cách xử lý |

## Cấu trúc thư mục

| Thư mục | Vai trò |
|---|---|
| `edge_agent/` | Edge Agent mỗi camera (`control_v1.py`): làn nhanh, làn sau (`slow_lane.py`, `*_task.py`), sự kiện + hộp thư gửi bù (`events.py`) |
| `cloud_core/` | worker Cloud: làn nhanh (`grpc_service.py`, `tracking.py`), làn sau (`slow_worker.py`, `uc3_crowd.py`, `uc4_disorder.py`, `plate_worker.py`), ghi DB (`results_writer.py`) |
| `contracts/` | kiểu dữ liệu chung, Kafka (`kafka_transport.py`), tensor chia đôi (`cut_tensor.py`), proto |
| `inference/` | model: Triton client (`triton_detr.py`), NPU (`*_snpe.py`), đám đông, khung xương, hành động |
| `media/` | giải mã phần cứng (`hw_capture.py`), nguồn khung, phát luồng chính lên MediaMTX |
| `web/` | Cloud app của worker làn nhanh (`cloud_app.py`) |
| `drl/`, `benchmark/` | khung DRL và công cụ đo (dùng bởi agent / script) |
| `legacy/adapters/` | adapter NPU/SNPE (daemon, YOLO trên NPU) mà v4 dùng lại |
| `native/` | mã nguồn daemon SNPE |
| `scripts/` | chạy (`run_*_v4.sh`), cấu hình Triton (`triton_config_v4.sh`), DB (`db/schema.sql`), cấu hình từng Edge (`hosts/edge-*.sh`); kiểm nội dung Edge → DB (`export_run.sh`, `verify_db_export.py`, `review_page.py`), tài nguyên (`resource_log.py`), bộ giải mã (`hw_frame_dump.py`, `hw_stress.py`), nén tensor (`compress_survey.py`, `tail_quant.py`), OCR (`plate_ocr_parity.py`) |
| `yolo11s_contract.py` | tiền / hậu xử lý YOLO11s dùng chung (cách A trên Edge, nửa đầu B trên 247) |
| `tests/` | ~630 test tự động: `python -m pytest tests` (cần thêm `lap`, `grpcio-tools` cho vài test ByteTrack / proto) |

Không có trong repo: file model (`.onnx`, `.dlc`, `.pt`), video thử, kết quả đo — xem danh sách và vị trí trong
[RUNBOOK §4](docs/RUNBOOK.md#4-file-model-và-dữ-liệu-không-có-trong-repo).
