# Các luồng

## 1. Một khung đi qua làn nhanh

```
giải mã (Edge) ─► khung 640 + (mỗi 5 khung) ảnh lớn ─► bộ đệm khung 3 s (làn sau)
      │
      ├─ N / A: model trên Edge ─► ép đúng tỉ lệ ─► ByteTrack trên Edge ─► box + mã bám
      │                                                   └─► Kafka edge.tracked-result.v1 (chạy nền, không chờ) ─► DB
      │
      └─ B / C / D: vé suy luận (tensor hoặc JPEG) ─► Kafka ─► worker Cloud (cùng camera → cùng worker)
                         ─► Triton (gom lô) ─► ByteTrack trên Cloud ─► kết quả ─► Kafka (chung + riêng Edge)
                         ─► Edge nhận ─► box + mã bám ─► BUS làn sau          └─► DB (frames, detections)
```

- Deadline mỗi khung: quá hạn ở Edge hay Cloud thì bỏ, không xử lý muộn.
- Thứ tự: khung cũ hơn khung đã thấy bị từ chối (`out_of_order`) để ByteTrack không lùi thời gian.

## 2. Làn sau

```
BUS (mỗi việc 1 hàng đợi riêng, đầy thì bỏ tin cũ) ─► luật trên box + mã bám
   ├─ số đếm (5 s)                        ─► Kafka (chạy nền) ─► DB counts
   ├─ sự kiện (bắt đầu/kết thúc + ảnh)     ─► hộp thư (ghi file) ─► Kafka (chờ xác nhận) ─► DB events
   ├─ ảnh biển (UC1)                       ─► Kafka ─► OCR ─► DB plates
   ├─ ảnh đông người (UC3)                 ─► Kafka ─► DM-Count ─► DB crowd (+ sự kiện "crowd")
   └─ clip / ảnh cắt UC4                   ─► Kafka ─► RTMO-s (+ ST-GCN++) ─► DB actions (+ sự kiện "fight"/"crash")
```

Phản hồi giãn nhịp: worker làn sau đo độ trễ tin → `cloud.slow-feedback.v1` → Edge nhân khoảng cách gửi (ảnh đông
người, clip, biển số).

## 3. Video cho web

Luồng chính của camera được chép nguyên (không mã hoá lại) lên MediaMTX của chính Edge; trình duyệt xem qua WebRTC (WHEP).
Box lấy từ DB và ghép đúng khung nhờ số khung in trong dải nhận dạng của video thử (`web/live_sync.js`).
Camera thật không có dải này: web sản phẩm cần cách ghép khác.

## 4. Phản hồi điều khiển

Cloud gửi trạng thái tải mỗi ~1 s (`controlv1.cloud.feedback.v1`): số vé đang chờ, áp lực hàng đợi. Agent mỗi camera
ghi trạng thái mỗi cửa sổ 1 s (`--state-jsonl`) — đầu vào cho DRL sau này.

## 5. Xử lý sự cố (đã kiểm trên máy thật)

| Sự cố | Hành vi |
|---|---|
| **Mất Kafka** (đã thử 40–55 s) | Edge không chờ Kafka trên đường xử lý khung (gửi nền, tối đa 64 tin, quá thì bỏ và đếm `kafka_background_dropped`); cách N/A chạy tiếp ~100 %; B/C/D mất đúng khoảng mất Kafka rồi chạy lại. Bộ gửi tin tự tạo lại khi broker quên số thứ tự (`OutOfOrderSequenceNumber`). Vòng đọc tin trên Cloud tự khởi động lại từ điểm đã xác nhận. Sự kiện trong lúc mất được gửi bù đủ. |
| **Tắt worker làn sau** (60 s) | Làn nhanh không ảnh hưởng; tin tồn quá hạn bị bỏ khi worker lên lại. |
| **Triton khởi động lại** | Worker tự đăng ký lại vùng nhớ dùng chung, gửi lại 1 lần. |
| **Bộ giải mã phần cứng đứng** (252, `qtic2vdec` lỗi -5) | Đọc khung chờ tối đa 10 s, ghi log `[HwCapture] no frame...`, khởi động lại bộ giải mã tại khung đang dở. |
| **Kiểm thử bộ giải mã treo lúc khởi động** | Dừng nhẹ (SIGINT), thử lại 1 lần sau 3 s. |
| **Hết CMA** (246, `vb2_streamon failed, -12`) | Không tự hồi phục được: khởi động lại Edge (xem RUNBOOK). |
