# VMS event bridge (plate reads)

`cloud_core/vms_event_publisher.py` forwards validated plate OCR results from Kafka topic `cloud.plate-read.v1` into the Camera Platform VMS ingestion API. It is the supported integration path; do not write VMS PostgreSQL or Kafka `traffic.events.v1` from AI workers.

## Required environment

| Variable | Purpose |
| --- | --- |
| `KAFKA_BOOTSTRAP_SERVERS` | AI cluster Kafka |
| `VMS_INGESTION_ORIGIN` | HTTPS gateway base (e.g. `https://nv9110.local`) |
| `VMS_INGESTION_CA_FILE` | Trust anchor for gateway TLS |
| `VMS_AI_CLIENT_CERT_FILE` / `VMS_AI_CLIENT_KEY_FILE` | mTLS identity registered in `device_registry.ai_service_camera_grants` |
| `VMS_AI_EDGE_ID` | Must match `node_id` in plate-read messages and VMS `edge_id` |
| `VMS_AI_CAMERA_ID` | Scoped camera grant |
| `V4_MEDIA_DIR` | Root containing plate crop JPEGs referenced by `image_path` |

## Run

```bash
python -m cloud_core.vms_event_publisher \
  --bootstrap "$KAFKA_BOOTSTRAP_SERVERS" \
  --vms-origin "$VMS_INGESTION_ORIGIN" \
  --vms-ca "$VMS_INGESTION_CA_FILE" \
  --vms-cert "$VMS_AI_CLIENT_CERT_FILE" \
  --vms-key "$VMS_AI_CLIENT_KEY_FILE" \
  --edge-id "$VMS_AI_EDGE_ID" \
  --camera-id "$VMS_AI_CAMERA_ID" \
  --media-dir "$V4_MEDIA_DIR"
```

Only messages with `valid=true`, VN plate pattern, and matching edge/camera IDs are submitted. Confidence in the VMS manifest comes from `det_conf` on the Kafka message.

See the Camera Platform handoff: `docs/deployment/AI_VMS_INTEGRATION_RUNBOOK.md`.