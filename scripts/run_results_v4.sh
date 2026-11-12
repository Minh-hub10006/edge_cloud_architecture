#!/usr/bin/env bash
# v4 results store on 249 (6/11): TimescaleDB container + results writer (the check page was removed 13/11).
#   bash scripts/run_results_v4.sh db       # first time: creates the password file and the database, applies the schema
#   bash scripts/run_results_v4.sh start    # results writer (DB port 25432)
#   bash scripts/run_results_v4.sh stop     # stops writer + page; the database keeps running
# Python: env qnn + ~/v4_dbdeps (psycopg only, kept out of qnn on purpose). The password stays in ~/.v4_db.env (mode 600).
set -euo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"
envfile="$HOME/.v4_db.env"
py="${CLOUD_PY:-$HOME/miniconda3/envs/qnn/bin/python}"
hosts_default='{"edge-246":"192.168.180.246","edge-252":"192.168.180.252","edge-247":"192.168.180.247"}'
case "${1:?db|start|stop}" in
  db)
    if [[ ! -f "$envfile" ]]; then
      (umask 077; echo "POSTGRES_PASSWORD=$(openssl rand -hex 16)" > "$envfile")
    fi
    if ! docker ps -a --format '{{.Names}}' | grep -qx v4-tsdb; then
      # memory and CPU are capped: 249 is shared with other groups
      docker run -d --name v4-tsdb --restart unless-stopped --env-file "$envfile" -e POSTGRES_DB=v4 \
        -e TS_TUNE_MEMORY=8GB -e TS_TUNE_NUM_CPUS=4 -p 25432:5432 -v v4-tsdb-data:/var/lib/postgresql/data \
        timescale/timescaledb:2.17.2-pg16 >/dev/null
    fi
    for _ in $(seq 1 60); do
      docker exec v4-tsdb pg_isready -U postgres -d v4 -q && break
      sleep 2
    done
    sleep 3  # the image restarts Postgres once after its first init
    docker exec v4-tsdb pg_isready -U postgres -d v4 -q || { echo "database not ready" >&2; exit 1; }
    docker exec -i v4-tsdb psql -q -U postgres -d v4 -v ON_ERROR_STOP=1 < "$here/scripts/db/schema.sql" >/dev/null
    docker exec v4-tsdb psql -U postgres -d v4 -Atc "SELECT 'bảng theo thời gian: ' || string_agg(hypertable_name, ', ') FROM timescaledb_information.hypertables"
    ;;
  start)
    pgrep -f "v4-results-" >/dev/null && { echo "already running; stop first" >&2; exit 2; }
    [[ -f "$envfile" ]] || { echo "run: $0 db" >&2; exit 2; }
    # shellcheck disable=SC1090
    source "$envfile"
    out="$HOME/v4_runs/results_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$out"
    # 8/11: the writer subscribes at start and only sees a topic created later after a metadata refresh (~5 min), so
    # every topic it reads is created here first (same settings as scripts/run_cloud_v4.sh)
    for t in controlv1.cloud.detection-result.v1 cloud.detection-result.v1 edge.tracked-result.v1 edge.count.v1              edge.event.v1 cloud.plate-read.v1 cloud.crowd.v1 cloud.action.v1; do
      docker exec v4-kafka /opt/kafka/bin/kafka-topics.sh --bootstrap-server localhost:29092 --create --if-not-exists         --topic "$t" --partitions 4 --config retention.ms=60000 --config max.message.bytes=2097152 2>&1 | grep -v WARNING || true
    done
    export DB_DSN="postgresql://postgres:${POSTGRES_PASSWORD}@localhost:25432/v4" KAFKA_BOOTSTRAP_SERVERS=localhost:29092 \
      PYTHONPATH="$here:$HOME/v4_dbdeps" OMP_NUM_THREADS=1 DB_FLUSH_S=0.2 NODE_HOSTS="${NODE_HOSTS:-$hosts_default}"
    nohup bash -c 'exec -a v4-results-writer "$0" -B -m cloud_core.results_writer' "$py" > "$out/writer.log" 2>&1 &
    sleep 6
    tail -2 "$out/writer.log" | cut -c1-160
    pgrep -af "v4-results-" | cut -c1-70
    echo "logs: $out"
    ;;
  stop)
    pkill -f "v4-results-" && echo "stopped" || echo "not running"
    ;;
  *) echo "db|start|stop" >&2; exit 2 ;;
esac
