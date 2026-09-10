#!/usr/bin/env bash
# Brings up the local observability stack for AgentIQ and points the API at it.
#
# Prereqs (Homebrew -- Docker Desktop's Rosetta install fails on this arm64
# machine, so local verification used native services):
#   brew install prometheus grafana redis
#   brew services start redis
#   createdb agentiq_checkpoints
#
# Then:
#   ./scripts/run_observability_stack.sh
#   # in another shell:
#   CHECKPOINT_DSN=postgresql://localhost/agentiq_checkpoints \
#   REDIS_URL=redis://localhost:6379/0 \
#   uvicorn api.main:app --host 127.0.0.1 --port 8000
#
# Grafana: http://localhost:3000  (admin/admin, dashboard "AgentIQ")
# Prometheus: http://localhost:9090
# App metrics: http://localhost:8000/metrics
set -euo pipefail
cd "$(dirname "$0")/.."

PROM_DATA="${PROM_DATA:-/tmp/agentiq-prom-data}"
mkdir -p "$PROM_DATA"

echo "starting prometheus (scraping localhost:8000/metrics)..."
prometheus \
  --config.file=observability/prometheus.yml \
  --storage.tsdb.path="$PROM_DATA" \
  --web.listen-address=127.0.0.1:9090 &
PROM_PID=$!

echo "starting grafana..."
brew services start grafana >/dev/null

# Wait for Grafana, then provision the Prometheus datasource + dashboard
# via its HTTP API (the file-provisioning paths differ per install).
until curl -sf http://127.0.0.1:3000/api/health >/dev/null; do sleep 1; done

curl -sf -u admin:admin -X POST http://127.0.0.1:3000/api/datasources \
  -H 'Content-Type: application/json' \
  -d '{"name":"Prometheus","type":"prometheus","access":"proxy","url":"http://127.0.0.1:9090","isDefault":true}' \
  >/dev/null 2>&1 || echo "  (datasource already exists)"

DS_UID=$(curl -sf -u admin:admin http://127.0.0.1:3000/api/datasources/name/Prometheus | python3 -c 'import json,sys;print(json.load(sys.stdin)["uid"])')
python3 - "$DS_UID" <<'PY'
import json, sys, urllib.request, base64
ds = sys.argv[1]
d = json.load(open("observability/grafana/dashboards/agentiq.json"))
d = json.loads(json.dumps(d).replace("${DS_PROMETHEUS}", ds))
d.pop("templating", None); d.pop("__inputs", None)
body = json.dumps({"dashboard": d, "overwrite": True}).encode()
req = urllib.request.Request(
    "http://127.0.0.1:3000/api/dashboards/db", data=body,
    headers={"Content-Type": "application/json",
             "Authorization": "Basic " + base64.b64encode(b"admin:admin").decode()})
print("dashboard:", json.load(urllib.request.urlopen(req))["url"])
PY

echo
echo "Grafana:    http://localhost:3000/d/agentiq-main/agentiq  (admin/admin)"
echo "Prometheus: http://localhost:9090"
echo "Ctrl-C to stop prometheus (grafana keeps running via brew services)."
wait $PROM_PID
