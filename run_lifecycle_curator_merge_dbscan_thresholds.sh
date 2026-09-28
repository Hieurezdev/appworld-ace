#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export APPWORLD_PROJECT_PATH="${APPWORLD_PROJECT_PATH:-$PROJECT_ROOT}"

THRESHOLDS=(0.70 0.75 0.80 0.85 0.90 0.95)
DBSCAN_EPS=(0.30 0.25 0.20 0.15 0.10 0.05)

wait_for_service() {
  local url="$1"
  local label="$2"
  local start_command="$3"
  local attempts=24

  for ((attempt = 1; attempt <= attempts; attempt++)); do
    if curl --fail --silent --show-error --max-time 5 "$url" >/dev/null; then
      return 0
    fi
    sleep 5
  done

  echo "${label} is unavailable at ${url} after 120 seconds" >&2
  echo "Start it in another terminal: ${start_command}" >&2
  exit 1
}

wait_for_service http://127.0.0.1:5000/v1/models \
  "Local model server" \
  "vllm serve Qwen/Qwen3-4B-Instruct-2507 --host 0.0.0.0 --port 5000"
wait_for_service http://127.0.0.1:8000/ \
  "AppWorld environment server" \
  "appworld serve environment --port 8000"
wait_for_service http://127.0.0.1:9000/docs \
  "AppWorld APIs server" \
  "appworld serve apis --port 9000"

for index in "${!THRESHOLDS[@]}"; do
  threshold="${THRESHOLDS[$index]}"
  eps="${DBSCAN_EPS[$index]}"
  label="${threshold#0.}"
  adaptation_config="ACE_lifecycle_curator_merge_dbscan_${label}_adaptation"
  evaluation_config="ACE_lifecycle_curator_merge_dbscan_${label}_evaluation"

  echo ">>> [Curator MERGE] Full adaptation: cosine similarity >= ${threshold} (DBSCAN eps=${eps})"
  appworld run "$adaptation_config"

  echo ">>> [threshold ${threshold}] Evaluation rollout on test_normal"
  appworld run "$evaluation_config"
  echo ">>> [threshold ${threshold}] Aggregate test_normal"
  appworld evaluate "$evaluation_config" test_normal

  echo ">>> [threshold ${threshold}] Evaluation rollout on test_challenge"
  appworld run "$evaluation_config" \
    --override '{"config":{"dataset":"test_challenge","agent":{"max_steps":20}}}'
  echo ">>> [threshold ${threshold}] Aggregate test_challenge"
  appworld evaluate "$evaluation_config" test_challenge
done

echo ">>> Curator MERGE DBSCAN threshold sweep completed: ${THRESHOLDS[*]}"
