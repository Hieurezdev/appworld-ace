#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export APPWORLD_PROJECT_PATH="${APPWORLD_PROJECT_PATH:-$PROJECT_ROOT}"

INTERVALS=(25 50 75 90)

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

for interval in "${INTERVALS[@]}"; do
  adaptation_config="ACE_lifecycle_add_delete_prune_${interval}_adaptation"
  evaluation_config="ACE_lifecycle_add_delete_prune_${interval}_evaluation"

  echo ">>> [ADD + DELETE, prune every ${interval} tasks] Full adaptation on 90 train tasks"
  appworld run "$adaptation_config"

  echo ">>> [prune ${interval}] Evaluation rollout on test_normal"
  appworld run "$evaluation_config"
  echo ">>> [prune ${interval}] Aggregate test_normal"
  appworld evaluate "$evaluation_config" test_normal

  echo ">>> [prune ${interval}] Evaluation rollout on test_challenge"
  appworld run "$evaluation_config" \
    --override '{"config":{"dataset":"test_challenge","agent":{"max_steps":20}}}'
  echo ">>> [prune ${interval}] Aggregate test_challenge"
  appworld evaluate "$evaluation_config" test_challenge
done

echo ">>> ADD + DELETE prune-interval experiments completed: ${INTERVALS[*]}"
