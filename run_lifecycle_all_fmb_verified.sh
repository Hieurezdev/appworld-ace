#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export APPWORLD_PROJECT_PATH="${APPWORLD_PROJECT_PATH:-$PROJECT_ROOT}"

# Effective lifecycle operations:
#   ADD + UPDATE + DELETE + MERGE + CREATE_META
# Failure memory:
#   reflector_memory_mode=verified
#   failure_memory_bank_FMB_curator_operations_v2.jsonl
ADAPTATION_CONFIG="ACE_offline_with_GT_curator_operations_FMB_improved"
NORMAL_EVALUATION_CONFIG="ACE_offline_with_GT_curator_operations_FMB_improved_evaluation"
CHALLENGE_EVALUATION_CONFIG="ACE_offline_with_GT_curator_operations_FMB_improved_evaluation_challenge"

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

wait_for_service http://0.0.0.0:8000/ \
  "AppWorld environment server" \
  "appworld serve environment --port 8000"
wait_for_service http://0.0.0.0:9000/docs \
  "AppWorld APIs server" \
  "appworld serve apis --port 9000"

echo ">>> [full + FMB verified] Offline adaptation on train"
appworld run "$ADAPTATION_CONFIG"

echo ">>> [full + FMB verified] Evaluation rollout on test_normal"
appworld run "$NORMAL_EVALUATION_CONFIG"
echo ">>> [full + FMB verified] Aggregate test_normal"
appworld evaluate "$NORMAL_EVALUATION_CONFIG" test_normal

echo ">>> [full + FMB verified] Evaluation rollout on test_challenge"
appworld run "$CHALLENGE_EVALUATION_CONFIG"
echo ">>> [full + FMB verified] Aggregate test_challenge"
appworld evaluate "$CHALLENGE_EVALUATION_CONFIG" test_challenge

echo ">>> Full lifecycle + verified FMB experiment completed."
