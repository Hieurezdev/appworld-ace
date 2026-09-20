

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export APPWORLD_PROJECT_PATH="${APPWORLD_PROJECT_PATH:-$PROJECT_ROOT}"

# Effective adaptation:
#   ADD + UPDATE + DELETE + MERGE + CREATE_META
#   periodic unused-bullet pruning + DBSCAN merge candidates
#   improved adversarial agent with attack and outcome verification
# RAE/FMB are intentionally disabled in this ablation.
ADAPTATION_CONFIG="ACE_lifecycle_all_adversarial_verified_adaptation"
EVALUATION_CONFIG="ACE_lifecycle_all_adversarial_verified_evaluation"

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

echo ">>> [full + adversarial verified] Offline adaptation on train"
appworld run "$ADAPTATION_CONFIG"

echo ">>> [full + adversarial verified] Evaluation rollout on test_normal"
appworld run "$EVALUATION_CONFIG"
echo ">>> [full + adversarial verified] Aggregate test_normal"
appworld evaluate "$EVALUATION_CONFIG" test_normal

echo ">>> [full + adversarial verified] Evaluation rollout on test_challenge"
appworld run "$EVALUATION_CONFIG" \
  --override '{"config":{"dataset":"test_challenge","agent":{"max_steps":20}}}'
echo ">>> [full + adversarial verified] Aggregate test_challenge"
appworld evaluate "$EVALUATION_CONFIG" test_challenge

echo ">>> Full lifecycle + verified adversarial experiment completed."
