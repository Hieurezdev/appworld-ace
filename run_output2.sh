

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export APPWORLD_PROJECT_PATH="${APPWORLD_PROJECT_PATH:-$PROJECT_ROOT}"
export PATH="$PROJECT_ROOT/.venv/bin:$PATH"

ADAPTATION_CONFIG="output2_adaptation"
EVALUATION_CONFIG="output2"

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

# echo ">>> [output2] Offline adaptation: adversarial agent + full operations"
# appworld run "$ADAPTATION_CONFIG"

# echo ">>> [output2] Evaluation rollout on test_normal"
# appworld run "$EVALUATION_CONFIG"
# echo ">>> [output2] Aggregate test_normal"
# appworld evaluate "$EVALUATION_CONFIG" test_normal

echo ">>> [output2] Evaluation rollout on test_challenge"
appworld run "$EVALUATION_CONFIG" \
  --override '{"config":{"dataset":"test_challenge","agent":{"max_steps":20}}}'
echo ">>> [output2] Aggregate test_challenge"
appworld evaluate "$EVALUATION_CONFIG" test_challenge

echo ">>> output2 completed: $PROJECT_ROOT/experiments/outputs/output2"
