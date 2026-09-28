
#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export APPWORLD_PROJECT_PATH="${APPWORLD_PROJECT_PATH:-$PROJECT_ROOT}"

TOP_K_VALUES=(5 10 15 20 25 30)

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

reset_fmb() {
  local top_k="$1"
  local fmb_file="$PROJECT_ROOT/experiments/playbooks/failure_memory_bank_lifecycle_fmb_topk_${top_k}.jsonl"
  mkdir -p "$(dirname "$fmb_file")"
  : > "$fmb_file"
  echo ">>> Reset FMB storage: $fmb_file"
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

for top_k in "${TOP_K_VALUES[@]}"; do
  adaptation_config="ACE_lifecycle_fmb_topk_${top_k}_adaptation"
  normal_config="ACE_lifecycle_fmb_topk_${top_k}_normal_evaluation"
  challenge_config="ACE_lifecycle_fmb_topk_${top_k}_challenge_evaluation"

  reset_fmb "$top_k"

  echo ">>> [FMB verified, top_k=${top_k}] Full adaptation on train"
  appworld run "$adaptation_config"

  echo ">>> [FMB top_k=${top_k}] Evaluation rollout on test_normal"
  appworld run "$normal_config"
  echo ">>> [FMB top_k=${top_k}] Aggregate test_normal"
  appworld evaluate "$normal_config" test_normal

  echo ">>> [FMB top_k=${top_k}] Evaluation rollout on test_challenge"
  appworld run "$challenge_config"
  echo ">>> [FMB top_k=${top_k}] Aggregate test_challenge"
  appworld evaluate "$challenge_config" test_challenge
done

echo ">>> FMB top-k sweep completed: ${TOP_K_VALUES[*]}"
