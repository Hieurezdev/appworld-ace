#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export APPWORLD_PROJECT_PATH="${APPWORLD_PROJECT_PATH:-$PROJECT_ROOT}"

ALL_PAIRS=(update_delete update_merge merge_delete)

usage() {
  cat <<'EOF'
Usage:
  ./run_lifecycle_operation_pairs.sh             # run all operation pairs
  ./run_lifecycle_operation_pairs.sh PAIR [...]  # run selected pairs

Available pairs:
  update_delete  = ADD + UPDATE + DELETE
  update_merge   = ADD + UPDATE + MERGE
  merge_delete   = ADD + MERGE + DELETE

Each pair runs adaptation on train, followed by test_normal and test_challenge.
EOF
}

is_valid_pair() {
  local requested="$1"
  local candidate
  for candidate in "${ALL_PAIRS[@]}"; do
    if [[ "$candidate" == "$requested" ]]; then
      return 0
    fi
  done
  return 1
}

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

run_pair() {
  local pair="$1"
  local adaptation_config="ACE_lifecycle_${pair}_adaptation"
  local evaluation_config="ACE_lifecycle_${pair}_evaluation"

  echo ">>> [${pair}] Offline adaptation on train"
  appworld run "$adaptation_config"

  echo ">>> [${pair}] Evaluation rollout on test_normal"
  appworld run "$evaluation_config"
  echo ">>> [${pair}] Aggregate test_normal"
  appworld evaluate "$evaluation_config" test_normal

  echo ">>> [${pair}] Evaluation rollout on test_challenge"
  appworld run "$evaluation_config" \
    --override '{"config":{"dataset":"test_challenge","agent":{"max_steps":20}}}'
  echo ">>> [${pair}] Aggregate test_challenge"
  appworld evaluate "$evaluation_config" test_challenge
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi

if (($# == 0)); then
  PAIRS=("${ALL_PAIRS[@]}")
else
  PAIRS=("$@")
fi

for pair in "${PAIRS[@]}"; do
  if ! is_valid_pair "$pair"; then
    echo "Unknown operation pair: $pair" >&2
    usage >&2
    exit 2
  fi
done

wait_for_service http://0.0.0.0:8000/ \
  "AppWorld environment server" \
  "appworld serve environment --port 8000"
wait_for_service http://0.0.0.0:9000/docs \
  "AppWorld APIs server" \
  "appworld serve apis --port 9000"

for pair in "${PAIRS[@]}"; do
  run_pair "$pair"
done

echo ">>> Lifecycle operation-pair ablations completed."
