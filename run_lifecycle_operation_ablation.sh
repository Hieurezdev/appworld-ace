#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export APPWORLD_PROJECT_PATH="${APPWORLD_PROJECT_PATH:-$PROJECT_ROOT}"

OPERATIONS=(update delete_prune merge lifecycle_all)

for operation in "${OPERATIONS[@]}"; do
  if [[ "$operation" == "lifecycle_all" ]]; then
    adaptation_config="ACE_lifecycle_all_adaptation"
    evaluation_config="ACE_lifecycle_all_evaluation"
  else
    adaptation_config="ACE_lifecycle_${operation}_adaptation"
    evaluation_config="ACE_lifecycle_${operation}_evaluation"
  fi

  echo ">>> [${operation}] Offline adaptation on train"
  appworld run "$adaptation_config"

  echo ">>> [${operation}] Evaluation rollout on test_normal"
  appworld run "$evaluation_config"
  echo ">>> [${operation}] Aggregate test_normal"
  appworld evaluate "$evaluation_config" test_normal

  echo ">>> [${operation}] Evaluation rollout on test_challenge"
  appworld run "$evaluation_config" \
    --override '{"config":{"dataset":"test_challenge","agent":{"max_steps":20}}}'
  echo ">>> [${operation}] Aggregate test_challenge"
  appworld evaluate "$evaluation_config" test_challenge
done

echo ">>> Lifecycle-operation ablation completed."
