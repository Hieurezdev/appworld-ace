#!/usr/bin/env bash
# Exit on error
set -e

# Get absolute path of this script's directory and switch there
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Export project path as the script directory
export APPWORLD_PROJECT_PATH="$(pwd)"

# vLLM listen port (keep consistent with readiness check)
VLLM_PORT=5000

# Start appworld services and capture their PIDs for cleanup
uv run appworld serve apis &
APIS_PID=$!
uv run appworld serve environment &
ENV_PID=$!

echo "🚀 Starting server..."
uv run vllm serve "Qwen/Qwen3-4B-Instruct-2507" \
    --dtype auto \
    --gpu-memory-utilization 0.85 \
    --max-model-len 262144 \
    --host 0.0.0.0 \
    --port 5000 \
    --kv-cache-dtype fp8 \
    --enable-chunked-prefill \
    --max-num-batched-tokens 4096 \
    --trust-remote-code &
VLLM_PID=$!

# Ensure background processes are cleaned up on exit
cleanup() {
    echo "Cleaning up background processes..."
    kill -9 ${VLLM_PID} 2>/dev/null || true
    kill -9 ${APIS_PID} 2>/dev/null || true
    kill -9 ${ENV_PID} 2>/dev/null || true
}
trap cleanup EXIT

# Optimized polling interval and process liveness check
MAX_RETRIES=120
RETRY_DELAY=5

echo "⏳ Waiting for vLLM server to be ready on port ${VLLM_PORT}..."
for ((i=1;i<=MAX_RETRIES;i++)); do
    # Check if the vLLM process has died
    if ! kill -0 ${VLLM_PID} 2>/dev/null; then
        echo "❌  vLLM server process died!"
        exit 1
    fi

    if curl -s "http://localhost:${VLLM_PORT}/v1/models" > /dev/null; then
        echo "✅  server is ready!"
        break
    fi

    echo "⏳ Waiting... ($((i*RETRY_DELAY))s/$((MAX_RETRIES*RETRY_DELAY))s)"
    sleep ${RETRY_DELAY}
done

if [ ${i} -gt ${MAX_RETRIES} ]; then
    echo "❌  server failed to start in time"
    kill -9 ${VLLM_PID} 2>/dev/null || true
    exit 1
fi

# Run experiments (these run in foreground as before)
uv run appworld run ACE_offline_with_GT_adaptation_casebank_FMB_adversarial
uv run appworld run ACE_offline_with_GT__casebank_FMB_adversarial_evaluation
uv run appworld evaluate ACE_offline_with_GT_casebank_FMB_adversarial_evaluation test_normal
uv run appworld run ACE_offline_with_GT_casebank_FMB_adversarial_evaluation_challenge
uv run appworld evaluate ACE_offline_with_GT_casebank_FMB_adversarial_evaluation_challenge test_challenge