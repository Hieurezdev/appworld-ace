# Exit on error
set -e
export APPWORLD_PROJECT_PATH="$(pwd)"
# Get absolute path of this script's directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Ensure we start in the project directory
cd "$SCRIPT_DIR"

echo "🚀 Running origin evaluations..."
uv run appworld run ACE_offline_with_GT__RAE_FMB_adversarial_evaluation_origin --sample-size 20
# Fixed configuration name typo (added _origin)
uv run appworld run ACE_offline_with_GT__RAE_FMB_adversarial_evaluation_origin_challenge --sample-size 10

# Dynamically kill the process on port 62726 (instead of hardcoded PID 271485)
echo "🧹 Stopping existing server on port 62726..."
python3 -c "import os, glob; [os.kill(int(p), 9) for f in ['/proc/net/tcp', '/proc/net/tcp6'] if os.path.exists(f) for l in open(f).readlines()[1:] for parts in [l.split()] if parts and parts[1].endswith(':F506') for inode in [parts[9]] for fd in glob.glob('/proc/[0-9]*/fd/[0-9]*') if f'socket:[{inode}]' in os.readlink(fd) for p in [fd.split('/')[2]]]" 2>/dev/null || true

# Robust path navigation using SCRIPT_DIR
cd "$SCRIPT_DIR/../TokenSelectExperiment"

echo "🚀 Starting TokenSelect server..."
uv run python benchmark/serve.py --model-path Qwen/Qwen2.5-7B-Instruct --dp 1 --disable-cuda-graph --port 62726 --mem-fraction-static 0.85 --context-length 1048576 --chunked-prefill-size 8192 --max-prefill-tokens 1048576 --sgl-conf-file config/qwen-token-retrieval.yaml &

SERVER_PID=$!

# Optimized polling interval and process liveness check
MAX_RETRIES=120
RETRY_DELAY=5

echo "⏳ Waiting for vLLM server to be ready on port 62726..."
for ((i=1;i<=MAX_RETRIES;i++)); do
    # Check if the server process has died
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo "❌ TokenSelect server process died!"
        exit 1
    fi

    if curl -s http://localhost:62726/v1/models > /dev/null; then
        echo "✅ TokenSelect server is ready!"
        break
    fi

    echo "⏳ Waiting... ($((i*RETRY_DELAY))s/$((MAX_RETRIES*RETRY_DELAY))s)"
    sleep $RETRY_DELAY
done

if [ $i -gt $MAX_RETRIES ]; then
    echo "❌ TokenSelect server failed to start in time"
    kill -9 $SERVER_PID 2>/dev/null || true
    exit 1
fi

# Navigate back to the appworld-ace directory before running evaluations
cd "$SCRIPT_DIR"

echo "🚀 Running TokenSelect evaluations..."
uv run appworld run ACE_offline_with_GT__RAE_FMB_adversarial_evaluation_tksl --sample-size 20
uv run appworld run ACE_offline_with_GT__RAE_FMB_adversarial_evaluation_tksl_challenge --sample-size 10
