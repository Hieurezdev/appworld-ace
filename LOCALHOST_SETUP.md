# Localhost Model Server Setup Guide

This guide explains how to run models locally on localhost:8000 using OpenAI API compatible servers, and configure the ACE-AppWorld framework to use them.

## Quick Start

### 1. Start a Local Model Server

You have several options to run a model server compatible with OpenAI API format:

#### Option A: Using vLLM (Recommended)

```bash
# Install vLLM
pip install vllm

# Start the server with Qwen model
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-4B-Instruct-2507 \
    --port 8000 \
    --tensor-parallel-size 1
```

#### Option B: Using Ollama

```bash
# Install Ollama from https://ollama.ai
# Then run:
ollama serve

# In another terminal, pull a model:
ollama pull qwen2:4b

# Access on http://localhost:11434 (you may need to proxy to 8000)
```

#### Option C: Using Text Generation WebUI

```bash
# See: https://github.com/oobabooga/text-generation-webui
# Follow their setup instructions and enable OpenAI API endpoint
```

### 2. Configure ACE-AppWorld

#### Option A: Using JSONNET Config (Recommended)

Edit or create a config file with localhost provider:

```jsonnet
local localhost_model_config = {
    "name": "Qwen/Qwen3-4B-Instruct-2507",
    "provider": "localhost",
    "localhost_url": "http://localhost:8000",
    "localhost_api_key": "not-needed",
    "temperature": 0,
    "max_tokens": 4096,
    "retry_after_n_seconds": 1,
    "use_cache": true,
    "max_retries": 10,
};

{
    "type": "ace",
    "config": {
        "agent": {
            "generator_model_config": localhost_model_config,
            "reflector_model_config": localhost_model_config,
            "curator_model_config": localhost_model_config,
            // ... rest of config
        }
    }
}
```

#### Option B: Using Python Config

```python
from appworld_experiments.code.ace.localhost_utils import get_localhost_config

# Generate config programmatically
model_config = get_localhost_config(
    model_name="Qwen/Qwen3-4B-Instruct-2507",
    localhost_url="http://localhost:8000",
    localhost_api_key="not-needed"
)

# Use in your experiment configuration
config = {
    "generator_model_config": model_config,
    "reflector_model_config": model_config,
    "curator_model_config": model_config,
    # ...
}
```

### 3. Run ACE Experiment

```bash
# Set environment variable if needed
export APPWORLD_PROJECT_PATH=$(pwd)

# Run the experiment with localhost config
python experiments/code/ace/run.py \
    --config_file experiments/configs/ACE_offline_localhost_qwen.jsonnet
```

## Supported Localhost Models

The following models are pre-configured and tested:

| Model | Context | Max Output | Notes |
|-------|---------|-----------|-------|
| `Qwen/Qwen3-4B-Instruct-2507` | 32,768 | 4,096 | **Recommended** - Fast, lightweight |
| `Qwen/Qwen2.5-4B-Instruct` | 32,768 | 4,096 | Alternative Qwen variant |
| `meta-llama/Llama-2-7b-chat-hf` | 4,096 | 4,096 | Requires 16GB+ VRAM |
| `meta-llama/Llama-3-8b-Instruct` | 8,192 | 4,096 | Better quality, needs 20GB+ VRAM |
| `mistralai/Mistral-7B-Instruct-v0.3` | 32,768 | 4,096 | Good balance of quality and speed |

## Environment Variables

You can configure localhost behavior via environment variables:

```bash
# Set custom localhost URL
export LOCALHOST_URL="http://localhost:8000"

# Set custom API key (if your server requires authentication)
export LOCALHOST_API_KEY="your-api-key"
```

## Configuration Options

When setting up `localhost_model_config`:

```python
{
    "name": str,                    # Model ID (e.g., "Qwen/Qwen3-4B-Instruct-2507")
    "provider": "localhost",        # Must be "localhost"
    "localhost_url": str,           # e.g., "http://localhost:8000"
    "localhost_api_key": str,       # API key (can be "not-needed" for local servers)
    "temperature": float,           # 0 for deterministic, 1 for random (0-1)
    "max_tokens": int,             # Maximum output tokens
    "seed": int,                    # Random seed for reproducibility
    "retry_after_n_seconds": int,  # Wait time between retries on failure
    "use_cache": bool,             # Cache completions (true/false)
    "max_retries": int,            # Maximum retry attempts
}
```

## Performance Tips

1. **GPU Memory**: Use `--tensor-parallel-size` in vLLM to distribute across multiple GPUs
2. **Batch Size**: Larger batches are more efficient (vLLM handles this automatically)
3. **Quantization**: Use 4-bit or 8-bit quantization to reduce memory: `--load-in-4bit`, `--load-in-8bit`
4. **Prefix Caching**: Enable for faster repeated prefixes: `--enable-prefix-caching`

Example optimized vLLM command:

```bash
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-4B-Instruct-2507 \
    --port 8000 \
    --tensor-parallel-size 2 \
    --load-in-4bit \
    --enable-prefix-caching \
    --max-model-len 2048
```

## Troubleshooting

### Connection Error: "Connection refused"
- Ensure your model server is running on the correct port
- Check if firewall is blocking localhost:8000
- Verify with: `curl http://localhost:8000/v1/models`

### Model Not Found Error
- Make sure the model name exactly matches what's available on your server
- For vLLM, verify with: `python -m vllm.entrypoints.openai.api_server --help`

### Out of Memory (OOM) Error
- Reduce max_model_len in vLLM
- Use smaller model (e.g., 4B instead of 7B/13B)
- Enable quantization: `--load-in-8bit`

### Slow Responses
- Check server logs for bottlenecks
- Increase batch size if possible
- Consider using a smaller model or quantization

## Integration with Legacy Code

For legacy `openai_language_model.py`:

```python
from appworld_experiments.code.legacy.plain.language_models.openai_language_model import OpenAILanguageModel

# Create model with localhost
model = OpenAILanguageModel(
    model="Qwen/Qwen3-4B-Instruct-2507",
    localhost_url="http://localhost:8000",
    localhost_api_key="not-needed"
)

# Use normally
output = model.generate(messages=[{"role": "user", "content": "Hello"}])
```

## Integration with Bridge (smolagents)

For smolagents bridge:

```python
from appworld_experiments.code.bridge.smolagents.models import OpenAIServerModel

# Note: Use openai_server model type with custom base_url
model_config = {
    "type": "openai_server",
    "base_url": "http://localhost:8000/v1",
    "model": "Qwen/Qwen3-4B-Instruct-2507",
    "api_key": "not-needed"
}
```

## Accessing Different Endpoints

If your model server runs on a different port or URL:

```python
# In JSONNET config
"localhost_url": "http://192.168.1.100:8001"

# Or via environment variable
export LOCALHOST_URL="http://192.168.1.100:8001"
```

## Testing Your Setup

```python
from appworld_experiments.code.ace.localhost_utils import list_supported_models

# List all supported models
list_supported_models()

# Verify connection
import requests
response = requests.get("http://localhost:8000/v1/models")
print(response.json())
```

## Next Steps

- Check `experiments/configs/ACE_offline_localhost_qwen.jsonnet` for a complete example
- See `experiments/code/ace/localhost_utils.py` for utility functions
- Review existing ACE configs for comparison
