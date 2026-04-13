"""Utility functions for localhost model server with OpenAI API format."""

import os
from typing import Optional


def configure_localhost(
    localhost_url: str = "http://localhost:8000",
    localhost_api_key: str = "not-needed",
) -> None:
    """Configure environment variables for localhost model server.
    
    Args:
        localhost_url: Base URL of the localhost model server (default: http://localhost:8000)
        localhost_api_key: API key for localhost server (default: not-needed)
    
    Example:
        >>> configure_localhost("http://localhost:8000", "your-api-key")
    """
    os.environ["LOCALHOST_URL"] = localhost_url
    os.environ["LOCALHOST_API_KEY"] = localhost_api_key


def get_localhost_config(
    model_name: str = "Qwen/Qwen3-4B-Instruct-2507",
    localhost_url: str = "http://localhost:8000",
    localhost_api_key: str = "not-needed",
    temperature: float = 0,
    max_tokens: int = 4096,
    retry_after_n_seconds: int = 1,
    use_cache: bool = True,
    max_retries: int = 10,
) -> dict:
    """Generate configuration dictionary for localhost model provider.
    
    Args:
        model_name: Model identifier (default: Qwen/Qwen3-4B-Instruct-2507)
        localhost_url: Base URL of localhost model server
        localhost_api_key: API key for localhost server
        temperature: Sampling temperature (0-1, default 0 for deterministic)
        max_tokens: Maximum tokens to generate
        retry_after_n_seconds: Seconds to wait between retries
        use_cache: Whether to cache completions
        max_retries: Maximum number of retries
    
    Returns:
        Configuration dict for use in experiment configs
        
    Example:
        >>> config = get_localhost_config(
        ...     model_name="Qwen/Qwen3-4B-Instruct-2507",
        ...     localhost_url="http://localhost:8000"
        ... )
    """
    return {
        "name": model_name,
        "provider": "localhost",
        "localhost_url": localhost_url,
        "localhost_api_key": localhost_api_key,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "seed": 100,
        "stop": ["<|endoftext|>", "<|eot_id|>", "<|start_header_id|>"],
        "logprobs": False,
        "top_logprobs": None,
        "frequency_penalty": 0,
        "presence_penalty": 0,
        "n": 1,
        "response_format": {"type": "text"},
        "retry_after_n_seconds": retry_after_n_seconds,
        "use_cache": use_cache,
        "max_retries": max_retries,
    }


def create_localhost_config_jsonnet(
    model_name: str = "Qwen/Qwen3-4B-Instruct-2507",
    localhost_url: str = "http://localhost:8000",
    localhost_api_key: str = "not-needed",
) -> str:
    """Generate JSONNET configuration snippet for localhost provider.
    
    Args:
        model_name: Model identifier
        localhost_url: Base URL of localhost model server
        localhost_api_key: API key for localhost server
    
    Returns:
        JSONNET snippet string
        
    Example:
        >>> snippet = create_localhost_config_jsonnet()
        >>> print(snippet)
    """
    return f'''local localhost_model_config = {{
    "name": "{model_name}",
    "provider": "localhost",
    "localhost_url": "{localhost_url}",
    "localhost_api_key": "{localhost_api_key}",
    "temperature": 0,
    "seed": 100,
    "stop": ["<|endoftext|>", "<|eot_id|>", "<|start_header_id|>"],
    "logprobs": false,
    "top_logprobs": null,
    "frequency_penalty": 0,
    "presence_penalty": 0,
    "n": 1,
    "response_format": {{"type": "text"}},
    "retry_after_n_seconds": 1,
    "use_cache": true,
    "max_retries": 10,
}};'''


# Supported localhost models
SUPPORTED_LOCALHOST_MODELS = {
    "Qwen/Qwen3-4B-Instruct-2507": {
        "context_length": 32768,
        "max_output_tokens": 4096,
        "description": "Qwen 3 4B Instruct model (recommended for lightweight deployment)",
    },
    "Qwen/Qwen2.5-4B-Instruct": {
        "context_length": 32768,
        "max_output_tokens": 4096,
        "description": "Qwen 2.5 4B Instruct model",
    },
    "Qwen/Qwen2-4B-Instruct": {
        "context_length": 32768,
        "max_output_tokens": 4096,
        "description": "Qwen 2 4B Instruct model",
    },
    "meta-llama/Llama-2-7b-chat-hf": {
        "context_length": 4096,
        "max_output_tokens": 4096,
        "description": "Meta Llama 2 7B Chat model",
    },
    "meta-llama/Llama-3-8b-Instruct": {
        "context_length": 8192,
        "max_output_tokens": 4096,
        "description": "Meta Llama 3 8B Instruct model",
    },
    "mistralai/Mistral-7B-Instruct-v0.3": {
        "context_length": 32768,
        "max_output_tokens": 4096,
        "description": "Mistral 7B Instruct v0.3 model",
    },
}


def list_supported_models() -> None:
    """Print list of supported localhost models."""
    print("\n" + "="*80)
    print("Supported localhost Models for OpenAI API Compatible Server")
    print("="*80)
    for model_name, info in SUPPORTED_LOCALHOST_MODELS.items():
        print(f"\n  Model: {model_name}")
        print(f"    Description: {info['description']}")
        print(f"    Context Length: {info['context_length']:,} tokens")
        print(f"    Max Output: {info['max_output_tokens']:,} tokens")
    print("\n" + "="*80 + "\n")


def get_default_localhost_model() -> str:
    """Get the default recommended localhost model."""
    return "Qwen/Qwen3-4B-Instruct-2507"


if __name__ == "__main__":
    # Example usage
    list_supported_models()
    
    # Generate config example
    config = get_localhost_config()
    print("\nExample configuration dict:")
    import json
    print(json.dumps(config, indent=2))
    
    # Generate JSONNET example
    print("\nExample JSONNET configuration:")
    print(create_localhost_config_jsonnet())
