import hashlib
import inspect
import json
import os
import time
import uuid
from typing import Any, Literal

import litellm
from joblib import Memory
from litellm import completion_cost, token_counter
from openai import (
    APIConnectionError,
    APIError,
    APIResponseValidationError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    ConflictError,
    InternalServerError,
    NotFoundError,
    OpenAI,
    OpenAIError,
    PermissionDeniedError,
    RateLimitError,
    UnprocessableEntityError,
)
from rich.panel import Panel

from appworld import AppWorld
from appworld.common.path_store import path_store
from appworld.common.utils import rprint, write_jsonl

litellm.drop_params = True
cache = Memory(os.path.join(path_store.cache, "llm_calls"), verbose=0)


class LocalhostResponseCache:
    """Cache responses from port 5000 vLLM server to avoid redundant processing on timeout."""
    
    def __init__(self, max_cache_size: int = 100):
        self.cache = {}
        self.max_cache_size = max_cache_size
    
    @staticmethod
    def _get_cache_key(messages: list[dict[str, str]], **kwargs) -> str:
        """Generate a deterministic cache key from request parameters."""
        # Create consistent key from messages and kwargs
        messages_str = json.dumps(messages, sort_keys=True)
        # Filter out non-serializable and non-deterministic kwargs
        clean_kwargs = {k: v for k, v in kwargs.items() 
                       if k not in ['use_localhost_cache', 'localhost_timeout', 'retry_after_n_seconds']}
        params_str = json.dumps(clean_kwargs, sort_keys=True, default=str)
        combined = f"{messages_str}:{params_str}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    def get(self, messages: list[dict[str, str]], **kwargs) -> dict | None:
        """Retrieve cached response if available."""
        key = self._get_cache_key(messages, **kwargs)
        return self.cache.get(key)
    
    def set(self, response: dict, messages: list[dict[str, str]], **kwargs) -> None:
        """Store response in cache, evicting oldest entry if cache is full."""
        if len(self.cache) >= self.max_cache_size:
            # Remove oldest entry (first key inserted)
            first_key = next(iter(self.cache))
            del self.cache[first_key]
        
        key = self._get_cache_key(messages, **kwargs)
        self.cache[key] = response
    
    def clear(self) -> None:
        """Clear all cached responses."""
        self.cache.clear()


# Global cache instance for localhost responses
_localhost_cache = LocalhostResponseCache(max_cache_size=100)


RETRY_ERROR = (
    APIConnectionError,
    APIError,
    APIResponseValidationError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    ConflictError,
    InternalServerError,
    NotFoundError,
    OpenAIError,
    PermissionDeniedError,
    RateLimitError,
    UnprocessableEntityError,
)
CHAT_COMPLETION = {  # These are lambda so set environment variables take effect at runtime
    "openai": lambda: OpenAI(api_key="9b419298-ffce-4d50-a42c-0b4a0b911a89", base_url="https://api.sambanova.ai/v1").chat.completions.create,
    "litellm": lambda: litellm.completion,
}

def get_localhost_client(
    base_url: str = "http://localhost:5000",
    api_key: str = "not-needed",
    timeout_seconds: float | None = 120,
) -> OpenAI:
    """
    Create OpenAI client for localhost model server with v1 API format.
    
    Args:
        base_url: Base URL of the localhost server (default: http://localhost:5000)
        api_key: API key for authentication (default: "not-needed" for local servers)
        timeout_seconds: Request timeout in seconds (default: 120 seconds)
    
    Returns:
        OpenAI client configured for the localhost server with timeout
    """
    if timeout_seconds is None:
        timeout_seconds = 120  # Default timeout if not specified
    
    # Create OpenAI client with explicit timeout configuration
    from openai import APITimeoutError
    client = OpenAI(
        api_key=api_key,
        base_url=base_url.rstrip("/") + "/v1",
        timeout=timeout_seconds,
    )
    return client

def non_cached_chat_completion(
    completion_method: str,
    provider: str,
    model: str,
    messages: list[dict[str, str]],
    frequency_penalty: float | None = None,
    logprobs: bool | None = None,
    top_logprobs: int | None = None,
    max_completion_tokens: int | None = None,
    max_tokens: int | None = None,
    n: int | None = None,
    parallel_tool_calls: bool | None = None,
    presence_penalty: float | None = None,
    reasoning_effort: Literal["low", "medium", "high"] | None = None,
    response_format: dict | None = None,
    seed: int | None = None,
    stop: str | list[str] | None = None,
    temperature: float | None = None,
    tool_choice: str | dict | None = None,
    tools: list | None = None,
    top_p: float | None = None,
    # above params are shared by litellm and openai
    # below params are only for litellm
    logit_bias: dict | None = None,
    thinking: dict | None = None,
    base_url: str | None = None,
    api_version: str | None = None,
    api_key: str | None = None,
    model_list: list | None = None,
    custom_llm_provider: str | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    # Extract localhost parameters FIRST before modifying kwargs
    localhost_url = kwargs.pop("localhost_url", "http://localhost:5000")
    localhost_api_key = kwargs.pop("localhost_api_key", "not-needed")
    localhost_timeout = kwargs.pop("localhost_timeout", None)
    use_localhost_cache = kwargs.pop("use_localhost_cache", True)
    measure_ttft_tpot = kwargs.pop("measure_ttft_tpot", False)
    
    # Check cache for localhost requests BEFORE adding messages to kwargs
    if use_localhost_cache and provider.strip().lower() == "localhost":
        cached_response = _localhost_cache.get(messages, **kwargs)
        if cached_response:
            print(f"✓ [CACHE HIT] Retrieved cached response for localhost request")
            return cached_response
    
    kwargs["model"] = model
    kwargs["messages"] = messages
    # if frequency_penalty is not None:
    #     kwargs["frequency_penalty"] = frequency_penalty
    # if logprobs is not None:
    #     kwargs["logprobs"] = logprobs
    # if top_logprobs is not None:
    #     kwargs["top_logprobs"] = top_logprobs
    # if max_completion_tokens is not None:
    #     kwargs["max_completion_tokens"] = max_completion_tokens
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    # if n is not None:
    #     kwargs["n"] = n
    # if parallel_tool_calls is not None:
    #     kwargs["parallel_tool_calls"] = parallel_tool_calls
    # if presence_penalty is not None:
    #     kwargs["presence_penalty"] = presence_penalty
    # if reasoning_effort is not None:
    #     kwargs["reasoning_effort"] = reasoning_effort
    # if response_format is not None:
    #     kwargs["response_format"] = response_format
    # if seed is not None:
    #     kwargs["seed"] = seed
    if stop is not None:
        kwargs["stop"] = stop
    if temperature is not None:
        kwargs["temperature"] = temperature
    # if tool_choice is not None:
    #     kwargs["tool_choice"] = tool_choice
    # if tools is not None:
    #     kwargs["tools"] = tools
    if top_p is not None:
        kwargs["top_p"] = top_p
    # if logit_bias is not None:
    #     kwargs["logit_bias"] = logit_bias
    # if thinking is not None:
    #     kwargs["thinking"] = thinking
    # if base_url is not None:
    #     kwargs["base_url"] = base_url
    # if api_version is not None:
    #     kwargs["api_version"] = api_version
    # if api_key is not None:
    #     kwargs["api_key"] = api_key
    # if model_list is not None:
    #     kwargs["model_list"] = model_list
    # if custom_llm_provider is not None:
    #     kwargs["custom_llm_provider"] = custom_llm_provider
    
    # Remove non-OpenAI parameters that may have been passed
    params_to_remove = [
        "retry_after_n_seconds", "use_cache", "max_retries", "completion_method",
        "provider", "token_cost_data", "custom_llm_provider"
    ]
    for param in params_to_remove:
        kwargs.pop(param, None)
    
    # Remove None values for optional parameters that OpenAI might reject
    if kwargs.get("tools") is None:
        kwargs.pop("tools", None)
    
    if completion_method not in ["openai", "litellm"]:
        raise ValueError(
            f"Invalid completion_method: {completion_method}. "
            "Valid values are: 'openai' or 'litellm'."
        )
    
    # client = OpenAI(api_key="9b419298-ffce-4d50-a42c-0b4a0b911a89", base_url="https://api.sambanova.ai/v1")
    # # completion = client.chat.completions.create(
    # response = client.chat.completions.create(**kwargs)

    timeout_val = localhost_timeout if localhost_timeout is not None else 120.0

    if provider.strip().lower() == "sambanova":
        from sambanova import SambaNova
        client = SambaNova(timeout=timeout_val)
    elif provider.strip().lower() == "together":
        from together import Together
        client = Together(timeout=timeout_val)
    elif provider.strip().lower() == "openai":
        from openai import OpenAI
        client = OpenAI(timeout=timeout_val)
    elif provider.strip().lower() == "localhost":
        # Support for localhost model server with OpenAI API format
        # Uses configurable timeout for port 5000 requests
        client = get_localhost_client(
            base_url=localhost_url,
            api_key=localhost_api_key,
            timeout_seconds=timeout_val,
        )
    else:
        raise ValueError(
            f"Invalid provider: {provider}. Valid providers: 'openai', 'sambanova', 'together', 'localhost'"
        )

    # Log the actual parameters being sent to the API
    print(f"DEBUG: Sending {len(kwargs)} parameters to {provider} API")
    print(f"DEBUG: API parameters: {list(kwargs.keys())}")
    
    try:
        start_time = time.time()
        if measure_ttft_tpot:
            try:
                stream_kwargs = kwargs.copy()
                stream_kwargs["stream"] = True
                response_stream = client.chat.completions.create(**stream_kwargs)

                full_content = ""
                tool_calls_deltas = {}
                finish_reason = None
                role = "assistant"
                response_id = f"chatcmpl-{uuid.uuid4().hex}"
                ttft = None

                for chunk in response_stream:
                    choices = getattr(chunk, "choices", [])
                    if not choices:
                        continue

                    chunk_id = getattr(chunk, "id", None)
                    if chunk_id:
                        response_id = chunk_id

                    choice = choices[0]
                    delta = getattr(choice, "delta", None)
                    if delta is None:
                        continue

                    has_content = False
                    content_delta = getattr(delta, "content", None)
                    if content_delta is not None and content_delta != "":
                        full_content += content_delta
                        has_content = True

                    tool_calls_delta = getattr(delta, "tool_calls", None)
                    if tool_calls_delta is not None:
                        has_content = True
                        for tc in tool_calls_delta:
                            idx = getattr(tc, "index", None)
                            if idx is None:
                                continue
                            if idx not in tool_calls_deltas:
                                tool_calls_deltas[idx] = {
                                    "id": getattr(tc, "id", None),
                                    "type": getattr(tc, "type", "function"),
                                    "function": {
                                        "name": "",
                                        "arguments": "",
                                    },
                                }
                            tc_id = getattr(tc, "id", None)
                            if tc_id is not None:
                                tool_calls_deltas[idx]["id"] = tc_id

                            func_delta = getattr(tc, "function", None)
                            if func_delta is not None:
                                name_delta = getattr(func_delta, "name", None)
                                if name_delta is not None:
                                    tool_calls_deltas[idx]["function"]["name"] += name_delta
                                args_delta = getattr(func_delta, "arguments", None)
                                if args_delta is not None:
                                    tool_calls_deltas[idx]["function"]["arguments"] += args_delta

                    f_reason = getattr(choice, "finish_reason", None)
                    if f_reason is not None:
                        finish_reason = f_reason

                    role_delta = getattr(delta, "role", None)
                    if role_delta is not None:
                        role = role_delta

                    if has_content and ttft is None:
                        ttft = time.time() - start_time

                end_time = time.time()
                total_time = end_time - start_time

                # Format tool_calls list if any
                tool_calls = []
                if tool_calls_deltas:
                    for idx in sorted(tool_calls_deltas.keys()):
                        tool_calls.append(tool_calls_deltas[idx])

                output_tokens = 0
                if full_content:
                    try:
                        output_tokens += token_counter(model=model, text=full_content)
                    except Exception:
                        output_tokens += len(full_content.split())
                if tool_calls_deltas:
                    for idx, tc in tool_calls_deltas.items():
                        func = tc.get("function", {})
                        try:
                            output_tokens += token_counter(model=model, text=func.get("name", "") + func.get("arguments", ""))
                        except Exception:
                            output_tokens += len(func.get("name", "").split()) + len(func.get("arguments", "").split())

                if ttft is None:
                    ttft = total_time
                    tpot = 0.0
                else:
                    if output_tokens > 0:
                        tpot = (total_time - ttft) / output_tokens
                    else:
                        tpot = 0.0

                try:
                    input_tokens = token_counter(model=model, messages=messages)
                except Exception:
                    input_tokens = len(str(messages).split())

                usage = {
                    "prompt_tokens": input_tokens,
                    "completion_tokens": output_tokens,
                    "total_tokens": input_tokens + output_tokens,
                }

                response = {
                    "id": response_id,
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": model,
                    "choices": [
                        {
                            "message": {
                                "role": role,
                                "content": full_content,
                            },
                            "finish_reason": finish_reason or "stop",
                            "index": 0,
                        }
                    ],
                    "usage": usage,
                    "ttft": ttft,
                    "tpot": tpot,
                }
                if tool_calls:
                    response["choices"][0]["message"]["tool_calls"] = tool_calls

            except Exception as stream_err:
                print(f"Warning: streaming failed ({stream_err}), falling back to non-streaming.")
                start_time = time.time()
                response_raw = client.chat.completions.create(**kwargs)
                end_time = time.time()
                total_time = end_time - start_time

                response = to_dict(response_raw)

                output_tokens = 0
                try:
                    choice = response["choices"][0]
                    content = choice["message"].get("content", "")
                    if content:
                        output_tokens += token_counter(model=model, text=content)
                    tcs = choice["message"].get("tool_calls", [])
                    for tc in tcs:
                        func = tc.get("function", {})
                        output_tokens += token_counter(model=model, text=func.get("name", "") + func.get("arguments", ""))
                except Exception:
                    pass

                if output_tokens == 0:
                    output_tokens = 1

                ttft = total_time * 0.5
                tpot = (total_time - ttft) / output_tokens

                response["ttft"] = ttft
                response["tpot"] = tpot
        else:
            response_raw = client.chat.completions.create(**kwargs)
            response = to_dict(response_raw)
            
    except TypeError as e:
        print(f"TypeError calling {provider} API: {str(e)}")
        print(f"DEBUG: Invalid parameters: {list(kwargs.keys())}")
        raise
    except Exception as e:
        print(f"Error calling {provider} API: {str(e)}")
        raise
    
    if response is None:
        raise ValueError(f"API call returned None for provider {provider}")
    
    response = to_dict(response)
    if response is None:
        raise ValueError(f"to_dict returned None for provider {provider}")
    
    # Ensure response is a valid dict with required structure
    if not isinstance(response, dict):
        raise ValueError(f"Expected dict response, got {type(response)} for provider {provider}")
    
    if "choices" not in response or not response["choices"]:
        # If choices are missing, try to add a default empty choice
        print(f"WARNING: Response missing or empty 'choices' field for provider {provider}")
        print(f"DEBUG: Response keys: {list(response.keys()) if isinstance(response, dict) else 'N/A'}")
        raise ValueError(f"Response must contain 'choices' field for provider {provider}")
    
    # Cache successful response
    if use_localhost_cache:
        # Create cache kwargs without messages (already passed as positional arg)
        cache_kwargs = {k: v for k, v in kwargs.items() if k != 'messages'}
        _localhost_cache.set(response, messages, **cache_kwargs)
        print(f"✓ [CACHE STORED] Response cached for {provider} request")
    
    return response


@cache.cache
def cached_chat_completion(
    completion_method: str,
    provider: str,
    model: str,
    messages: list[dict[str, str]],
    frequency_penalty: float | None = None,
    logprobs: bool | None = None,
    top_logprobs: int | None = None,
    max_completion_tokens: int | None = None,
    max_tokens: int | None = None,
    n: int | None = None,
    parallel_tool_calls: bool | None = None,
    presence_penalty: float | None = None,
    reasoning_effort: Literal["low", "medium", "high"] | None = None,
    response_format: dict | None = None,
    seed: int | None = None,
    stop: str | list[str] | None = None,
    temperature: float | None = None,
    tool_choice: str | dict | None = None,
    tools: list | None = None,
    top_p: float | None = None,
    # above params are shared by litellm and openai
    # below params are only for litellm
    logit_bias: dict | None = None,
    thinking: dict | None = None,
    base_url: str | None = None,
    api_version: str | None = None,
    api_key: str | None = None,
    model_list: list | None = None,
    custom_llm_provider: str | None = None,
    **kwargs: Any,
) -> dict[str, Any]:

    return non_cached_chat_completion(
        completion_method=completion_method,
        provider=provider,
        model=model,
        messages=messages,
        frequency_penalty=frequency_penalty,
        logprobs=logprobs,
        top_logprobs=top_logprobs,
        max_completion_tokens=max_completion_tokens,
        max_tokens=max_tokens,
        n=n,
        parallel_tool_calls=parallel_tool_calls,
        presence_penalty=presence_penalty,
        reasoning_effort=reasoning_effort,
        response_format=response_format,
        seed=seed,
        stop=stop,
        temperature=temperature,
        tool_choice=tool_choice,
        tools=tools,
        top_p=top_p,
        logit_bias=logit_bias,
        thinking=thinking,
        base_url=base_url,
        api_version=api_version,
        api_key=api_key,
        model_list=model_list,
        custom_llm_provider=custom_llm_provider,
        **kwargs,
    )


class LiteLLMGenerator:
    def __init__(
        self,
        name: str,
        completion_method: Literal["openai", "litellm"] = "openai",
        retry_after_n_seconds: int | None = None,
        max_retries: int = 500,
        use_cache: bool = False,
        token_cost_data: dict | None = None,
        localhost_timeout: float | None = None,
        use_localhost_cache: bool = True,
        measure_ttft_tpot: bool = False,
        **generation_kwargs: Any,
    ) -> None:
        self.model = name
        self.localhost_timeout = localhost_timeout if localhost_timeout is not None else 120  # Default 2 minutes for port 5000
        self.use_localhost_cache = use_localhost_cache
        default_custom_llm_provider = (
            "openai" if name not in litellm.model_cost and completion_method == "openai" else None
        )
        self.custom_llm_provider = generation_kwargs.get(
            "custom_llm_provider", default_custom_llm_provider
        )
        # Extract provider from generation_kwargs, default to "openai"
        self.provider = generation_kwargs.get("provider", "openai")
        valid_providers = ["openai", "sambanova", "together", "localhost"]
        if self.provider not in valid_providers:
            raise ValueError(
                f"Invalid provider: {self.provider}. Valid providers: {valid_providers}"
            )
        if token_cost_data:
            litellm.model_cost[name] = token_cost_data
        elif name not in litellm.model_cost:
            warning_message = (
                f"[yellow]litellm does not have token cost data for model '{name}'. "
                "So the cost tracking and logging will not work. If you need it, though, pass 'token_cost_data' "
                "in the config file in the same format as litellm.model_cost[name].[/yellow]"
            )
            rprint(
                Panel(warning_message, title="[bold red]Warning[/bold red]", border_style="yellow")
            )
        if completion_method not in ["openai", "litellm"]:
            raise ValueError(
                f"Invalid completion_method: {completion_method}. "
                "Valid values are: 'openai' or 'litellm'."
            )
        self.max_input_tokens = litellm.model_cost.get("name", {}).get("max_input_tokens", None)
        self.max_output_tokens = litellm.model_cost.get("name", {}).get("max_output_tokens", None)
        self.retry_after_n_seconds = retry_after_n_seconds
        self.max_retries = max_retries
        self.measure_ttft_tpot = measure_ttft_tpot
        self.chat_completion = {
            True: cached_chat_completion,
            False: non_cached_chat_completion,
        }[use_cache]
        if completion_method == "openai":
            # LiteLLM accepts these two arguments in completion function, whereas OpenAI
            # accepts them in the OpenAI constructor or in the environment variables.
            if "api_key" in generation_kwargs:
                os.environ["OPENAI_API_KEY"] = generation_kwargs.pop("api_key")
            if "base_url" in generation_kwargs:
                os.environ["OPENAI_BASE_URL"] = generation_kwargs.pop("base_url")
            generation_kwargs.pop("custom_llm_provider", None)
        # For localhost provider, keep localhost_url and localhost_api_key in kwargs
        # They will be used in non_cached_chat_completion function
        valid_generation_kwargs_keys = set(
            inspect.signature(CHAT_COMPLETION[completion_method]()).parameters.keys()
        )
        invalid_keys = set(generation_kwargs.keys()) - valid_generation_kwargs_keys
        # if invalid_keys:
        #     raise ValueError(
        #         f"Invalid generation kwargs: {invalid_keys}. "
        #         f"Valid keys are: {valid_generation_kwargs_keys}"
        #     )
        if "max_tokens" not in generation_kwargs and self.max_output_tokens:
            generation_kwargs["max_tokens"] = self.max_output_tokens
        generation_kwargs["completion_method"] = completion_method
        generation_kwargs["provider"] = self.provider
        generation_kwargs["localhost_timeout"] = self.localhost_timeout  # Add timeout for localhost requests
        generation_kwargs["use_localhost_cache"] = self.use_localhost_cache  # Add cache flag for localhost
        generation_kwargs["measure_ttft_tpot"] = self.measure_ttft_tpot
        self.generation_kwargs = generation_kwargs
        self.cost = 0
        self.log_file_path = None

    def generate(
        self,
        messages: list[dict[str, str]],
        tools: list[dict] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        used_num_tokens = token_counter(model=self.model, messages=messages)
        if self.max_input_tokens and used_num_tokens > self.max_input_tokens:
            print(
                "WARNING: Ran out of context limit of this model. "
                f"Model: {self.model}, used_num_tokens: {used_num_tokens}, "
                f"max_num_tokens: {self.max_num_tokens}"
            )
            return {"content": "", "tool_calls": [], "cost": 0}

        success = False
        last_exception = None
        for attempt in range(self.max_retries):
            try:
                arguments = {
                    "model": self.model,
                    "messages": messages,
                    "tools": tools,
                    **(self.generation_kwargs | kwargs),
                }
                print(f"DEBUG: Calling chat_completion with provider={self.provider}, model={self.model}")
                if self.measure_ttft_tpot:
                    response = self.chat_completion(**arguments)
                    print(f"DEBUG: Chat completion returned: {type(response)} - {response is None}")
                    if response is None:
                        print(f"ERROR: Chat completion returned None for provider={self.provider}")
                        raise ValueError("Chat completion returned None")
                    response["cost"] = self.completion_cost(completion_response=response)
                    self.may_log_call(arguments, response)
                else:
                    response_raw = self.chat_completion(**arguments)
                    print(f"DEBUG: Chat completion returned: {type(response_raw)} - {response_raw is None}")
                    if response_raw is None:
                        print(f"ERROR: Chat completion returned None for provider={self.provider}")
                        raise ValueError("Chat completion returned None")
                    response = to_dict(response_raw)
                    response["cost"] = self.completion_cost(completion_response=response)
                    self.may_log_call(arguments, response)
                success = True
                break
            except APITimeoutError as exception:
                success = False
                last_exception = exception
                
                # Check cache on timeout
                timeout_val = self.localhost_timeout if self.localhost_timeout is not None else 120.0
                print(f"[TIMEOUT after {timeout_val}s] Checking cache...")
                
                # 1. Check persistent joblib cache (cached_chat_completion)
                try:
                    if cached_chat_completion.check_call_in_cache(**arguments):
                        print(f"✓ [CACHE RESCUE] Retrieved response from joblib cache")
                        response = cached_chat_completion(**arguments)
                        response["cost"] = self.completion_cost(completion_response=response)
                        self.may_log_call(arguments, response)
                        success = True
                        break
                except Exception as cache_err:
                    print(f"Error checking joblib cache: {cache_err}")
                
                # 2. Check in-memory cache as fallback
                if self.use_localhost_cache:
                    time.sleep(0.5)  # Brief moment for server to complete
                    exclude_keys = {
                        "messages", "completion_method", "provider", "model",
                        "frequency_penalty", "logprobs", "top_logprobs", "max_completion_tokens",
                        "max_tokens", "n", "parallel_tool_calls", "presence_penalty",
                        "reasoning_effort", "response_format", "seed", "stop", "temperature",
                        "tool_choice", "tools", "top_p", "logit_bias", "thinking", "base_url",
                        "api_version", "api_key", "model_list", "custom_llm_provider",
                        "localhost_url", "localhost_api_key", "localhost_timeout", "use_localhost_cache",
                        "retry_after_n_seconds", "use_cache", "max_retries"
                    }
                    cache_kwargs = {k: v for k, v in (self.generation_kwargs | kwargs).items() if k not in exclude_keys}
                    cached_response = _localhost_cache.get(messages, **cache_kwargs)
                    
                    if cached_response:
                        print(f"✓ [CACHE RESCUE] Retrieved cached response from in-memory cache")
                        cached_dict = to_dict(cached_response)
                        output = {**cached_dict["choices"][0]["message"], "cost": 0}
                        return output
                    elif self.provider.strip().lower() == "localhost":
                        # No cache available for localhost - return default response instead of retrying
                        print(f"⚠️ [TIMEOUT] No cached response. Returning empty result (no retry).")
                        return {
                            "content": f"[Request timed out after {timeout_val}s. No cached response available.]",
                            "tool_calls": [],
                            "cost": 0
                        }
                
                # For non-localhost providers: continue with retry logic
                if self.retry_after_n_seconds is None:
                    import traceback
                    print(traceback.format_exc())
                    exit()
                print(f"Timeout Error: {str(exception)[:200]}")
                print(f"Will try again in {self.retry_after_n_seconds} seconds...")
                time.sleep(self.retry_after_n_seconds)
                
            except RETRY_ERROR as exception:
                success = False
                last_exception = exception
                if self.retry_after_n_seconds is None:
                    import traceback
                    print(traceback.format_exc())
                    exit()
                print(f"Encountered LM Error: {exception.message[:200].strip()}...")
                print(f"Will try again in {self.retry_after_n_seconds} seconds.")
                time.sleep(self.retry_after_n_seconds)
            except (ValueError, KeyError, TypeError) as exception:
                success = False
                last_exception = exception
                if self.retry_after_n_seconds is None:
                    import traceback
                    print(traceback.format_exc())
                    exit()
                print(f"Encountered Error: {str(exception)[:200]}...")
                print(f"Will try again in {self.retry_after_n_seconds} seconds.")
                time.sleep(self.retry_after_n_seconds)

        if not success:
            if last_exception:
                raise Exception(f"Could not complete LM call after {self.max_retries} retries: {str(last_exception)}")
            else:
                raise Exception("Could not complete LM call")
        
        if "chat_template_kwargs" in self.generation_kwargs:
            response["choices"][0]["message"]["content"] = response["choices"][0]["message"]["content"].split("<think>\n")[-1]

        if not hasattr(self, "ttft_history"):
            self.ttft_history = []
        if not hasattr(self, "tpot_history"):
            self.tpot_history = []
        if not hasattr(self, "input_tokens_history"):
            self.input_tokens_history = []
        if not hasattr(self, "output_tokens_history"):
            self.output_tokens_history = []

        ttft = response.get("ttft")
        tpot = response.get("tpot")
        if ttft is not None:
            self.ttft_history.append(ttft)
        if tpot is not None:
            self.tpot_history.append(tpot)

        input_tokens = response.get("usage", {}).get("prompt_tokens", 0)
        output_tokens = response.get("usage", {}).get("completion_tokens", 0)
        self.input_tokens_history.append(input_tokens)
        self.output_tokens_history.append(output_tokens)

        output = {
            **response["choices"][0]["message"],
            "cost": response["cost"],
            "ttft": ttft,
            "tpot": tpot,
            "prompt_tokens": input_tokens,
            "completion_tokens": output_tokens
        }
        return output

    def may_log_call(self, arguments: dict, response: dict) -> None:
        log_data = {"id": uuid.uuid4().hex, "input": arguments, "output": response}
        if self.log_file_path:
            os.makedirs(os.path.dirname(self.log_file_path), exist_ok=True)
            write_jsonl([log_data], self.log_file_path, append=True, silent=True)

    def log_calls_to(self, file_path: str | None = None, world: AppWorld | None = None) -> None:
        if (world and file_path) or (not world and not file_path):
            raise ValueError("Either world or file_path must be provided.")
        if world:
            file_path = os.path.join(world.output_logs_directory, "lm_calls.jsonl")
        self.log_file_path = file_path

    def completion_cost(self, *args: Any, **kwargs: Any) -> float:
        if self.model in litellm.model_cost:
            if self.custom_llm_provider:
                kwargs["custom_llm_provider"] = self.custom_llm_provider
            return round(completion_cost(*args, **kwargs), 8)
        return 0.0


def to_dict(obj: Any) -> Any:
    if hasattr(obj, "json"):
        return {k: to_dict(v) for k, v in dict(obj).items()}
    elif isinstance(obj, list):
        return [to_dict(item) for item in obj]
    elif isinstance(obj, dict):
        return {k: to_dict(v) for k, v in obj.items()}
    return obj