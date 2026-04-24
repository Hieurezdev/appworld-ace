# Quick Fix Reference - ReadTimeout Error Resolution

## What Was Fixed

**Error:** `ReadTimeout: HTTPConnectionPool(host='0.0.0.0', port=8000): Read timed out. (read timeout=40)`

**Solution:** Comprehensive timeout, retry, and connection pooling implementation.

## Key Changes

| Component | Change | Benefit |
|-----------|--------|---------|
| Timeout | Added explicit 40-second timeout to all HTTP requests | Prevents indefinite hangs |
| Retries | Automatic retry with exponential backoff (3 attempts) | Handles transient failures |
| Sessions | Session pooling per remote URL | Reduces connection overhead |
| Error Handling | Try-except with logging on all requests | Visible error debugging |

## Usage

### Basic Usage (Recommended)
```python
from appworld.environment import Environment

env = Environment(
    task_id="your_task_id",
    remote_apis_url="http://0.0.0.0:9000",
    remote_environment_url="http://0.0.0.0:8000",
    # Uses default timeout of 40 seconds
)
```

### Custom Timeout
```python
env = Environment(
    task_id="your_task_id",
    remote_apis_url="http://0.0.0.0:9000",
    remote_environment_url="http://0.0.0.0:8000",
    timeout_seconds=120,  # 2 minutes
)
```

### Enable Debug Logging
```python
import logging
logging.basicConfig(level=logging.DEBUG)

env = Environment(
    task_id="your_task_id",
    remote_apis_url="http://0.0.0.0:9000",
    timeout_seconds=60,
)
```

## Retry Behavior

With default settings (max_retries=3, backoff_factor=0.3):
- **Failure timing:** Request fails after ~42 seconds
- **Retry pattern:** 0.3s → 0.6s → 1.2s delays between retries
- **Total time:** ~42 seconds (timeout + retry delays)

## Configuration Parameters

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `timeout_seconds` | 40 | Timeout per request in seconds |
| `max_retries` | 3 | Number of retry attempts |
| `backoff_factor` | 0.3 | Exponential backoff multiplier |

## Troubleshooting

### Still Getting Timeouts?
1. **Increase timeout:**
   ```python
   timeout_seconds=180  # 3 minutes
   ```

2. **Check server logs** - See if remote server is responding slowly

3. **Enable debug logging** - Identify which endpoints are timing out

### Server Unreachable?
1. **Verify connectivity:** `ping 0.0.0.0` or `curl http://0.0.0.0:8000/health`
2. **Check firewall** - Ensure port 8000/9000 are accessible
3. **Reduce timeout to fail faster:**
   ```python
   timeout_seconds=10  # Fail after ~12 seconds instead of 42
   ```

## Performance Tips

1. **Use connection pooling** - Already enabled automatically
2. **Reuse Environment instances** - Don't create new ones unnecessarily
3. **Batch requests** - Send multiple requests in sequence to benefit from pooling
4. **Monitor latency** - Log successful request times to tune timeout

## API Reference

### Requester Class
```python
from appworld.requester import Requester

requester = Requester(
    to_db_home_path="...",
    from_db_home_path="...",
    remote_apis_url="http://0.0.0.0:9000",
    timeout_seconds=40,          # New parameter
    max_retries=3,               # New parameter
    backoff_factor=0.3,          # New parameter
)

# All methods now support timeout
response = requester.get(url)
response = requester.post(url, data)
response = requester.put(url, data)
response = requester.patch(url, data)
response = requester.delete(url, data)
```

### Environment Class
```python
from appworld.environment import Environment

env = Environment(
    task_id="...",
    timeout_seconds=60,  # New parameter
    remote_apis_url="http://0.0.0.0:9000",
    remote_environment_url="http://0.0.0.0:8000",
)
```

## Log Examples

### Successful Request
```
[No log output - request succeeded within timeout]
```

### Timeout Error
```
ERROR:appworld.requester:GET request timeout for http://0.0.0.0:9000/api/v1/users: 
    HTTPConnectionPool(host='0.0.0.0', port=9000): Read timed out. (read timeout=40)
```

### Connection Error
```
ERROR:appworld.requester:GET request failed for http://0.0.0.0:9000/api/v1/users: 
    ConnectionError: Failed to establish a new connection
```

## Migration Guide

### Before (Problematic)
```python
env = Environment(task_id="my_task")
# Could timeout indefinitely without retry
```

### After (Fixed)
```python
env = Environment(
    task_id="my_task",
    timeout_seconds=40,  # Explicit timeout
    # Automatic retries included
    # Connection pooling enabled
)
```

## Summary

✅ **Fixed:** Indefinite hangs and lack of retry logic
✅ **Added:** Explicit timeouts with automatic exponential backoff retries
✅ **Improved:** Connection reuse through session pooling
✅ **Enabled:** Error logging for debugging
✅ **Maintained:** Backward compatibility

For detailed documentation, see: [TIMEOUT_FIX_DOCUMENTATION.md](TIMEOUT_FIX_DOCUMENTATION.md)
