# ReadTimeout Error Fix - Comprehensive Solution

## Problem Summary
The application was experiencing `ReadTimeout: HTTPConnectionPool(host='0.0.0.0', port=8000): Read timed out. (read timeout=40)` errors when making remote HTTP requests to the environment server (port 8000) and APIs server (port 9000).

**Root Causes:**
1. **No explicit timeout handling** - HTTP requests were made without specifying timeout values, causing indefinite hangs or system default timeouts
2. **No retry logic** - Transient failures (connection errors, temporary timeouts) were not retried
3. **No connection pooling** - Each request created a new connection instead of reusing existing connections
4. **No session reuse** - Multiple sessions were created instead of reusing a single session per URL
5. **No error monitoring** - Lack of logging made debugging difficult

## Solution Overview

### 1. Session Creation with Automatic Retries (`requester.py`)

**Added:** `create_session_with_retries()` function
- Creates a `requests.Session` with automatic retry logic
- Uses `urllib3.util.retry.Retry` with exponential backoff
- Retries on status codes: 429, 500, 502, 503, 504
- Configurable parameters:
  - `timeout`: Default 40 seconds (customizable)
  - `max_retries`: Default 3 retries
  - `backoff_factor`: Default 0.3 (exponential backoff multiplier)

```python
def create_session_with_retries(
    timeout: float = 40,
    max_retries: int = 3,
    backoff_factor: float = 0.3,
) -> requests.Session:
    # Creates session with retry strategy
    # Mounted on both http:// and https://
```

### 2. Enhanced Requester Class

**Changes to `Requester.__init__()`:**
- Added timeout configuration parameters:
  - `timeout_seconds` (default: 40)
  - `max_retries` (default: 3)
  - `backoff_factor` (default: 0.3)
- Added session caching per remote URL:
  - `remote_sessions` class variable stores sessions by URL
  - Sessions are reused across Requester instances
  - Reduces connection overhead

### 3. Timeout Applied to All HTTP Requests

**Updated methods:** `_get()`, `_post()`, `_put()`, `_patch()`, `_delete()`

**Key improvements:**
- All HTTP requests now include `timeout=self.timeout_seconds`
- Wrapped in try-except to catch timeout and connection errors
- Error logging for debugging:
  ```python
  try:
      response = self.session.get(url, params=data, headers=headers, 
                                   timeout=self.timeout_seconds)
  except requests.exceptions.Timeout as e:
      logger.error(f"GET request timeout for {url}: {e}")
      raise
  except requests.exceptions.RequestException as e:
      logger.error(f"GET request failed for {url}: {e}")
      raise
  ```

### 4. Configuration Propagation

**Updated `ApiCollection.load()`:**
- Added parameters: `timeout_seconds`, `max_retries`, `backoff_factor`
- Passes these to `Requester` instantiation

**Updated `Environment.initialize()`:**
- Passes `timeout_seconds` from environment config to `ApiCollection.load()`

## How the Fix Works

### Retry Mechanism
When a request fails with a transient error (e.g., temporary network issue):
1. urllib3's Retry strategy automatically retries the request
2. Each retry uses exponential backoff: `backoff_factor × (2 ^ (retry - 1))`
3. Example with default backoff_factor=0.3:
   - Retry 1: 0.3 seconds
   - Retry 2: 0.6 seconds
   - Retry 3: 1.2 seconds

### Connection Pooling
- Each remote URL has a single `requests.Session` object
- Session maintains connection pool and reuses connections
- Significantly reduces overhead for repeated requests

### Timeout Handling
- Every HTTP request has an explicit timeout (default 40 seconds)
- Timeout can be increased if needed by passing `timeout_seconds` parameter
- Prevents indefinite hangs

## Configuration

### Default Settings
```python
timeout_seconds = 40  # seconds
max_retries = 3       # number of retries
backoff_factor = 0.3  # exponential backoff multiplier
```

### Custom Configuration
When creating an Environment:
```python
env = Environment(
    task_id="...",
    timeout_seconds=120,  # 2 minutes instead of 40 seconds
    remote_apis_url="http://0.0.0.0:9000",
    remote_environment_url="http://0.0.0.0:8000",
)
```

### Environment Variable (in AppWorldInitDefaults)
```python
timeout_seconds: int | None = 100  # Can be overridden per instance
```

## Monitoring & Debugging

### Error Logging
Enable logging to see request errors:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Log Messages
- Connection timeouts: `"GET request timeout for {url}"`
- Connection failures: `"GET request failed for {url}"`
- Includes exception details for debugging

## Performance Benefits

1. **Faster for healthy networks**: Connection reuse reduces handshake overhead
2. **More reliable**: Automatic retries handle transient failures
3. **Better resource usage**: Session pooling reduces memory and file descriptor usage
4. **Observable**: Logging provides visibility into request issues

## Fallback Behavior

If the remote server becomes unavailable:
- Initial request timeout: 40 seconds (default)
- With max_retries=3 and backoff_factor=0.3:
  - Total time before failure: ~42 seconds
  - Pattern: Initial + retry(0.3s) + retry(0.6s) + retry(1.2s)

To fail faster, reduce `timeout_seconds`:
```python
env = Environment(
    task_id="...",
    timeout_seconds=10,  # Fail after ~12 seconds
)
```

## Technical Details

### Session Caching Implementation
```python
class Requester:
    remote_sessions: ClassVar[dict[str, requests.Session]] = {}
    
    def __init__(self, remote_apis_url: str | None = None, ...):
        if remote_apis_url:
            if remote_apis_url not in self.remote_sessions:
                self.remote_sessions[remote_apis_url] = create_session_with_retries(...)
            self.session = self.remote_sessions[remote_apis_url]
```

### Retry Strategy Configuration
```python
retry_strategy = Retry(
    total=max_retries,                      # Total retries
    status_forcelist=[429, 500, 502, 503, 504],  # HTTP codes to retry on
    method_whitelist=[...],                 # HTTP methods that can be retried
    backoff_factor=backoff_factor,          # Exponential backoff factor
)
```

## Files Modified

1. **src/appworld/requester.py**
   - Added logging and retry imports
   - Added `create_session_with_retries()` function
   - Updated `Requester` class with timeout configuration
   - Updated all HTTP request methods with timeout and error handling

2. **src/appworld/collections/apis.py**
   - Updated `ApiCollection.load()` to accept timeout parameters
   - Passes parameters to `Requester` initialization

3. **src/appworld/environment.py**
   - Updated `Environment.initialize()` to pass `timeout_seconds` to `ApiCollection.load()`

## Verification

To verify the fix is working:

```python
from appworld.environment import Environment

env = Environment(
    task_id="example_task_id",
    remote_apis_url="http://0.0.0.0:9000",
    remote_environment_url="http://0.0.0.0:8000",
    timeout_seconds=60,  # Custom timeout
)

# These requests will now:
# - Use connection pooling
# - Have explicit timeout of 60 seconds
# - Retry up to 3 times on transient failures
response = env.requester.get("http://0.0.0.0:9000/some/endpoint")
```

## Future Improvements

Potential enhancements not included in this fix:

1. **Circuit Breaker Pattern**: Fail fast after repeated failures
2. **Health Checks**: Periodic health checks to detect server unavailability early
3. **Adaptive Timeouts**: Adjust timeout based on historical latency
4. **Request Prioritization**: Queue important requests ahead of others
5. **Metrics Collection**: Gather statistics on request latency and success rates
6. **Request Caching**: Cache frequently requested data

## Conclusion

This comprehensive fix addresses the root causes of the ReadTimeout error by implementing:
- **Explicit timeout handling** on all requests
- **Automatic retry logic** for transient failures
- **Connection pooling** for performance
- **Session reuse** to reduce overhead
- **Error logging** for visibility

The solution is backward compatible and configurable, allowing users to adjust timeout values based on their specific network conditions and requirements.
