#!/usr/bin/env python
import requests
import json

# Test remote environment initialization
url = "http://0.0.0.0:8000/initialize"

payload = {
    "task_id": "024c982_1",
    "experiment_name": "test",
    "remote_apis_url": "http://0.0.0.0:9000",
    "remote_environment_url": None,
    "remote_docker": False,
    "max_interactions": 10,
    "max_api_calls_per_interaction": 100,
    "raise_on_unsafe_syntax": False,
    "null_patch_unsafe_execution": False,
    "load_ground_truth": True,
    "ground_truth_mode": "minimal",
    "raise_on_failure": False,
    "random_seed": 123,
    "timeout_seconds": None,
    "gc_threshold": None,
    "raise_on_extra_parameters": False,
    "import_utils": True,
    "parse_datetimes": False,
    "allow_datetime_change": False,
    "add_login_shortcut": False,
    "munchify_response": False,
    "show_api_response_schemas": True,
}

print("Sending initialize request...")
print(f"URL: {url}")
print(f"Payload: {json.dumps(payload, indent=2)}")
print()

try:
    response = requests.post(url, json=payload)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.text}")
except Exception as e:
    print(f"Error: {e}")
