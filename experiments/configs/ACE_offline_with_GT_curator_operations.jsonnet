// Original ACE adaptation + all improved Curator lifecycle operations.
// RAE, FMB, adversarial generation, and automatic hygiene remain disabled.
local project_home_path = std.extVar("APPWORLD_PROJECT_PATH");
local experiment_prompts_path = project_home_path + "/experiments/prompts";
local experiment_playbooks_path = project_home_path + "/experiments/playbooks";

local generator_model_config = {
    "name": "Qwen/Qwen3-4B-Instruct-2507",
    "provider": "localhost",
    "localhost_url": "http://localhost:5000",
    "localhost_api_key": "not-needed",
    "temperature": 0,
    "seed": 100,
    "stop": ["<|endoftext|>", "<|eot_id|>", "<|start_header_id|>"],
    "logprobs": false,
    "top_logprobs": null,
    "frequency_penalty": 0,
    "presence_penalty": 0,
    "n": 1,
    "response_format": {"type": "text"},
    "retry_after_n_seconds": 10,
    "use_cache": true,
    "max_retries": 50,
};

local reflector_model_config = generator_model_config;
local curator_model_config = generator_model_config;

{
    "type": "ace",
    "config": {
        "run_type": "ace-adaptation",
        "agent": {
            "type": "ace_adaptation_react",
            "generator_model_config": generator_model_config,
            "reflector_model_config": reflector_model_config,
            "curator_model_config": curator_model_config,
            "appworld_config": {
                "random_seed": 123,
                "remote_environment_url": "http://0.0.0.0:8000",
                "remote_apis_url": "http://0.0.0.0:9000",
            },
            "logger_config": {
                "color": true,
                "verbose": true,
            },
            "generator_prompt_file_path": experiment_prompts_path + "/appworld_react_generator_prompt.txt",
            "reflector_prompt_file_path": experiment_prompts_path + "/appworld_react_reflector_with_gt_prompt.txt",
            "curator_prompt_file_path": experiment_prompts_path + "/appworld_react_curator_prompt.txt",
            "initial_playbook_file_path": experiment_playbooks_path + "/appworld_initial_playbook.txt",
            "trained_playbook_file_path": experiment_playbooks_path + "/appworld_offline_trained_with_gt_curator_operations_playbook.txt",

            // ADD (legacy) + UPDATE, DELETE, MERGE, CREATE_META.
            "use_lifecycle_curator": true,
            "use_curator_update": false,
            "use_curator_delete": false,
            "use_curator_merge": false,
            "use_curator_create_meta": false,
            "delete_harmful_margin": 4,
            "delete_min_harmful": 3,
            // Full lifecycle runs conservative zero-evidence pruning every 50 tasks.
            "prune_unused_bullets": true,
            "prune_unused_interval": 50,

            // Keep separate post-Curator hygiene and DBSCAN proposals off.
            "use_bulletpoint_analyzer": false,
            "use_dbscan_merge": false,
            "use_dbscan_merge_candidates": true,

            "ignore_multiple_calls": true,
            "max_steps": 40,
            "max_cost_overall": 1000,
            "max_cost_per_task": 10,
            "log_lm_calls": true,
            "use_gt_code": true,
        },
        "dataset": "train",
    },
}
