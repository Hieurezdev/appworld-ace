local project_home_path = std.extVar("APPWORLD_PROJECT_PATH");
local experiment_prompts_path = project_home_path + "/experiments/prompts";
local experiment_playbooks_path = project_home_path + "/experiments/playbooks";
local experiment_configs_path = project_home_path + "/experiments/configs";
local experiment_code_path = project_home_path + "/experiments/code";

local model_config = {
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

{
    "type": "ace",
    "config": {
        "run_type": "ace-adaptation",
        "agent": {
            "type": "ace_adaptation_react",
            "generator_model_config": model_config,
            "reflector_model_config": model_config,
            "curator_model_config": model_config,
            "adversarial_model_config": model_config,
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
            "adversarial_prompt_file_path": experiment_prompts_path + "/appworld_react_adversarial_prompt.txt",
            "initial_playbook_file_path": experiment_playbooks_path + "/appworld_initial_playbook.txt",
            "trained_playbook_file_path": experiment_playbooks_path + "/appworld_offline_with_GT_casebank_FMB_adversarial_playbook.txt",
            "ignore_multiple_calls": true,
            "max_steps": 40,
            "max_cost_overall": 1000,
            "max_cost_per_task": 10,
            "log_lm_calls": true,
            "use_gt_code": true,
            "use_hybrid_adversarial": true,

            "casebank_file_path": project_home_path + "/MementoExperiment/memory/memory_casebank_FMB_adversarial.jsonl",
            "casebank_top_k": 4,
            "casebank_retrieval_type": "non-parametric", // Options: "non-parametric" or "parametric"
            "casebank_retriever_model_path": project_home_path + "/MementoExperiment/memory/ckpts/retriever/best.pt",
            "casebank_model": "BAAI/bge-m3",
            "reflector_memory_top_k": 10,
            "reflector_memory_bank_file": experiment_playbooks_path + "/failure_memory_bank_casebank_FMB_adversarial.jsonl",
            
        },
        "dataset": "train",
    }
}
