// Full lifecycle operations plus the improved, verifier-gated adversarial agent.
// ADD is always enabled. RAE and FMB are intentionally disabled so this run
// isolates the effect of verified adversarial training on the full lifecycle.
local project_home_path = std.extVar("APPWORLD_PROJECT_PATH");
local experiment_prompts_path = project_home_path + "/experiments/prompts";
local experiment_playbooks_path = project_home_path + "/experiments/playbooks";
local adaptation_base = import "ACE_offline_with_GT_adaptation.jsonnet";

adaptation_base + {
    config+: {
        agent+: {
            // ADD + UPDATE + DELETE + MERGE + CREATE_META.
            use_lifecycle_curator: true,
            use_curator_update: false,
            use_curator_delete: false,
            use_curator_merge: false,
            use_curator_create_meta: false,

            // Match the full-lifecycle pruning and merge-candidate settings.
            prune_unused_bullets: true,
            prune_unused_interval: 50,
            delete_harmful_margin: 4,
            delete_min_harmful: 3,
            use_bulletpoint_analyzer: false,
            use_dbscan_merge: false,
            use_dbscan_merge_candidates: true,
            dbscan_eps: 0.12,
            dbscan_min_samples: 2,

            // Improved adversarial pipeline with candidate and outcome verification.
            adversarial_model_config: adaptation_base.config.agent.generator_model_config,
            adversarial_prompt_file_path:
                experiment_prompts_path + "/appworld_react_adversarial_prompt.txt",
            use_hybrid_adversarial: true,
            adversarial_mode: "improved",
            adversarial_num_candidates: 5,
            adversarial_min_confidence: 0.8,

            // Keep retrieval/FMB off for an interpretable adversarial ablation.
            playbook_rae_top_k: null,
            reflector_memory_top_k: null,
            reflector_memory_bank_file: null,

            curator_prompt_file_path:
                experiment_prompts_path + "/appworld_react_curator_prompt.txt",
            trained_playbook_file_path:
                experiment_playbooks_path + "/appworld_offline_lifecycle_all_adversarial_verified_playbook.txt",
            appworld_config+: {
                remote_environment_url: "http://0.0.0.0:8000",
                remote_apis_url: "http://0.0.0.0:9000",
                timeout_seconds: 120,
            },
        },
    },
}
