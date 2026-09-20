// Effective operations: ADD (default) + UPDATE + MERGE.
local project_home_path = std.extVar("APPWORLD_PROJECT_PATH");
local experiment_prompts_path = project_home_path + "/experiments/prompts";
local experiment_playbooks_path = project_home_path + "/experiments/playbooks";
local adaptation_base = import "ACE_offline_with_GT_adaptation.jsonnet";

adaptation_base + {
    config+: {
        agent+: {
            use_lifecycle_curator: false,
            use_curator_update: true,
            use_curator_delete: false,
            use_curator_merge: true,
            use_curator_create_meta: false,
            prune_unused_bullets: false,
            use_dbscan_merge_candidates: true,
            use_dbscan_merge: false,
            dbscan_eps: 0.12,
            dbscan_min_samples: 2,
            delete_harmful_margin: 4,
            delete_min_harmful: 3,
            prune_unused_interval: 50,
            use_bulletpoint_analyzer: false,
            curator_prompt_file_path:
                experiment_prompts_path + "/appworld_react_curator_prompt.txt",
            trained_playbook_file_path:
                experiment_playbooks_path + "/appworld_offline_lifecycle_update_merge_playbook.txt",
            appworld_config+: {
                remote_environment_url: "http://0.0.0.0:8000",
                remote_apis_url: "http://0.0.0.0:9000",
                timeout_seconds: 120,
            },
        },
    },
}
