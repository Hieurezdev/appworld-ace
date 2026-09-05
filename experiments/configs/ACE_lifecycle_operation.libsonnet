local project_home_path = std.extVar("APPWORLD_PROJECT_PATH");
local experiment_prompts_path = project_home_path + "/experiments/prompts";
local experiment_playbooks_path = project_home_path + "/experiments/playbooks";
local adaptation_base = import "ACE_offline_with_GT_adaptation.jsonnet";
local evaluation_base = import "ACE_offline_with_GT_curator_operations_evaluation.jsonnet";

local operation_settings = {
    update: {
        use_lifecycle_curator: false,
        use_curator_update: true,
        use_curator_delete: false,
        use_curator_merge: false,
        use_curator_create_meta: false,
        prune_unused_bullets: false,
        use_dbscan_merge_candidates: false,
    },
    delete_prune: {
        use_lifecycle_curator: false,
        use_curator_update: false,
        use_curator_delete: true,
        use_curator_merge: false,
        use_curator_create_meta: false,
        prune_unused_bullets: true,
        use_dbscan_merge_candidates: false,
    },
    merge: {
        use_lifecycle_curator: false,
        use_curator_update: false,
        use_curator_delete: false,
        use_curator_merge: true,
        use_curator_create_meta: false,
        prune_unused_bullets: false,
        use_dbscan_merge_candidates: true,
    },
    lifecycle_all: {
        use_lifecycle_curator: true,
        use_curator_update: false,
        use_curator_delete: false,
        use_curator_merge: false,
        use_curator_create_meta: false,
        prune_unused_bullets: true,
        use_dbscan_merge_candidates: true,
    },
};

local playbook_path(operation) =
    experiment_playbooks_path + "/appworld_offline_lifecycle_" + operation + "_playbook.txt";

{
    adaptation(operation):
        local settings = operation_settings[operation];
        adaptation_base + {
            config+: {
                agent+: settings + {
                    curator_prompt_file_path: experiment_prompts_path + "/appworld_react_curator_prompt.txt",
                    trained_playbook_file_path: playbook_path(operation),
                    delete_harmful_margin: 4,
                    delete_min_harmful: 3,
                    prune_unused_interval: 50,
                    use_bulletpoint_analyzer: false,
                    use_dbscan_merge: false,
                    dbscan_eps: 0.12,
                    dbscan_min_samples: 2,
                },
            },
        },

    evaluation(operation):
        evaluation_base + {
            config+: {
                agent+: {
                    trained_playbook_file_path: playbook_path(operation),
                },
            },
        },
}
