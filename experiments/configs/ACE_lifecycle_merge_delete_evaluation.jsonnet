// Evaluate the playbook trained with ADD + DELETE + MERGE.
local project_home_path = std.extVar("APPWORLD_PROJECT_PATH");
local experiment_playbooks_path = project_home_path + "/experiments/playbooks";
local evaluation_base = import "ACE_offline_with_GT_curator_operations_evaluation.jsonnet";

evaluation_base + {
    config+: {
        agent+: {
            trained_playbook_file_path:
                experiment_playbooks_path + "/appworld_offline_lifecycle_merge_delete_playbook.txt",
            appworld_config+: {
                remote_environment_url: "http://0.0.0.0:8000",
                remote_apis_url: "http://0.0.0.0:9000",
                timeout_seconds: 120,
            },
        },
    },
}
