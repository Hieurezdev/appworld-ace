local project_home_path = std.extVar("APPWORLD_PROJECT_PATH");
local experiment_playbooks_path = project_home_path + "/experiments/playbooks";
local lifecycle = import "ACE_lifecycle_operation.libsonnet";

local playbook_path(label) =
    experiment_playbooks_path
    + "/appworld_offline_lifecycle_curator_merge_dbscan_threshold_"
    + label
    + "_playbook.txt";

{
    adaptation(similarity_threshold, label):
        lifecycle.adaptation("merge") + {
            config+: {
                agent+: {
                    trained_playbook_file_path: playbook_path(label),
                    dbscan_eps: 1.0 - similarity_threshold,
                },
            },
        },

    evaluation(label):
        lifecycle.evaluation("merge") + {
            config+: {
                agent+: {
                    trained_playbook_file_path: playbook_path(label),
                },
            },
        },
}
