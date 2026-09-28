local project_home_path = std.extVar("APPWORLD_PROJECT_PATH");
local experiment_playbooks_path = project_home_path + "/experiments/playbooks";
local lifecycle = import "ACE_lifecycle_operation.libsonnet";

local playbook_path(interval) =
    experiment_playbooks_path
    + "/appworld_offline_lifecycle_add_delete_prune_"
    + std.toString(interval)
    + "_playbook.txt";

{
    adaptation(interval):
        lifecycle.adaptation("delete_prune") + {
            config+: {
                agent+: {
                    trained_playbook_file_path: playbook_path(interval),
                    prune_unused_interval: interval,
                },
            },
        },

    evaluation(interval):
        lifecycle.evaluation("delete_prune") + {
            config+: {
                agent+: {
                    trained_playbook_file_path: playbook_path(interval),
                },
            },
        },
}
