// Evaluate the playbook trained with full lifecycle + verified FMB/adversarial.
local project_home_path = std.extVar("APPWORLD_PROJECT_PATH");
local experiment_playbooks_path = project_home_path + "/experiments/playbooks";
local evaluation_base = import "ACE_lifecycle_all_adversarial_verified_evaluation.jsonnet";

evaluation_base + {
    config+: {
        agent+: {
            trained_playbook_file_path:
                experiment_playbooks_path + "/appworld_offline_lifecycle_all_fmb_adversarial_verified_playbook.txt",
        },
    },
}
