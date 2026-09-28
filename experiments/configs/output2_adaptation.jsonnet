// Rerun of full lifecycle operations + verified adversarial adaptation.
// Keep the trained playbook isolated from the original experiment.
local project_home_path = std.extVar("APPWORLD_PROJECT_PATH");
local experiment_playbooks_path = project_home_path + "/experiments/playbooks";
local base = import "ACE_lifecycle_all_adversarial_verified_adaptation.jsonnet";

base + {
    config+: {
        agent+: {
            trained_playbook_file_path:
                experiment_playbooks_path + "/appworld_output2_lifecycle_all_adversarial_verified_playbook.txt",
        },
    },
}
