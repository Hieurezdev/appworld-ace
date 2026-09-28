local project_home_path = std.extVar("APPWORLD_PROJECT_PATH");
local experiment_playbooks_path = project_home_path + "/experiments/playbooks";
local adaptation_base = import "ACE_offline_with_GT_curator_operations_FMB_improved.jsonnet";
local normal_evaluation_base =
    import "ACE_offline_with_GT_curator_operations_FMB_improved_evaluation.jsonnet";
local challenge_evaluation_base =
    import "ACE_offline_with_GT_curator_operations_FMB_improved_evaluation_challenge.jsonnet";

local playbook_path(label) =
    experiment_playbooks_path
    + "/appworld_offline_lifecycle_fmb_topk_"
    + label
    + "_playbook.txt";

local memory_bank_path(label) =
    experiment_playbooks_path
    + "/failure_memory_bank_lifecycle_fmb_topk_"
    + label
    + ".jsonl";

{
    adaptation(top_k, label):
        adaptation_base + {
            config+: {
                agent+: {
                    trained_playbook_file_path: playbook_path(label),
                    reflector_memory_top_k: top_k,
                    reflector_memory_bank_file: memory_bank_path(label),
                },
            },
        },

    normal_evaluation(label):
        normal_evaluation_base + {
            config+: {
                agent+: {
                    trained_playbook_file_path: playbook_path(label),
                },
            },
        },

    challenge_evaluation(label):
        challenge_evaluation_base + {
            config+: {
                agent+: {
                    trained_playbook_file_path: playbook_path(label),
                },
            },
        },
}
