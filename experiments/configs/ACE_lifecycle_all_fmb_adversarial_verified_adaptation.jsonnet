// Full lifecycle operations + verified FMB + verified adversarial pipeline.
// RAE remains disabled so this experiment isolates the requested combination.
local project_home_path = std.extVar("APPWORLD_PROJECT_PATH");
local experiment_playbooks_path = project_home_path + "/experiments/playbooks";
local adaptation_base = import "ACE_lifecycle_all_adversarial_verified_adaptation.jsonnet";

adaptation_base + {
    config+: {
        agent+: {
            trained_playbook_file_path:
                experiment_playbooks_path + "/appworld_offline_lifecycle_all_fmb_adversarial_verified_playbook.txt",

            // Evidence-verified Failure Memory Bank for Reflector retrieval.
            reflector_memory_top_k: 10,
            reflector_memory_bank_file:
                experiment_playbooks_path + "/failure_memory_bank_lifecycle_all_fmb_adversarial_verified_v2.jsonl",
            reflector_memory_mode: "verified",
            reflector_memory_min_confidence: 0.8,
            reflector_memory_min_retrieval_score: 0.2,
            reflector_memory_candidate_multiplier: 4,
        },
    },
}
