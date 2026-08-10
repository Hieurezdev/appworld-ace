# Failure Memory Bank v2

FMB v2 is opt-in. Existing configurations remain on `legacy`, preserving the
original JSONL schema and semantic Top-K behavior.

Enable the verified pipeline in an agent config:

```jsonnet
"reflector_memory_top_k": 10,
"reflector_memory_bank_file": experiment_playbooks_path + "/failure_memory_bank_v2.jsonl",
"reflector_memory_mode": "verified",
"reflector_memory_min_confidence": 0.8,
"reflector_memory_min_retrieval_score": 0.2,
"reflector_memory_candidate_multiplier": 4,
```

The verified mode stores only failures grounded by one of these methods:

- `appworld_evaluator` for original AppWorld tasks (confidence `1.0`);
- `adversarial_outcome_verifier` for improved generated attacks, when the
  candidate oracle is satisfied and the targeted vulnerability is exposed.

Legacy adversarial results, parsing errors, execution errors, and unsupported
failure types are logged but not admitted as learning memories. Each schema-v2
record contains evidence, verification metadata, root cause, playbook refs,
retrieval feedback counters, and the Curator operations applied to the playbook.

Retrieval runs four logged stages: eligibility filtering, semantic candidate
generation, lexical/root-cause plus usefulness reranking, and thresholded Top-K
selection. Detailed events are appended to `failure_memory_events.jsonl` beside
the configured bank file.

`ACE_offline_with_GT_adversarial_improved_adaptation.jsonnet` enables FMB v2.
Older FMB configs keep their original flag behavior.
