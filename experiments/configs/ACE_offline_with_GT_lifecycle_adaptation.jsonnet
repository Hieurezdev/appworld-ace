// Lifecycle Curator ablation config.  All flags are independent: enable only
// the operation being evaluated, or set use_lifecycle_curator for the full set.
local base = import "ACE_offline_with_GT_adaptation.jsonnet";

base + {
    config+: {
        agent+: {
            // Legacy ACE remains ADD-only when every flag below is false.
            "use_lifecycle_curator": true,
            "use_curator_update": false,
            "use_curator_delete": false,
            "delete_harmful_margin": 4,
            "delete_min_harmful": 3,
            // Full lifecycle runs conservative zero-evidence pruning every 50 tasks.
            "prune_unused_bullets": true,
            "prune_unused_interval": 50,
            "use_curator_merge": false,
            "use_curator_create_meta": false,

            // Automatic post-Curator hygiene, independent of Curator MERGE.
            "use_bulletpoint_analyzer": false,
            "bulletpoint_analyzer_threshold": 0.90,
            // When enabled, this makes BulletpointAnalyzer cluster with
            // DBSCAN instead of legacy pairwise similarity.
            "use_dbscan_merge": false,
            // DBSCAN proposals for evidence-aware Curator MERGE only; this
            // never mutates the Playbook by itself.
            "use_dbscan_merge_candidates": false,
            "dbscan_eps": 0.12,
            "dbscan_min_samples": 2,
        },
    },
}
