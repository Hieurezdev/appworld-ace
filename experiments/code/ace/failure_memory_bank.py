"""Persistent failure memory with legacy and evidence-verified retrieval modes."""

from __future__ import annotations

import json
import os
import re
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


VERIFIED_FAILURE_TYPES = {
    "PLAYBOOK_GAP",
    "PLAYBOOK_MISAPPLICATION",
    "REASONING_ERROR",
    "RETRIEVAL_ERROR",
    "VERIFICATION_ERROR",
    "INSTRUCTION_FOLLOWING_ERROR",
    "TOOL_SELECTION_ERROR",
    "TOOL_ARGUMENT_ERROR",
}

DIAGNOSTIC_ONLY_FAILURE_TYPES = {
    "MODEL_FORMAT_ERROR",
    "EXECUTION_ERROR",
    "ENVIRONMENT_ERROR",
    "INVALID_ATTACK",
    "AMBIGUOUS_INPUT",
}


class FailureMemoryBank:
    """Persistent AppWorld failure bank.

    ``legacy`` retains the original semantic Top-K API. ``verified`` applies an
    evidence gate and staged retrieval: eligibility filtering, semantic
    candidate generation, then lexical/root-cause and usefulness reranking.
    """

    def __init__(
        self,
        bank_file_path: str,
        top_k: int = 3,
        model_name: str = "BAAI/bge-m3",
        sentence_transformer: "SentenceTransformer | None" = None,
        mode: str = "legacy",
        min_verifier_confidence: float = 0.8,
        min_retrieval_score: float = 0.2,
        candidate_multiplier: int = 4,
    ) -> None:
        if mode not in {"legacy", "verified"}:
            raise ValueError("FMB mode must be 'legacy' or 'verified'")
        self.bank_file_path = bank_file_path
        self.top_k = top_k
        self.model_name = model_name
        self.mode = mode
        self.min_verifier_confidence = min_verifier_confidence
        self.min_retrieval_score = min_retrieval_score
        self.candidate_multiplier = max(1, candidate_multiplier)
        os.makedirs(os.path.dirname(os.path.abspath(bank_file_path)), exist_ok=True)
        self.event_log_path = os.path.join(
            os.path.dirname(os.path.abspath(bank_file_path)),
            "failure_memory_events.jsonl",
        )
        self._model = sentence_transformer or self._load_model(model_name)
        self._entries = self._load_entries()
        self._next_id = self._find_next_id()
        self._log(
            "initialized",
            {
                "bank_file_path": bank_file_path,
                "top_k": top_k,
                "min_verifier_confidence": min_verifier_confidence,
                "min_retrieval_score": min_retrieval_score,
                "candidate_multiplier": self.candidate_multiplier,
            },
        )
        print(
            f"[FMB] Initialised mode={mode} with {len(self._entries)} entries "
            f"from '{bank_file_path}'."
        )

    def _log(self, event: str, payload: dict[str, Any]) -> None:
        record = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "event": event,
            "mode": self.mode,
            "bank_size": len(self._entries),
            **payload,
        }
        with open(self.event_log_path, "a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _find_next_id(self) -> int:
        highest = 0
        for entry in self._entries:
            match = re.fullmatch(r"fmb-(\d+)", str(entry.get("failure_id", "")))
            if match:
                highest = max(highest, int(match.group(1)))
        return highest + 1

    def _new_id(self) -> str:
        failure_id = f"fmb-{self._next_id:06d}"
        self._next_id += 1
        return failure_id

    @staticmethod
    def _fix_strategy(reflection: dict[str, Any]) -> str:
        return " | ".join(
            filter(
                None,
                [reflection.get("correct_approach", ""), reflection.get("key_insight", "")],
            )
        )

    def add(
        self,
        task_id: str,
        task_instruction: str,
        error_summary: str,
        reflection: dict[str, Any],
    ) -> str | None:
        """Original FMB write API, intentionally rejected in verified mode."""
        if self.mode == "verified":
            self._log("store_rejected", {"task_id": task_id, "reason": "legacy_add_in_verified_mode"})
            return None
        entry = {
            "failure_id": self._new_id(),
            "task_id": task_id,
            "task_instruction": task_instruction,
            "error_summary": error_summary,
            "reflection": reflection,
            "fix_strategy": self._fix_strategy(reflection),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        self._append(entry)
        self._log("failure_stored", {"failure": entry})
        return entry["failure_id"]

    def add_verified(
        self,
        *,
        task_id: str,
        task_instruction: str,
        error_summary: str,
        reflection: dict[str, Any],
        verification: dict[str, Any],
        evidence: list[str],
        failure_type: str = "PLAYBOOK_GAP",
        source: str = "appworld",
        playbook_refs: list[str] | None = None,
        vulnerability_id: str = "",
        candidate_id: str = "",
        curator_operations: list[dict[str, Any]] | None = None,
    ) -> str | None:
        """Store only evaluator/outcome-verifier grounded failures."""
        if self.mode != "verified":
            return self.add(task_id, task_instruction, error_summary, reflection)
        try:
            confidence = float(verification.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        failed_checks = []
        if verification.get("verified") is not True:
            failed_checks.append("not_verified")
        if confidence < self.min_verifier_confidence:
            failed_checks.append("confidence_below_threshold")
        if failure_type not in VERIFIED_FAILURE_TYPES:
            failed_checks.append("unsupported_failure_type")
        if not evidence:
            failed_checks.append("missing_evidence")
        accepted = not failed_checks
        self._log(
            "verification_gate",
            {
                "task_id": task_id,
                "source": source,
                "failure_type": failure_type,
                "confidence": confidence,
                "accepted": accepted,
                "failed_checks": failed_checks,
                "evidence_count": len(evidence),
            },
        )
        if not accepted:
            print(f"[FMB/verified] Rejected task={task_id}: {failed_checks}")
            return None

        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        operations = list(curator_operations or [])
        entry = {
            "schema_version": 2,
            "failure_id": self._new_id(),
            "task_id": task_id,
            "source": source,
            "failure_type": failure_type,
            "status": "curated" if operations else "verified",
            "task_instruction": task_instruction,
            "error_summary": error_summary,
            "reflection": reflection,
            "root_cause": reflection.get("root_cause_analysis", ""),
            "fix_strategy": self._fix_strategy(reflection),
            "evidence": list(evidence),
            "verification": {**verification, "confidence": confidence},
            "playbook_refs": list(playbook_refs or []),
            "vulnerability_id": vulnerability_id,
            "candidate_id": candidate_id,
            "curator_operations": operations,
            "curator_applied": bool(operations),
            "times_retrieved": 0,
            "times_helpful": 0,
            "times_harmful": 0,
            "timestamp": now,
            "updated_at": now,
        }
        self._append(entry)
        self._log("failure_stored", {"failure": entry})
        if operations:
            self._log(
                "curator_result_attached",
                {
                    "failure_id": entry["failure_id"],
                    "operation_count": len(operations),
                    "operations": operations,
                    "applied": True,
                },
            )
        return entry["failure_id"]

    @staticmethod
    def _tokens(text: str) -> set[str]:
        return set(re.findall(r"[a-z0-9_]+", (text or "").lower()))

    @classmethod
    def _lexical_score(cls, query: str, entry: dict[str, Any]) -> float:
        query_tokens = cls._tokens(query)
        reflection = entry.get("reflection", {})
        memory_tokens = cls._tokens(
            " ".join(
                [
                    entry.get("task_instruction", ""),
                    entry.get("error_summary", ""),
                    entry.get("root_cause", ""),
                    reflection.get("error_identification", ""),
                    reflection.get("key_insight", ""),
                ]
            )
        )
        if not query_tokens or not memory_tokens:
            return 0.0
        return len(query_tokens & memory_tokens) / len(query_tokens | memory_tokens)

    @staticmethod
    def _usefulness(entry: dict[str, Any]) -> float:
        helpful = int(entry.get("times_helpful", 0))
        harmful = int(entry.get("times_harmful", 0))
        return (helpful + 1) / (helpful + harmful + 2)

    def query(
        self,
        task_instruction: str,
        error_summary: str = "",
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        k = top_k or self.top_k
        if not self._entries:
            self._log("retrieval_skipped", {"reason": "empty_bank"})
            return []
        import numpy as np

        query_text = self._embed_key(task_instruction, error_summary)
        query_embedding = self._model.encode([query_text], normalize_embeddings=True)
        self._log("retrieval_started", {"query": query_text, "requested_top_k": k})

        if self.mode == "legacy":
            return self._legacy_query(query_text, query_embedding, k)

        eligible = [
            entry
            for entry in self._entries
            if entry.get("schema_version") == 2
            and entry.get("status") in {"verified", "curated", "validated"}
            and entry.get("failure_type") in VERIFIED_FAILURE_TYPES
            and entry.get("verification", {}).get("verified") is True
        ]
        self._log(
            "eligibility_filter",
            {"total": len(self._entries), "eligible": len(eligible)},
        )
        if not eligible:
            return []

        corpus_texts = [
            self._embed_key(
                entry["task_instruction"],
                " | ".join(
                    filter(
                        None,
                        [entry.get("error_summary", ""), entry.get("root_cause", ""), entry.get("fix_strategy", "")],
                    )
                ),
            )
            for entry in eligible
        ]
        corpus_embeddings = self._model.encode(corpus_texts, normalize_embeddings=True)
        semantic_scores = np.asarray(corpus_embeddings) @ np.asarray(query_embedding[0])
        pool_size = min(len(eligible), max(k, k * self.candidate_multiplier))
        pool_indices = np.argsort(-semantic_scores)[:pool_size]
        self._log(
            "semantic_candidates",
            {
                "pool_size": pool_size,
                "candidates": [
                    {
                        "failure_id": eligible[int(index)].get("failure_id"),
                        "semantic_score": float(semantic_scores[int(index)]),
                    }
                    for index in pool_indices
                ],
            },
        )

        reranked = []
        for index in pool_indices:
            entry = eligible[int(index)]
            semantic = float(semantic_scores[int(index)])
            lexical = self._lexical_score(query_text, entry)
            usefulness = self._usefulness(entry)
            score = 0.65 * semantic + 0.25 * lexical + 0.10 * usefulness
            if score >= self.min_retrieval_score:
                reranked.append((score, semantic, lexical, usefulness, entry))
        reranked.sort(key=lambda item: item[0], reverse=True)
        self._log(
            "candidates_reranked",
            {
                "minimum_score": self.min_retrieval_score,
                "candidates": [
                    {
                        "failure_id": entry.get("failure_id"),
                        "retrieval_score": score,
                        "semantic_score": semantic,
                        "lexical_score": lexical,
                        "usefulness_score": usefulness,
                    }
                    for score, semantic, lexical, usefulness, entry in reranked
                ],
            },
        )
        results = []
        for rank, (score, semantic, lexical, _, entry) in enumerate(reranked[:k], start=1):
            entry["times_retrieved"] = int(entry.get("times_retrieved", 0)) + 1
            result = dict(entry)
            result.update(
                {
                    "rank": rank,
                    "_score": round(score, 4),
                    "semantic_score": semantic,
                    "lexical_score": lexical,
                }
            )
            results.append(result)
        self._rewrite_entries()
        self._log(
            "retrieval_completed",
            {
                "results": [
                    {"failure_id": item.get("failure_id"), "rank": item["rank"], "score": item["_score"]}
                    for item in results
                ]
            },
        )
        return results

    def _legacy_query(self, query_text: str, query_embedding: Any, k: int) -> list[dict[str, Any]]:
        import faiss
        import numpy as np

        corpus_texts = [
            self._embed_key(entry["task_instruction"], entry.get("error_summary", ""))
            for entry in self._entries
        ]
        embeddings = self._model.encode(corpus_texts, normalize_embeddings=True)
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(np.asarray(embeddings, dtype=np.float32))
        scores, indices = index.search(np.asarray(query_embedding, dtype=np.float32), min(k, len(self._entries)))
        results = []
        for rank, (score, index_value) in enumerate(zip(scores[0], indices[0]), start=1):
            entry = dict(self._entries[int(index_value)])
            entry.update({"rank": rank, "_score": round(float(score), 4)})
            results.append(entry)
        self._log(
            "retrieval_completed",
            {"results": [{"failure_id": item.get("failure_id"), "rank": item["rank"], "score": item["_score"]} for item in results]},
        )
        return results

    def size(self) -> int:
        return len(self._entries)

    def _append(self, entry: dict[str, Any]) -> None:
        with open(self.bank_file_path, "a", encoding="utf-8") as file:
            file.write(json.dumps(entry, ensure_ascii=False) + "\n")
        self._entries.append(entry)

    def _rewrite_entries(self) -> None:
        temporary_path = self.bank_file_path + ".tmp"
        with open(temporary_path, "w", encoding="utf-8") as file:
            for entry in self._entries:
                file.write(json.dumps(entry, ensure_ascii=False) + "\n")
        os.replace(temporary_path, self.bank_file_path)

    @staticmethod
    def _embed_key(instruction: str, error_summary: str) -> str:
        return " | ".join(filter(None, [instruction.strip(), (error_summary or "").strip()]))

    def _load_entries(self) -> list[dict[str, Any]]:
        if not os.path.exists(self.bank_file_path):
            return []
        entries = []
        with open(self.bank_file_path, "r", encoding="utf-8") as file:
            for line in file:
                try:
                    if line.strip():
                        entries.append(json.loads(line))
                except json.JSONDecodeError as error:
                    print(f"[FMB] Skipping malformed line: {error}")
        return entries

    @staticmethod
    def _load_model(model_name: str) -> "SentenceTransformer":
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            import subprocess

            subprocess.check_call(["uv", "pip", "install", "sentence-transformers", "faiss-cpu", "numpy"])
            from sentence_transformers import SentenceTransformer
        return SentenceTransformer(model_name)


def build_analogical_context(similar_cases: list[dict[str, Any]]) -> str:
    if not similar_cases:
        return "(No sufficiently relevant verified failure cases found.)"
    lines = [
        f"Retrieved {len(similar_cases)} evidence-grounded past failure case(s):",
        "",
    ]
    for index, case in enumerate(similar_cases, start=1):
        reflection = case.get("reflection", {})
        lines.extend(
            [
                f"--- Case {index} (id={case.get('failure_id', 'legacy')}, score={case.get('_score', '?')}) ---",
                f"Past Task: {case.get('task_instruction', 'N/A')[:300]}",
                f"Failure Type: {case.get('failure_type', 'legacy')}",
                f"Past Error: {case.get('error_summary', 'N/A')[:300]}",
                f"Root Cause: {str(case.get('root_cause') or reflection.get('root_cause_analysis') or 'N/A')[:300]}",
                f"Fix Strategy: {str(case.get('fix_strategy') or 'N/A')[:300]}",
                f"Verified Evidence: {case.get('evidence', [])}",
                f"Curator Operations: {case.get('curator_operations', [])}",
                "",
            ]
        )
    return "\n".join(lines)
