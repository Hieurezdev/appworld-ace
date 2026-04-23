"""
Failure Memory Bank (FMB)
=========================
Stores past failure cases from the Adaptation phase and supports Top-K
semantic retrieval (via BGE-M3 + FAISS) to enable Analogical Reflection:
  "Last time I saw a similar error, fix A was tried but failed — try fix B."

Storage format: JSONL (one JSON entry per line).
Index: rebuilt in-memory at query time (FAISS IndexFlatIP, cosine sim via
       normalized embeddings).  No persistent index file needed since the bank
       grows incrementally and is typically small relative to embedding time.
"""
from __future__ import annotations

import json
import os
import time
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


class FailureMemoryBank:
    """Manages a persistent JSONL bank of failure cases and supports semantic
    Top-K retrieval for analogical reflection.

    Parameters
    ----------
    bank_file_path : str
        Absolute path to the JSONL file (created automatically if missing).
    top_k : int
        Default number of similar cases to return from ``query()``.
    model_name : str
        HuggingFace model identifier for the embedding model
        (default: "BAAI/bge-m3").
    sentence_transformer : SentenceTransformer | None
        Pre-loaded SentenceTransformer instance.  Pass the one already loaded
        by the RAE subsystem to avoid loading the model twice.
    """

    def __init__(
        self,
        bank_file_path: str,
        top_k: int = 3,
        model_name: str = "BAAI/bge-m3",
        sentence_transformer: "SentenceTransformer | None" = None,
    ) -> None:
        self.bank_file_path = bank_file_path
        self.top_k = top_k
        self.model_name = model_name

        # Ensure parent directory exists
        os.makedirs(os.path.dirname(os.path.abspath(bank_file_path)), exist_ok=True)

        # Load embedding model (reuse shared instance if provided)
        if sentence_transformer is not None:
            self._model = sentence_transformer
            print(f"[FMB] Reusing existing SentenceTransformer ({model_name}) for FailureMemoryBank.")
        else:
            self._model = self._load_model(model_name)

        # Load existing entries
        self._entries: list[dict[str, Any]] = self._load_entries()
        print(f"[FMB] FailureMemoryBank initialised — {len(self._entries)} existing entries from '{bank_file_path}'.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(
        self,
        task_id: str,
        task_instruction: str,
        error_summary: str,
        reflection: dict[str, Any],
    ) -> None:
        """Append a new failure entry to the bank.

        Parameters
        ----------
        task_id : str
            Unique task identifier (e.g. "train_0042").
        task_instruction : str
            The natural-language task description (used for embedding).
        error_summary : str
            Short description of the error/failure (e.g. test report snippet
            or exception message).  Concatenated with instruction for richer
            embedding signal.
        reflection : dict
            Full JSON dict produced by the Reflector
            (keys: reasoning, error_identification, root_cause_analysis,
             correct_approach, key_insight).
        """
        # Derive a human-readable fix strategy from the reflection
        correct_approach = reflection.get("correct_approach", "")
        key_insight = reflection.get("key_insight", "")
        fix_strategy = " | ".join(filter(None, [correct_approach, key_insight]))

        entry: dict[str, Any] = {
            "task_id": task_id,
            "task_instruction": task_instruction,
            "error_summary": error_summary,
            "reflection": reflection,
            "fix_strategy": fix_strategy,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

        # Persist to JSONL
        with open(self.bank_file_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")

        self._entries.append(entry)
        print(f"[FMB] Added failure entry for task '{task_id}' (bank size: {len(self._entries)}).")

    def query(
        self,
        task_instruction: str,
        error_summary: str = "",
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """Return Top-K most similar past failure cases.

        Parameters
        ----------
        task_instruction : str
            Current task's natural-language description.
        error_summary : str
            Current error/failure summary (appended to instruction for richer
            query embedding).
        top_k : int | None
            Override default top_k if provided.

        Returns
        -------
        list[dict]
            Up to ``top_k`` entries ordered by descending similarity.
            Each entry has keys: task_id, task_instruction, error_summary,
            reflection, fix_strategy, timestamp, _score.
        """
        k = top_k if top_k is not None else self.top_k

        if not self._entries:
            print("[FMB] Memory bank is empty — skipping retrieval.")
            return []

        import faiss
        import numpy as np

        query_text = self._embed_key(task_instruction, error_summary)
        query_emb = self._model.encode([query_text], normalize_embeddings=True)

        corpus_texts = [
            self._embed_key(e["task_instruction"], e.get("error_summary", ""))
            for e in self._entries
        ]
        corpus_embs = self._model.encode(corpus_texts, normalize_embeddings=True)

        d = corpus_embs.shape[1]
        index = faiss.IndexFlatIP(d)
        index.add(np.array(corpus_embs, dtype=np.float32))

        actual_k = min(k, len(self._entries))
        D, I = index.search(np.array(query_emb, dtype=np.float32), actual_k)

        results = []
        for score, idx in zip(D[0].tolist(), I[0].tolist()):
            entry = dict(self._entries[idx])
            entry["_score"] = round(float(score), 4)
            results.append(entry)

        print(f"[FMB] Retrieved {len(results)} similar failure cases (top-{k}).")
        return results

    def size(self) -> int:
        """Return the current number of entries in the bank."""
        return len(self._entries)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _embed_key(instruction: str, error_summary: str) -> str:
        """Combine instruction + error summary into a single embedding text."""
        parts = [instruction.strip()]
        if error_summary and error_summary.strip():
            parts.append(error_summary.strip())
        return " | ".join(parts)

    def _load_entries(self) -> list[dict[str, Any]]:
        """Load all entries from the JSONL file.  Returns empty list if file missing."""
        if not os.path.exists(self.bank_file_path):
            return []
        entries: list[dict[str, Any]] = []
        with open(self.bank_file_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError as exc:
                        print(f"[FMB] Warning: skipping malformed JSONL line — {exc}")
        return entries

    @staticmethod
    def _load_model(model_name: str) -> "SentenceTransformer":
        """Load (or auto-install) the SentenceTransformer model."""
        try:
            import sentence_transformers  # noqa: F401
            import faiss  # noqa: F401
        except ImportError:
            import subprocess
            print("[FMB] Dependencies not found. Auto-installing sentence-transformers and faiss-cpu...")
            subprocess.check_call(["uv", "pip", "install", "sentence-transformers", "faiss-cpu", "numpy"])

        from sentence_transformers import SentenceTransformer
        print(f"[FMB] Loading embedding model '{model_name}'...")
        model = SentenceTransformer(model_name)
        print("[FMB] Embedding model loaded.")
        return model


# ---------------------------------------------------------------------------
# Utility: build the analogical context block injected into Reflector prompt
# ---------------------------------------------------------------------------

def build_analogical_context(similar_cases: list[dict]) -> str:
    """Format retrieved failure cases into a human-readable context block.

    The block is designed to guide the Reflector toward analogical reasoning:
    comparing the current failure with past ones and suggesting novel fixes
    when previous strategies did not work.

    Parameters
    ----------
    similar_cases : list[dict]
        Output of ``FailureMemoryBank.query()``.

    Returns
    -------
    str
        A formatted string ready to be injected into the Reflector prompt.
    """
    if not similar_cases:
        return "(No similar past failure cases found in memory bank.)"

    lines: list[str] = [
        f"The following {len(similar_cases)} similar past failure case(s) were retrieved from the Failure Memory Bank.",
        "Use these analogies to guide your reflection:",
        "",
    ]

    for i, case in enumerate(similar_cases, start=1):
        score = case.get("_score", "?")
        task_instr = case.get("task_instruction", "N/A")
        err_summary = case.get("error_summary", "N/A")
        reflection = case.get("reflection", {})
        fix_strategy = case.get("fix_strategy", "N/A")

        error_id = reflection.get("error_identification", "N/A")
        root_cause = reflection.get("root_cause_analysis", "N/A")

        lines += [
            f"--- Case {i} (similarity={score}) ---",
            f"  Past Task     : {task_instr[:200]}",
            f"  Past Error    : {err_summary[:300]}",
            f"  Error Type    : {error_id[:200]}",
            f"  Root Cause    : {root_cause[:200]}",
            f"  Fix Tried     : {fix_strategy[:300]}",
            "",
        ]

    lines += [
        "ANALOGY INSTRUCTIONS:",
        "- If the current failure resembles a past case where fix A was tried and still failed,",
        "  consider a different strategy (fix B) rather than repeating fix A.",
        "- Explicitly note which past case is most similar and how it informs your reflection.",
    ]

    return "\n".join(lines)
