import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np


MODULE_PATH = (
    Path(__file__).parents[1] / "experiments" / "code" / "ace" / "failure_memory_bank.py"
)
SPEC = importlib.util.spec_from_file_location("failure_memory_bank", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)
FailureMemoryBank = MODULE.FailureMemoryBank


class FakeEncoder:
    vocabulary = ["email", "calendar", "contact", "wrong", "argument", "verify"]

    def encode(self, texts, normalize_embeddings=True):
        vectors = []
        for text in texts:
            lowered = text.lower()
            vector = np.array([lowered.count(token) for token in self.vocabulary], dtype=float)
            if not vector.any():
                vector[-1] = 1.0
            if normalize_embeddings:
                vector /= np.linalg.norm(vector)
            vectors.append(vector)
        return np.asarray(vectors)


class FailureMemoryBankTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.bank_path = str(Path(self.temp_dir.name) / "failure_memory_bank_v2.jsonl")
        self.bank = FailureMemoryBank(
            self.bank_path,
            mode="verified",
            sentence_transformer=FakeEncoder(),
            top_k=2,
            min_verifier_confidence=0.8,
            min_retrieval_score=0.0,
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def add_email_failure(self):
        return self.bank.add_verified(
            task_id="task-1",
            task_instruction="Send an email to the saved contact",
            error_summary="wrong email tool argument",
            reflection={"root_cause_analysis": "The contact email was not retrieved first"},
            verification={
                "verified": True,
                "confidence": 1.0,
                "oracle_type": "appworld_evaluator",
            },
            evidence=["AppWorld test failure: recipient mismatch"],
            failure_type="TOOL_ARGUMENT_ERROR",
            curator_operations=[{"type": "ADD", "section": "verification_checklist", "content": "Verify recipient"}],
        )

    def test_verified_gate_rejects_unverified_and_legacy_add(self):
        self.assertIsNone(self.bank.add("x", "task", "error", {}))
        self.assertIsNone(
            self.bank.add_verified(
                task_id="x",
                task_instruction="task",
                error_summary="error",
                reflection={},
                verification={"verified": False, "confidence": 1.0},
                evidence=["untrusted"],
            )
        )
        self.assertEqual(self.bank.size(), 0)

    def test_schema_curator_and_multistage_retrieval_are_persisted(self):
        failure_id = self.add_email_failure()
        self.assertEqual(failure_id, "fmb-000001")
        self.bank.add_verified(
            task_id="task-2",
            task_instruction="Create a calendar event",
            error_summary="calendar date was wrong",
            reflection={"root_cause_analysis": "calendar timezone"},
            verification={"verified": True, "confidence": 0.9, "oracle_type": "appworld_evaluator"},
            evidence=["event mismatch"],
            failure_type="REASONING_ERROR",
        )

        results = self.bank.query("email contact recipient", "wrong argument")
        self.assertEqual(results[0]["failure_id"], failure_id)
        self.assertEqual(results[0]["schema_version"], 2)
        self.assertTrue(results[0]["curator_applied"])

        stored = [json.loads(line) for line in Path(self.bank_path).read_text().splitlines()]
        self.assertEqual(stored[0]["times_retrieved"], 1)
        events = [
            json.loads(line)["event"]
            for line in (Path(self.temp_dir.name) / "failure_memory_events.jsonl").read_text().splitlines()
        ]
        for expected in (
            "verification_gate",
            "failure_stored",
            "curator_result_attached",
            "eligibility_filter",
            "semantic_candidates",
            "candidates_reranked",
            "retrieval_completed",
        ):
            self.assertIn(expected, events)


if __name__ == "__main__":
    unittest.main()
