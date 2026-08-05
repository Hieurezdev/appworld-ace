import json
from types import SimpleNamespace

from appworld_experiments.code.ace.adaptation_react import SimplifiedReActStarAgent


class FakeModel:
    def __init__(self, responses):
        self.responses = iter(responses)

    def generate(self, messages):
        return {"content": json.dumps(next(self.responses)), "cost": 0}


class FakeLogger:
    def show_message(self, **kwargs):
        return None


def make_agent(responses, min_confidence=0.7):
    agent = object.__new__(SimplifiedReActStarAgent)
    agent.adversarial_model = FakeModel(responses)
    agent.adversarial_mode = "improved"
    agent.adversarial_num_candidates = 2
    agent.adversarial_min_confidence = min_confidence
    agent.playbook = "[rule-1] Verify recipients before payment."
    agent.logger = FakeLogger()
    agent.world = SimpleNamespace(task_id="task-1", output_misc_directory=None)
    return agent


def test_improved_pipeline_selects_only_verified_candidate():
    responses = [
        {"vulnerabilities": [{"vulnerability_id": "v1", "description": "ambiguous recipient"}]},
        {
            "candidates": [
                {"candidate_id": "c1", "mock_query": "bad", "target_outcome": "x"},
                {"candidate_id": "c2", "mock_query": "good", "target_outcome": "y"},
            ]
        },
        {
            "verifications": [
                {"candidate_id": "c1", "valid": True, "confidence": 0.4, "target_correct": True, "feasible": True, "unambiguous": True, "safety_ok": True},
                {"candidate_id": "c2", "valid": True, "confidence": 0.9, "target_correct": True, "feasible": True, "unambiguous": True, "safety_ok": True},
            ]
        },
        {"selected_candidate_id": "c2", "selection": {"learning_value": 0.9}},
    ]
    agent = make_agent(responses)

    result = agent._improved_adversarial_call("task-1", "[]")

    assert result["candidate_id"] == "c2"
    assert result["verification"]["confidence"] == 0.9


def test_improved_pipeline_stops_when_all_candidates_are_rejected():
    responses = [
        {"vulnerabilities": [{"vulnerability_id": "v1", "description": "gap"}]},
        {"candidates": [{"candidate_id": "c1", "mock_query": "q", "target_outcome": "x"}]},
        {
            "verifications": [
                {"candidate_id": "c1", "valid": False, "confidence": 0.99, "target_correct": True, "feasible": True, "unambiguous": True, "safety_ok": True}
            ]
        },
    ]
    agent = make_agent(responses)

    assert agent._improved_adversarial_call("task-1", "[]") == {}


def test_outcome_requires_oracle_and_confidence():
    agent = make_agent(
        [{"exposed_vulnerability": True, "oracle_satisfied": False, "confidence": 0.95, "evidence": [], "reason": "not proven"}]
    )

    result = agent.adversarial_outcome_call({}, [], [])

    assert result["exposed_vulnerability"] is False
