import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from appworld_experiments.code.ace.adaptation_agent import StarAgent
from appworld_experiments.code.ace.adaptation_react import SimplifiedReActStarAgent


class RecordingBank:
    def __init__(self, mode="verified", top_k=7, model=None, path=""):
        self.mode = mode
        self.top_k = top_k
        self._model = model or object()
        self.bank_file_path = path
        self.calls = []

    def add(self, **kwargs):
        self.calls.append(("add", kwargs))
        return "legacy-1"

    def add_verified(self, **kwargs):
        self.calls.append(("add_verified", kwargs))
        return "verified-1"


def make_routing_agent():
    agent = object.__new__(SimplifiedReActStarAgent)
    agent.real_failure_memory = RecordingBank()
    agent.adversarial_failure_memory = RecordingBank()
    agent.failure_memory_bank = agent.real_failure_memory
    agent.test_report = json.dumps(
        {
            "exposed_vulnerability": True,
            "oracle_satisfied": True,
            "confidence": 1.0,
            "evidence": ["verified mismatch"],
        }
    )
    agent.last_evaluation_failed = True
    agent.current_adversarial_result = {"candidate_id": "candidate-1"}
    agent.world = SimpleNamespace(
        task_id="task-1",
        task=SimpleNamespace(instruction="Send a payment"),
    )
    agent.adversarial_mode = "improved"
    agent._lifecycle_log = lambda event: None
    return agent


def test_pre_train_correctness_is_frozen_after_first_evaluation():
    initially_correct = object.__new__(StarAgent)
    initially_correct.pre_train_was_correct = None
    assert initially_correct._record_pre_train_correctness(False) is False
    assert initially_correct._record_pre_train_correctness(True) is False
    assert initially_correct.pre_train_was_correct is True

    initially_wrong = object.__new__(StarAgent)
    initially_wrong.pre_train_was_correct = None
    assert initially_wrong._record_pre_train_correctness(True) is True
    assert initially_wrong._record_pre_train_correctness(False) is True
    assert initially_wrong.pre_train_was_correct is False


def test_initially_correct_sample_updates_only_bullet_counters(tmp_path):
    agent = object.__new__(SimplifiedReActStarAgent)
    agent.use_reflector = False
    agent.step_number = 1
    agent.playbook = (
        "## VERIFICATION CHECKLIST\n"
        "[verify-00001] helpful=0 harmful=0 :: Verify the recipient."
    )
    agent.trained_playbook_file_path = str(tmp_path / "playbook.txt")
    events = []
    agent._lifecycle_log = events.append

    operations = agent.curator_call(
        json.dumps(
            {
                "bullet_tags": [
                    {"id": "verify-00001", "tag": "helpful"},
                ]
            }
        ),
        allow_content_updates=False,
    )

    assert operations == []
    assert "helpful=1 harmful=0 :: Verify the recipient." in agent.playbook
    assert Path(agent.trained_playbook_file_path).read_text() == agent.playbook
    assert [event["event"] for event in events] == [
        "reflector_bullet_counters_updated",
        "curator_skipped_initial_success",
    ]


def test_split_banks_keep_full_top_k_and_share_encoder():
    shared_model = object()
    created = []

    def make_bank(**kwargs):
        model = kwargs["sentence_transformer"] or shared_model
        bank = RecordingBank(
            mode=kwargs["mode"],
            top_k=kwargs["top_k"],
            model=model,
            path=kwargs["bank_file_path"],
        )
        created.append(bank)
        return bank

    with (
        patch.object(StarAgent, "__init__", return_value=None),
        patch(
            "appworld_experiments.code.ace.adaptation_react.read_file",
            return_value="[rule-1] Verify recipients.",
        ),
        patch(
            "appworld_experiments.code.ace.adaptation_react.os.path.exists",
            return_value=True,
        ),
        patch(
            "appworld_experiments.code.ace.adaptation_react.FailureMemoryBank",
            side_effect=make_bank,
        ),
    ):
        agent = SimplifiedReActStarAgent(
            generator_prompt_file_path="generator.txt",
            reflector_prompt_file_path="reflector.txt",
            curator_prompt_file_path="curator.txt",
            initial_playbook_file_path="initial.txt",
            trained_playbook_file_path="trained.txt",
            reflector_memory_top_k=7,
            reflector_memory_bank_file="real.jsonl",
            adversarial_reflector_memory_bank_file="adversarial.jsonl",
            reflector_memory_mode="verified",
        )

    assert len(created) == 2
    assert created[0].top_k == 7
    assert created[1].top_k == 7
    assert created[0].bank_file_path == "real.jsonl"
    assert created[1].bank_file_path == "adversarial.jsonl"
    assert created[1]._model is created[0]._model
    assert agent.real_failure_memory is created[0]
    assert agent.adversarial_failure_memory is created[1]


def test_adversarial_failure_uses_exactly_one_learning_route():
    agent = make_routing_agent()
    reflection = json.dumps({"root_cause_analysis": "recipient was not verified"})
    curator_operation = {
        "type": "ADD",
        "section": "verification_checklist",
        "content": "Verify recipients.",
    }

    # A real Curator mutation consumes the adversarial signal.
    assert agent._persist_failure_memory(
        reflection,
        curator_operations=[curator_operation],
        content_updated=True,
    ) is None
    assert agent.real_failure_memory.calls == []
    assert agent.adversarial_failure_memory.calls == []

    # With no applied operation, the same kind of signal goes only to M_adv.
    assert agent._persist_failure_memory(
        reflection,
        curator_operations=[curator_operation],
        content_updated=False,
    ) == "verified-1"
    assert agent.real_failure_memory.calls == []
    assert len(agent.adversarial_failure_memory.calls) == 1
    _, payload = agent.adversarial_failure_memory.calls[0]
    assert payload["curator_operations"] == []


def test_real_failure_never_leaks_into_adversarial_bank():
    agent = make_routing_agent()
    agent.current_adversarial_result = None
    agent.test_report = "AppWorld evaluator mismatch"

    assert agent._persist_failure_memory(
        json.dumps({"root_cause_analysis": "wrong API argument"}),
        curator_operations=[],
        content_updated=False,
    ) == "verified-1"
    assert len(agent.real_failure_memory.calls) == 1
    assert agent.adversarial_failure_memory.calls == []
