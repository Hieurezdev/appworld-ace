import os
import json
import tempfile
import pytest
from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

# Import agents
from appworld_experiments.code.ace.adaptation_react import SimplifiedReActStarAgent
from appworld_experiments.code.ace.evaluation_react import SimplifiedReActAgent
from appworld_experiments.code.ace.utils import retrieve_and_format_cases, MemoryRetrieverClassifier

# Mock World structures
class MockGroundTruth:
    required_apis = ["email_get", "contacts_search"]

class MockSupervisor:
    first_name = "Jane"
    last_name = "Doe"
    email = "jane@example.com"
    phone_number = "123-456-7890"

class MockTask:
    id = "test_task_id"
    instruction = "Find the email of John Doe"
    supervisor = MockSupervisor()
    app_descriptions = {"email": "Email App", "contacts": "Contacts App"}
    ground_truth = MockGroundTruth()

class MockWorld:
    task = MockTask()
    task_id = "test_task_id"
    output_misc_directory = "/tmp"
    output_logs_directory = "/tmp"


@pytest.fixture
def temp_casebank():
    cases = [
        {"case": "Find email of John", "plan": {"plan": [{"id": 1, "description": "Search John"}, {"id": 2, "description": "Get email"}]}, "case_label": "positive"},
        {"case": "Send email to Alice", "plan": {"plan": [{"id": 1, "description": "Send email"}]}, "case_label": "negative"},
        {"case": "Query contacts of Bob", "plan": "1. Search Bob\n2. Get contact info", "case_label": "positive"},
    ]
    with tempfile.NamedTemporaryFile(suffix=".jsonl", mode="w", delete=False) as f:
        for c in cases:
            f.write(json.dumps(c) + "\n")
        temp_path = f.name
    yield temp_path
    if os.path.exists(temp_path):
        os.remove(temp_path)

@pytest.fixture
def temp_files():
    # Create temp files for prompt template and playbook
    generator_prompt = "USER:\nTask: {{ input_str }}\nPlaybook:\n{{ playbook }}\n"
    reflector_prompt = "Reflector prompt template"
    curator_prompt = "Curator prompt template"
    playbook = "## Playbook Section\n[rule-1] Rule content"
    
    with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", delete=False) as gen_f, \
         tempfile.NamedTemporaryFile(suffix=".txt", mode="w", delete=False) as ref_f, \
         tempfile.NamedTemporaryFile(suffix=".txt", mode="w", delete=False) as cur_f, \
         tempfile.NamedTemporaryFile(suffix=".txt", mode="w", delete=False) as play_f, \
         tempfile.NamedTemporaryFile(suffix=".txt", mode="w", delete=False) as train_play_f:
        
        gen_f.write(generator_prompt)
        ref_f.write(reflector_prompt)
        cur_f.write(curator_prompt)
        play_f.write(playbook)
        
        paths = {
            "generator": gen_f.name,
            "reflector": ref_f.name,
            "curator": cur_f.name,
            "playbook": play_f.name,
            "trained_playbook": train_play_f.name,
        }
        
    yield paths
    for p in paths.values():
        if os.path.exists(p):
            os.remove(p)

def test_agent_instantiation_and_non_parametric_retrieval(temp_casebank, temp_files):
    # Test SimplifiedReActStarAgent (adaptation)
    generator_config = {
        "name": "Qwen/Qwen3-4B-Instruct-2507",
        "provider": "localhost",
        "localhost_url": "http://localhost:5000",
        "localhost_api_key": "not-needed",
    }
    
    # 1. Test instantiation with casebank flags
    agent = SimplifiedReActStarAgent(
        generator_prompt_file_path=temp_files["generator"],
        reflector_prompt_file_path=temp_files["reflector"],
        curator_prompt_file_path=temp_files["curator"],
        initial_playbook_file_path=temp_files["playbook"],
        trained_playbook_file_path=temp_files["trained_playbook"],
        generator_model_config=generator_config,
        reflector_model_config=generator_config,
        curator_model_config=generator_config,
        casebank_file_path=temp_casebank,
        casebank_top_k=2,
        casebank_retrieval_type="non-parametric",
        casebank_model="BAAI/bge-m3",
    )
    
    assert agent.casebank_file_path == temp_casebank
    assert agent.casebank_top_k == 2
    assert agent.casebank_retrieval_type == "non-parametric"
    
    # 2. Test initialize and retrieval logic
    world = MockWorld()
    agent.initialize(world)
    
    # The playbook string should contain the retrieved cases since {{ casebank }} is missing from the template
    # (falls back to appending)
    assert "### CASEBANK BEGIN" in agent.messages[0]["content"]
    assert "### CASEBANK END" in agent.messages[0]["content"]
    assert "Find email of John" in agent.messages[0]["content"]

def test_agent_parametric_retrieval_fallback(temp_casebank, temp_files):
    # Test fallback to non-parametric when retriever model path is invalid
    generator_config = {
        "name": "Qwen/Qwen3-4B-Instruct-2507",
        "provider": "localhost",
        "localhost_url": "http://localhost:5000",
        "localhost_api_key": "not-needed",
    }
    
    agent = SimplifiedReActStarAgent(
        generator_prompt_file_path=temp_files["generator"],
        reflector_prompt_file_path=temp_files["reflector"],
        curator_prompt_file_path=temp_files["curator"],
        initial_playbook_file_path=temp_files["playbook"],
        trained_playbook_file_path=temp_files["trained_playbook"],
        generator_model_config=generator_config,
        reflector_model_config=generator_config,
        curator_model_config=generator_config,
        casebank_file_path=temp_casebank,
        casebank_top_k=2,
        casebank_retrieval_type="parametric",
        casebank_retriever_model_path="/invalid/path/best.pt",
        casebank_model="BAAI/bge-m3",
    )
    
    # Fallback to non-parametric should occur during init
    assert agent.casebank_retrieval_type == "non-parametric"

@patch("torch.load")
@patch("transformers.AutoModel.from_pretrained")
@patch("transformers.AutoTokenizer.from_pretrained")
@patch.object(MemoryRetrieverClassifier, "load_state_dict")
def test_agent_parametric_retrieval_success(mock_load_state, mock_tokenizer, mock_automodel, mock_load, temp_casebank, temp_files):
    generator_config = {
        "name": "Qwen/Qwen3-4B-Instruct-2507",
        "provider": "localhost",
        "localhost_url": "http://localhost:5000",
        "localhost_api_key": "not-needed",
    }
    
    # Setup mocks for transformers and torch
    mock_bert = MagicMock()
    mock_bert.config.hidden_size = 768
    
    mock_bert_out = MagicMock()
    mock_bert_out.last_hidden_state = torch.randn(32, 256, 768)
    mock_bert.return_value = mock_bert_out
    mock_automodel.return_value = mock_bert
    
    # Mock tokenizer output
    mock_tokenizer_instance = MagicMock()
    mock_tokenizer_instance.return_value = {
        "input_ids": torch.ones(32, 256, dtype=torch.long),
        "attention_mask": torch.ones(32, 256, dtype=torch.long)
    }
    mock_tokenizer.return_value = mock_tokenizer_instance
    
    mock_load.return_value = {}  # Mock loaded state dict
    
    with tempfile.NamedTemporaryFile(suffix=".pt") as temp_ckpt:
        agent = SimplifiedReActStarAgent(
            generator_prompt_file_path=temp_files["generator"],
            reflector_prompt_file_path=temp_files["reflector"],
            curator_prompt_file_path=temp_files["curator"],
            initial_playbook_file_path=temp_files["playbook"],
            trained_playbook_file_path=temp_files["trained_playbook"],
            generator_model_config=generator_config,
            reflector_model_config=generator_config,
            curator_model_config=generator_config,
            casebank_file_path=temp_casebank,
            casebank_top_k=2,
            casebank_retrieval_type="parametric",
            casebank_retriever_model_path=temp_ckpt.name,
            casebank_model="BAAI/bge-m3",
        )
        
        assert agent.casebank_retrieval_type == "parametric"
        assert agent.casebank_retriever_model is not None
        
        # Test initialization with parametric retriever
        world = MockWorld()
        
        # Mock classifier forward to return logits of shape (32, 2)
        # Note: 3 cases in the casebank, so batch size of logits will be 3
        mock_logits = torch.randn(3, 2)
        with patch.object(agent.casebank_retriever_model, "forward", return_value=mock_logits):
            agent.initialize(world)
            
        assert "### CASEBANK BEGIN" in agent.messages[0]["content"]

def test_evaluation_agent_instantiation(temp_casebank, temp_files):
    # Test SimplifiedReActAgent (evaluation)
    generator_config = {
        "name": "Qwen/Qwen3-4B-Instruct-2507",
        "provider": "localhost",
        "localhost_url": "http://localhost:5000",
        "localhost_api_key": "not-needed",
    }
    
    agent = SimplifiedReActAgent(
        generator_prompt_file_path=temp_files["generator"],
        trained_playbook_file_path=temp_files["playbook"],  # Evaluation needs existing trained playbook
        generator_model_config=generator_config,
        casebank_file_path=temp_casebank,
        casebank_top_k=2,
        casebank_retrieval_type="non-parametric",
        casebank_model="BAAI/bge-m3",
    )
    
    assert agent.casebank_file_path == temp_casebank
    assert agent.casebank_top_k == 2
    
    world = MockWorld()
    agent.initialize(world)
    
    assert "### CASEBANK BEGIN" in agent.messages[0]["content"]
