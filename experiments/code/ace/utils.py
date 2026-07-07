#!/usr/bin/env python3
"""
==============================================================================
utils.py
==============================================================================

This file contains utility functions for the project.

"""

from datetime import datetime
import random
import re
import time
import json
import openai
import os


def extract_answer(response):
    """Extract final answer from JSON model response"""
    try:
        parsed = json.loads(response)
        return str(parsed.get("final_answer", "No final answer found"))
    except (json.JSONDecodeError, KeyError):
        # Fallback to old format if JSON parsing fails
        matches = re.findall(r"Finish\[(.*?)\]", response)
        if matches:
            return matches[-1]
        
        # Trying to get the final answer from JSON style response with regex matching 
        matches = re.findall(r'[\'"]final_answer[\'"]\s*:\s*[\'"]([^\'"]+)[\'"]', response)
        if matches:
            return matches[-1]
        
        return "No final answer found"
    

def get_section_slug(section_name):
    """Convert section name to slug format (3-5 chars)"""
    # Common section mappings - updated to match new playbook sections
    slug_map = {
        "strategies_and_hard_rules": "shr",
        "hard_rules": "hr",
        "strategies_and_insights": "si",
        "apis_to_use_for_specific_information": "api",
        "useful_code_snippets_and_templates": "code",
        "code_snippets_and_templates": "code",
        "common_mistakes_and_correct_strategies": "cms",
        "common_mistakes_to_avoid": "err",
        "problem_solving_heuristics_and_workflows": "psw",
        "problem_solving_heuristics": "prob",
        "verification_checklist": "vc",
        "troubleshooting_and_pitfalls": "ts",
        "others": "misc",
        "meta_strategies": "meta"
    }
    
    # Clean and convert to snake_case
    clean_name = section_name.lower().strip().replace(" ", "_").replace("&", "and").rstrip(":")
    
    if clean_name in slug_map:
        return slug_map[clean_name]
    
    # Generate slug from first letters
    words = clean_name.split("_")
    if len(words) == 1:
        return words[0][:4]
    else:
        return "".join(w[0] for w in words[:5])


def process_think_blocks(text):
    """
    Process <think>...</think> blocks in generated text by converting them to Python comments.
    
    For each line in <think>...</think>, make each line start with '# ' and keep its original 
    whitespace (don't strip it) so no python indentation errors occur.
    
    Args:
        text (str): The text containing <think>...</think> blocks
        
    Returns:
        str: The processed text with think blocks converted to comments
    """
    # Find all <think>...</think> blocks
    think_pattern = r'<think>(.*?)</think>'
    
    def replace_think_block(match):
        think_content = match.group(1)
        
        # If content is empty, return empty string
        if not think_content or not think_content.strip():
            return ''
        
        # Split by lines and add '# ' to each line while preserving original whitespace
        lines = think_content.split('\n')
        commented_lines = []
        
        for line in lines:
            # For completely empty lines, just add '#'
            if len(line) == 0:
                commented_lines.append('#')
            # For lines with only whitespace, preserve the whitespace but comment it
            elif not line.strip():
                commented_lines.append(f"#{line}")
            else:
                # For lines with content, add '# ' prefix while preserving indentation
                commented_lines.append(f"# {line}")
        
        # Join lines and ensure we don't have leading/trailing empty lines
        result = '\n'.join(commented_lines)
        
        # Remove leading and trailing empty comment lines
        result_lines = result.split('\n')
        # Remove leading empty comment lines
        while result_lines and result_lines[0] == '#':
            result_lines.pop(0)
        # Remove trailing empty comment lines  
        while result_lines and result_lines[-1] == '#':
            result_lines.pop()
            
        return '\n'.join(result_lines) if result_lines else ''
    
    # Replace all <think>...</think> blocks with commented versions
    processed_text = re.sub(think_pattern, replace_think_block, text, flags=re.DOTALL)
    return processed_text


# --- Memento Casebank retrieval utilities ---

import torch
import torch.nn as nn
from typing import Optional, Union, Tuple, List, Dict, Any

class MemoryRetrieverClassifier(nn.Module):
    def __init__(self, sentence_bert):
        super().__init__()
        hidden = sentence_bert.config.hidden_size
        self.sentence_bert = sentence_bert
        self.classifier = nn.Sequential(
            nn.Linear(hidden * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 2)
        )
    def forward(self, ids1, mask1, ids2, mask2):
        o1 = self.sentence_bert(ids1, attention_mask=mask1).last_hidden_state[:, 0]
        o2 = self.sentence_bert(ids2, attention_mask=mask2).last_hidden_state[:, 0]
        return self.classifier(torch.cat([o1, o2], dim=1))


def _parse_plan(plan_field: Union[str, dict, list, None]) -> Optional[Union[dict, list]]:
    if plan_field is None:
        return None
    if isinstance(plan_field, (dict, list)):
        return plan_field
    if isinstance(plan_field, str):
        s = plan_field.strip()
        if not s:
            return None
        try:
            return json.loads(s)
        except Exception:
            return {"plan": [{"description": s}]}
    return None


def _pretty_plan(plan_obj: Union[dict, list]) -> str:
    try:
        steps = []
        if isinstance(plan_obj, dict) and "plan" in plan_obj and isinstance(plan_obj["plan"], list):
            for item in plan_obj["plan"]:
                if isinstance(item, dict):
                    sid = item.get("id")
                    desc = item.get("description") or item.get("desc") or item.get("step") or str(item)
                    steps.append(f"{sid}. {desc}" if sid is not None else f"- {desc}")
                else:
                    steps.append(f"- {str(item)}")
        elif isinstance(plan_obj, list):
            for i, item in enumerate(plan_obj, 1):
                if isinstance(item, dict):
                    desc = item.get("description") or item.get("desc") or item.get("step") or str(item)
                    steps.append(f"{i}. {desc}")
                else:
                    steps.append(f"{i}. {str(item)}")
        else:
            return json.dumps(plan_obj, ensure_ascii=False)
        return "\n".join(steps) if steps else json.dumps(plan_obj, ensure_ascii=False)
    except Exception:
        return json.dumps(plan_obj, ensure_ascii=False)


def retrieve_and_format_cases(
    task_instruction: str,
    casebank_file_path: str,
    top_k: int,
    retrieval_type: str,
    sentence_transformer=None,
    retriever_model=None,
    tokenizer=None,
    device=None,
) -> str:
    import json
    import os
    import numpy as np

    if not os.path.exists(casebank_file_path):
        print(f"[Casebank] Warning: Casebank file not found at {casebank_file_path}")
        return ""

    # Load cases from JSONL
    cases = []
    with open(casebank_file_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                cases.append(json.loads(line))
            except Exception:
                continue

    if not cases:
        print("[Casebank] Warning: Casebank is empty")
        return ""

    # Process metadata and build pool
    processed_cases = []
    icl_pool = []
    for i, obj in enumerate(cases):
        case_q = obj.get("case") or obj.get("question") or obj.get("query")
        if not case_q:
            keys = list(obj.keys())
            case_q = obj[keys[0]] if keys else f"Case {i}"
        
        plan_val = obj.get("plan")
        reward = obj.get("reward")
        truth_label = obj.get("truth_label")
        case_label = obj.get("case_label")
        
        if case_label is None:
            if reward is not None:
                case_label = "positive" if reward == 1 else "negative"
            elif truth_label is not None:
                case_label = "positive" if truth_label == 1 or truth_label is True else "negative"
            else:
                case_label = "positive"
        
        processed_cases.append({
            "case": str(case_q),
            "plan": plan_val,
            "case_label": case_label
        })

        # Format icl text for parametric scoring
        parts = ["[CASE]", str(case_q)]
        if plan_val is not None:
            pobj = _parse_plan(plan_val)
            parts += ["[PLAN]", _pretty_plan(pobj) if pobj is not None else str(plan_val)]
        icl_pool.append("\n".join(parts).strip())

    top_k = min(top_k, len(processed_cases))
    retrieved_indices = []

    if retrieval_type == "parametric" and retriever_model is not None and tokenizer is not None:
        # Neural retriever scoring
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        retriever_model.eval()
        scores = []
        bs = 32
        with torch.no_grad():
            for idx in range(0, len(icl_pool), bs):
                sub_icl = icl_pool[idx : idx + bs]
                sub_nat = [task_instruction] * len(sub_icl)
                t1 = tokenizer(sub_icl, padding=True, truncation=True, max_length=256, return_tensors="pt")
                t2 = tokenizer(sub_nat, padding=True, truncation=True, max_length=256, return_tensors="pt")
                ids1, mask1 = t1["input_ids"].to(device), t1["attention_mask"].to(device)
                ids2, mask2 = t2["input_ids"].to(device), t2["attention_mask"].to(device)
                logits = retriever_model(ids1, mask1, ids2, mask2)
                probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().tolist()
                scores.extend(probs)
        
        ranked = list(zip(range(len(processed_cases)), scores))
        ranked.sort(key=lambda x: x[1], reverse=True)
        retrieved_indices = [idx for idx, score in ranked[:top_k]]
    else:
        # Non-parametric retrieval using SentenceTransformer & FAISS
        import faiss
        
        if sentence_transformer is None:
            print("[Casebank] Warning: sentence_transformer not provided for non-parametric mode. Skipping retrieval.")
            return ""
        
        questions = [c["case"] for c in processed_cases]
        query_emb = sentence_transformer.encode([task_instruction], normalize_embeddings=True)
        question_embs = sentence_transformer.encode(questions, normalize_embeddings=True)

        d = question_embs.shape[1]
        index = faiss.IndexFlatIP(d)
        index.add(np.array(question_embs, dtype=np.float32))

        D, I = index.search(np.array(query_emb, dtype=np.float32), top_k)
        retrieved_indices = I[0].tolist()

    # Format output
    positive_cases = []
    negative_cases = []

    for idx in retrieved_indices:
        case = processed_cases[idx]
        if case["case_label"] == "positive":
            positive_cases.append(case)
        else:
            negative_cases.append(case)

    prompt_parts = []
    
    if positive_cases:
        prompt_parts.append("### Positive Examples:")
        for i, case in enumerate(positive_cases, 1):
            pobj = _parse_plan(case["plan"])
            plan_str = _pretty_plan(pobj) if pobj is not None else str(case["plan"])
            prompt_parts.append(f"Example {i}:\nQuestion: {case['case']}\nPlan:\n{plan_str}\n")

    if negative_cases:
        prompt_parts.append("### Negative Examples:")
        for i, case in enumerate(negative_cases, 1):
            pobj = _parse_plan(case["plan"])
            plan_str = _pretty_plan(pobj) if pobj is not None else str(case["plan"])
            prompt_parts.append(f"Example {i}:\nQuestion: {case['case']}\nPlan:\n{plan_str}\n")

    if prompt_parts:
        return "### CASEBANK BEGIN\n" + "\n".join(prompt_parts) + "\n### CASEBANK END"
    return ""

