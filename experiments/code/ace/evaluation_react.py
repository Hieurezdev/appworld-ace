import copy
import json
import os
import re
from typing import Any

from jinja2 import Template

from appworld import AppWorld
from appworld.common.utils import read_file
from appworld_experiments.code.ace.evaluation_agent import Agent, ExecutionIO
from .utils import retrieve_and_format_cases, MemoryRetrieverClassifier

@Agent.register("ace_evaluation_react")
class SimplifiedReActAgent(Agent):
    def __init__(
        self,
        generator_prompt_file_path: str | None = None,
        trained_playbook_file_path: str | None = None,
        ignore_multiple_calls: bool = True,
        max_prompt_length: int | None = None,
        max_output_length: int = 400000,
        casebank_file_path: str | None = None,
        casebank_top_k: int | None = None,
        casebank_retrieval_type: str = "non-parametric",
        casebank_retriever_model_path: str | None = None,
        casebank_model: str = "BAAI/bge-m3",
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.generator_prompt_template = read_file(generator_prompt_file_path.replace("/", os.sep)).lstrip()
        self.trained_playbook_file_path = trained_playbook_file_path
        self.max_prompt_length = max_prompt_length
        self.max_output_length = max_output_length
        self.ignore_multiple_calls = ignore_multiple_calls
        self.partial_code_regex = r".*```python\n(.*)"
        self.full_code_regex = r"```python\n(.*?)```"

        if os.path.exists(trained_playbook_file_path):
            playbook = read_file(trained_playbook_file_path.replace("/", os.sep))
            self.playbook = playbook
        else:
            raise FileNotFoundError(f"playbook file not found at {trained_playbook_file_path}")

        self.casebank_file_path = casebank_file_path
        self.casebank_top_k = casebank_top_k
        self.casebank_retrieval_type = casebank_retrieval_type
        self.casebank_retriever_model_path = casebank_retriever_model_path
        self.casebank_model = casebank_model
        
        self.casebank_retriever_model = None
        self.casebank_tokenizer = None
        self.sentence_transformer = None

        # Load casebank retriever classifier model if parametric retrieval is active
        if self.casebank_top_k is not None and self.casebank_top_k > 0:
            if self.casebank_retrieval_type == "parametric":
                if not self.casebank_retriever_model_path:
                    print("[Casebank] Warning: casebank_retriever_model_path is not set. Falling back to non-parametric retrieval.")
                    self.casebank_retrieval_type = "non-parametric"
                else:
                    try:
                        import torch
                        from transformers import AutoTokenizer, AutoModel
                        print(f"Loading Casebank retriever model from {self.casebank_retriever_model_path}...")
                        self.casebank_tokenizer = AutoTokenizer.from_pretrained(self.casebank_model)
                        backbone = AutoModel.from_pretrained(self.casebank_model)
                        self.casebank_retriever_model = MemoryRetrieverClassifier(backbone)
                        self.casebank_retriever_model.load_state_dict(
                            torch.load(self.casebank_retriever_model_path, map_location="cpu")
                        )
                        device = "cuda" if torch.cuda.is_available() else "cpu"
                        self.casebank_retriever_model.to(device)
                        self.casebank_retriever_model.eval()
                        print("Casebank retriever model loaded successfully.")
                    except Exception as e:
                        print(f"Error loading parametric casebank retriever: {e}. Falling back to non-parametric.")
                        self.casebank_retrieval_type = "non-parametric"

            if self.casebank_retrieval_type == "non-parametric":
                try:
                    import sentence_transformers
                    import faiss
                except ImportError:
                    import subprocess
                    print("Casebank dependencies not found. Auto-installing sentence-transformers and faiss-cpu via uv pip...")
                    subprocess.check_call(["uv", "pip", "install", "sentence-transformers", "faiss-cpu", "numpy"])

                from sentence_transformers import SentenceTransformer
                print(f"Loading embedding model {self.casebank_model}...")
                self.sentence_transformer = SentenceTransformer(self.casebank_model)
                print("Embedding model loaded successfully.")

    def initialize(self, world: AppWorld):
        super().initialize(world)
        
        playbook_str = self.playbook
        
        # --- Case Bank Retrieval ---
        casebank_str = ""
        if self.casebank_file_path and self.casebank_top_k is not None and self.casebank_top_k > 0:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            casebank_str = retrieve_and_format_cases(
                task_instruction=world.task.instruction,
                casebank_file_path=self.casebank_file_path.replace("/", os.sep),
                top_k=self.casebank_top_k,
                retrieval_type=self.casebank_retrieval_type,
                sentence_transformer=self.sentence_transformer,
                retriever_model=self.casebank_retriever_model,
                tokenizer=self.casebank_tokenizer,
                device=device
            )
            
            if casebank_str:
                log_msg = f"✅ [Casebank] Retrieved similar cases from {self.casebank_file_path} using {self.casebank_retrieval_type} retrieval.\n"
                if hasattr(self, "logger") and self.logger:
                    self.logger.show_message(role="environment", message=log_msg, step_number=getattr(self, "step_number", 0))
                else:
                    print(log_msg)

        if casebank_str:
            if "{{ casebank }}" in self.generator_prompt_template:
                pass
            else:
                playbook_str = playbook_str + "\n\n" + casebank_str

        template = Template(self.generator_prompt_template)
        app_descriptions = json.dumps(
            [{"name": k, "description": v} for (k, v) in world.task.app_descriptions.items()],
            indent=1,
        )
        template_params = {
            "input_str": world.task.instruction,
            "main_user": world.task.supervisor,
            "app_descriptions": app_descriptions,
            "relevant_apis": str(world.task.ground_truth.required_apis),
            "playbook": playbook_str,
            "casebank": casebank_str,
        }
        output_str = template.render(template_params)
        output_str = self.truncate_input(output_str) + "\n\n"
        self.messages = self.text_to_messages(output_str)
        self.num_instruction_messages = len(self.messages)

    def next_execution_inputs_and_cost(
        self, last_execution_outputs: list[ExecutionIO], world_gt_code: str = None
    ) -> tuple[ExecutionIO, float, str | None]:
        if last_execution_outputs:
            assert (
                len(last_execution_outputs) == 1
            ), "React expects exactly one last_execution_output."
            last_execution_output_content = last_execution_outputs[0].content
            potential_new_line = ""
            last_execution_output_content = (
                "Output:\n```\n" + self.truncate_output(last_execution_output_content) + potential_new_line + "```\n\n"
            )
            self.messages.append({"role": "user", "content": last_execution_output_content})
        messages = self.trimmed_messages
        output = self.language_model.generate(messages=messages)
        ttft = output.get("ttft")
        tpot = output.get("tpot")
        if ttft is not None and tpot is not None:
            prompt_tokens = output.get("prompt_tokens", 0)
            completion_tokens = output.get("completion_tokens", 0)
            self.logger.show_message(
                role="environment",
                message=f"⏱️ [LLM Generation speed] TTFT: {ttft:.4f}s | TPOT: {tpot:.4f}s | Input Tokens: {prompt_tokens} | Output Tokens: {completion_tokens}",
                step_number=self.step_number
            )
        code, fixed_output_content = self.extract_code_and_fix_content(output["content"])
        self.messages.append({"role": "assistant", "content": fixed_output_content + "\n\n"})
        self.logger.show_message(
            role="agent", message=fixed_output_content, step_number=self.step_number
        )
        return [ExecutionIO(content=code)], output["cost"], None

    def extract_code_and_fix_content(self, text: str) -> tuple[str, str]:
        if text is None:
            return "", ""
        original_text = text
        output_code = ""
        match_end = 0
        # Handle multiple calls
        for re_match in re.finditer(self.full_code_regex, original_text, flags=re.DOTALL):
            code = re_match.group(1).strip()
            if self.ignore_multiple_calls:
                text = original_text[: re_match.end()]
                return code, text
            output_code += code + "\n"
            match_end = re_match.end()
        # Check for partial code match at end (no terminating ```)  following the last match
        partial_match = re.match(
            self.partial_code_regex, original_text[match_end:], flags=re.DOTALL
        )
        if partial_match:
            output_code += partial_match.group(1).strip()
            # Terminated due to stop condition; add stop condition to output
            if not text.endswith("\n"):
                text = text + "\n"
            text = text + "```"
        if len(output_code) == 0:
            return "", text
        else:
            return output_code, text

    def truncate_input(self, input_str: str) -> str:
        if self.max_prompt_length is None:
            return input_str
        max_prompt_length = self.max_prompt_length
        goal_index = input_str.rfind("Task:")
        if goal_index == -1:
            raise ValueError(f"No goal found in input string:\n{input_str}")
        next_new_line_index = input_str.find("\n", goal_index) + 1
        init_prompt = input_str[:next_new_line_index]
        prompt = input_str[next_new_line_index:]
        if len(init_prompt) > max_prompt_length:
            raise ValueError("Input prompt longer than max allowed length")
        if len(prompt) > max_prompt_length - len(init_prompt):
            new_prompt = prompt[-(max_prompt_length - len(init_prompt)) :]
            cmd_index = new_prompt.find("ASSISTANT:") if "ASSISTANT:" in new_prompt else 0
            prompt = "\n[TRIMMED HISTORY]\n\n" + new_prompt[cmd_index:]
        return init_prompt + prompt
    
    def truncate_output(self, execution_output_content: str) -> str:
        if len(execution_output_content) > 20000:
            execution_output_content = execution_output_content[:20000] + "\n[REST NOT SHOWN FOR BREVITY]"
        return execution_output_content

    def text_to_messages(self, input_str: str) -> list[dict]:
        messages_json = []
        last_start = 0
        for m in re.finditer("(USER|ASSISTANT|SYSTEM):\n", input_str, flags=re.IGNORECASE):
            last_end = m.span()[0]
            if len(messages_json) == 0:
                if last_end != 0:
                    raise ValueError(
                        f"Start of the prompt has no assigned role: {input_str[:last_end]}"
                    )
            else:
                messages_json[-1]["content"] = input_str[last_start:last_end]
            role = m.group(1).lower()
            messages_json.append({"role": role, "content": None})
            last_start = m.span()[1]
        messages_json[-1]["content"] = input_str[last_start:]
        return messages_json

    def messages_to_text(self, messages: list[dict]) -> str:
        output_str = ""
        for message in messages:
            role = message["role"]
            if role == "system":
                output_str += "SYSTEM:\n" + message["content"]
            if role == "assistant":
                output_str += "ASSISTANT:\n" + message["content"]
            elif role == "user":
                output_str += "USER:\n" + message["content"]
            else:
                raise ValueError(f"Unknown message role {role} in: {message}")
        return output_str

    @property
    def trimmed_messages(self) -> list[dict]:
        messages = copy.deepcopy(self.messages)
        pre_messages = messages[: self.num_instruction_messages - 1]
        post_messages = messages[self.num_instruction_messages - 1 :]
        output_str = self.messages_to_text(post_messages)
        remove_prefix = output_str[: output_str.index("Task: ") + 6]
        output_str = output_str.removeprefix(
            remove_prefix
        )  # not needed, it's only to match the original code
        observation_index = 0
        while len(output_str) > self.max_output_length:
            found_block = False
            # Dont remove observations from the last 5 blocks
            if observation_index < len(post_messages) - 5:
                # Find the next observation block to remove
                for message_index, message in enumerate(post_messages[observation_index:]):
                    # Only keep the code blocks and remove observations
                    if message["role"] == "user" and message["content"].startswith("Output:"):
                        message["content"] = "Output:\n```\n[NOT SHOWN FOR BREVITY]```\n\n"
                        found_block = True
                        observation_index += message_index + 1
                        break
                if not found_block:
                    observation_index = len(post_messages)
            # If no observation block left to trim, we need to start removing complete history blocks
            if not found_block and len(post_messages):
                first_post_message = copy.deepcopy(post_messages[0])
                if not first_post_message["content"].endswith("[TRIMMED HISTORY]\n\n"):
                    first_post_message["content"] += "[TRIMMED HISTORY]\n\n"
                post_messages = [first_post_message] + post_messages[2:]
                found_block = True
            if not found_block:
                raise ValueError(f"No blocks found to be removed!\n{post_messages}")
            output_str = self.messages_to_text(
                post_messages
            )  # not needed, it's only to match the original code
            output_str = output_str.removeprefix(remove_prefix)
        messages = pre_messages + post_messages
        return messages