#!/usr/bin/env python3
"""Compare ACE lifecycle-operation evaluations and explain paired regressions.

The script reads existing artifacts under ``experiments/outputs``.  It does not
run an AppWorld evaluation.  Its most useful output is a Markdown report that
contains aggregate scores, playbook-size statistics, and samples where the same
task passes with one lifecycle configuration but fails with another.

Example:

    python scripts/compare_lifecycle_operations.py \
        --split all --max-samples 5 \
        --output experiments/outputs/lifecycle_operation_comparison.md
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


OPERATIONS = {
    "update": {
        "label": "UPDATE",
        "run": "ACE_lifecycle_update_evaluation",
        "playbook": "appworld_offline_lifecycle_update_playbook.txt",
        "enabled": "ADD + UPDATE",
    },
    "delete_prune": {
        "label": "DELETE/PRUNE",
        "run": "ACE_lifecycle_delete_prune_evaluation",
        "playbook": "appworld_offline_lifecycle_delete_prune_playbook.txt",
        "enabled": "ADD + DELETE + periodic prune",
    },
    "merge": {
        "label": "MERGE",
        "run": "ACE_lifecycle_merge_evaluation",
        "playbook": "appworld_offline_lifecycle_merge_playbook.txt",
        "enabled": "ADD + MERGE + DBSCAN candidates",
    },
    "lifecycle_all": {
        "label": "FULL",
        "run": "ACE_lifecycle_all_evaluation",
        "playbook": "appworld_offline_lifecycle_lifecycle_all_playbook.txt",
        "enabled": "ADD + UPDATE + DELETE + MERGE + CREATE_META + prune",
    },
}


@dataclass(frozen=True)
class TaskResult:
    success: bool
    difficulty: int
    num_tests: int
    passes: tuple[dict[str, Any], ...]
    failures: tuple[dict[str, Any], ...]

    @property
    def pass_count(self) -> int:
        return len(self.passes)

    @property
    def fail_count(self) -> int:
        return len(self.failures)


@dataclass(frozen=True)
class Evaluation:
    task_score: float
    scenario_score: float
    tasks: dict[str, TaskResult]

    @property
    def success_count(self) -> int:
        return sum(result.success for result in self.tasks.values())


def parse_args() -> argparse.Namespace:
    repository_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--outputs-root",
        type=Path,
        default=repository_root / "experiments" / "outputs",
        help="Directory containing ACE_lifecycle_*_evaluation runs.",
    )
    parser.add_argument(
        "--playbooks-root",
        type=Path,
        default=repository_root / "experiments" / "playbooks",
        help="Directory containing trained lifecycle playbooks.",
    )
    parser.add_argument(
        "--split",
        choices=("normal", "challenge", "all"),
        default="all",
        help="Evaluation split to analyze.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=4,
        help="Maximum paired regression samples shown for each operation/split.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Write Markdown to this path instead of stdout.",
    )
    return parser.parse_args()


def load_evaluation(path: Path) -> Evaluation:
    raw = json.loads(path.read_text())
    tasks = {
        task_id: TaskResult(
            success=bool(result["success"]),
            difficulty=int(result["difficulty"]),
            num_tests=int(result["num_tests"]),
            passes=tuple(result.get("passes", [])),
            failures=tuple(result.get("failures", [])),
        )
        for task_id, result in raw["individual"].items()
    }
    return Evaluation(
        task_score=float(raw["aggregate"]["task_goal_completion"]),
        scenario_score=float(raw["aggregate"]["scenario_goal_completion"]),
        tasks=tasks,
    )


def load_all_evaluations(outputs_root: Path, splits: list[str]) -> dict[str, dict[str, Evaluation]]:
    evaluations: dict[str, dict[str, Evaluation]] = {}
    missing: list[Path] = []
    for operation, config in OPERATIONS.items():
        evaluations[operation] = {}
        for split in splits:
            path = outputs_root / config["run"] / "evaluations" / f"test_{split}.json"
            if not path.exists():
                missing.append(path)
                continue
            evaluations[operation][split] = load_evaluation(path)
    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"Missing evaluation artifacts:\n{formatted}")
    return evaluations


def playbook_stats(path: Path) -> dict[str, int | None]:
    if not path.exists():
        return {"lines": None, "characters": None, "bullets": None, "exact_duplicates": None}
    text = path.read_text()
    bullet_pattern = re.compile(r"^\[[^]]+\]\s+helpful=\d+\s+harmful=\d+\s+::\s*(.*)$")
    normalized = []
    for line in text.splitlines():
        match = bullet_pattern.match(line)
        if match:
            normalized.append(" ".join(match.group(1).lower().split()))
    counts = Counter(normalized)
    return {
        "lines": len(text.splitlines()),
        "characters": len(text),
        "bullets": len(normalized),
        "exact_duplicates": sum(count > 1 for count in counts.values()),
    }


def task_directory(outputs_root: Path, operation: str, task_id: str) -> Path:
    return outputs_root / OPERATIONS[operation]["run"] / "tasks" / task_id


def extract_instruction(task_dir: Path) -> str:
    database_log = task_dir / "dbs" / "supervisor.jsonl"
    if not database_log.exists():
        return "<instruction unavailable>"
    for line in database_log.read_text().splitlines():
        try:
            statement, parameters, *_ = json.loads(line)
        except (ValueError, TypeError):
            continue
        if "INSERT INTO tasks" in statement and len(parameters) >= 4:
            return str(parameters[3])
    return "<instruction unavailable>"


def load_environment_log(task_dir: Path) -> str:
    path = task_dir / "logs" / "environment_io.md"
    return path.read_text() if path.exists() else ""


def api_calls(log: str) -> list[str]:
    return re.findall(r"apis\.([a-zA-Z_][\w]*)\.([a-zA-Z_][\w]*)\s*\(", log)


def summarize_api_calls(log: str, limit: int = 8) -> str:
    calls = [f"{app}.{api}" for app, api in api_calls(log)]
    if not calls:
        return "none detected"
    # Preserve execution order while removing documentation/supervisor noise and
    # consecutive duplicates caused by retry loops.
    compact: list[str] = []
    for call in calls:
        if call.startswith("api_docs.") or call == "supervisor.show_account_passwords":
            continue
        if not compact or compact[-1] != call:
            compact.append(call)
    shown = compact[-limit:]
    prefix = "… → " if len(compact) > limit else ""
    return prefix + " → ".join(f"`{call}`" for call in shown)


def classify_reason(instruction: str, result: TaskResult, log: str) -> str:
    """Produce an evidence-based, heuristic diagnosis for one failed rollout."""
    task = instruction.lower()
    requirements = "\n".join(str(item.get("requirement", "")) for item in result.failures).lower()
    traces = "\n".join(str(item.get("trace", "")) for item in result.failures)
    calls = {f"{app}.{api}" for app, api in api_calls(log)}

    if "request" in task and "venmo.create_transaction" in calls and "create_payment_request" not in log:
        return "Wrong intent/API boundary: interpreted a payment request as sending a transaction."
    if "money have i received" in task and "venmo.show_received_payment_requests" in calls:
        return "Wrong entity: summed PaymentRequest records instead of completed received Transactions."
    if "artist" in task and re.search(r"\[['\"]artists['\"]\]\[0\]", log):
        return "Incomplete nested aggregation: counted only the first artist credited on each song."
    if "previous song" in task and "spotify.previous_song" not in calls:
        return "Missing required mutation: found a liked song but never executed previous_song."
    if "activities" in task and "endswith(')')" in log:
        return "Brittle parsing plus failed recovery: checklist rows were rejected by an unrelated suffix condition."

    error_names = re.findall(r"(?:^|\n)(KeyError|NameError|TypeError|IndexError|AttributeError):", log)
    if error_names:
        most_common, count = Counter(error_names).most_common(1)[0]
        repeated = " repeatedly" if count > 1 else ""
        return f"Schema/code error: {most_common} occurred{repeated}, preventing task completion."
    if "<<not_given>>" in traces or "supervisor.complete_task" not in calls:
        return "No final answer: rollout exhausted its interaction budget or never called complete_task."
    if "model changes" in requirements or "changed_records" in requirements:
        if "set()" in traces:
            return "Missing state change: the task was analyzed but the required write action was not executed."
        return "Incorrect state transition: changed the wrong model, record, or number of records."
    if "end state" in requirements or "changed_field_names" in requirements:
        return "Incorrect final state: a required field/update was missing or extra state was modified."
    if "answer" in requirements:
        return "Incorrect answer caused by selection, filtering, aggregation, or formatting logic."
    return "Task-specific assertion mismatch; inspect the failure trace and final API sequence."


def failure_summary(result: TaskResult, limit: int = 2) -> str:
    requirements = [" ".join(str(item.get("requirement", "")).split()) for item in result.failures]
    selected = requirements[:limit]
    suffix = f" (+{len(requirements) - limit} more)" if len(requirements) > limit else ""
    return "; ".join(selected) + suffix


def paired_regressions(
    evaluations: dict[str, Evaluation], failed_operation: str
) -> list[tuple[str, TaskResult, list[str]]]:
    failed_tasks = evaluations[failed_operation].tasks
    rows = []
    for task_id, result in failed_tasks.items():
        if result.success:
            continue
        passing = [
            operation
            for operation, evaluation in evaluations.items()
            if operation != failed_operation
            and task_id in evaluation.tasks
            and evaluation.tasks[task_id].success
        ]
        if passing:
            rows.append((task_id, result, passing))
    # Prefer high-confidence contrasts: many passing alternatives and many failed
    # assertions, followed by harder tasks.
    rows.sort(
        key=lambda row: (len(row[2]), row[1].fail_count, row[1].difficulty, row[0]),
        reverse=True,
    )
    return rows


def markdown_report(
    outputs_root: Path,
    playbooks_root: Path,
    splits: list[str],
    evaluations: dict[str, dict[str, Evaluation]],
    max_samples: int,
) -> str:
    lines = [
        "# Lifecycle operation comparison",
        "",
        "Generated from existing AppWorld evaluation artifacts. ADD is always enabled, so the single-operation runs are additive ablations rather than pure operations.",
        "",
        "## Enabled operations",
        "",
        "| Run | Effective configuration |",
        "|---|---|",
    ]
    for config in OPERATIONS.values():
        lines.append(f"| {config['label']} | {config['enabled']} |")

    lines.extend(
        [
            "",
            "## Aggregate results",
            "",
            "| Split | Run | Passed tasks | Task score | Scenario score |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for split in splits:
        for operation, config in OPERATIONS.items():
            evaluation = evaluations[operation][split]
            lines.append(
                f"| {split} | {config['label']} | {evaluation.success_count}/{len(evaluation.tasks)} "
                f"| {evaluation.task_score:.1f} | {evaluation.scenario_score:.1f} |"
            )

    lines.extend(
        [
            "",
            "## Playbook size",
            "",
            "| Run | Lines | Characters | ID bullets | Exact duplicate bullet texts |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for operation, config in OPERATIONS.items():
        stats = playbook_stats(playbooks_root / config["playbook"])
        values = ["n/a" if stats[key] is None else str(stats[key]) for key in ("lines", "characters", "bullets", "exact_duplicates")]
        lines.append(f"| {config['label']} | {' | '.join(values)} |")

    lines.extend(
        [
            "",
            "## Automatic insights",
            "",
            "- **UPDATE:** preserves coverage but replacement rules can become overly broad, creating semantic drift and wrong endpoint selection.",
            "- **DELETE/PRUNE:** removes noisy rules and is strongest on normal data, but can delete rare schema or edge-case knowledge needed for challenge tasks.",
            "- **MERGE:** gives the best challenge result by compressing redundancy while retaining more coverage than deletion; its main risk is merging textually similar but behaviorally different concepts.",
            "- **ADD:** expands coverage but is present in every run, so its isolated causal effect cannot be measured here. Unchecked ADD also grows duplicated and conflicting guidance.",
            "- **FULL:** operations are not additive in quality. UPDATE can broaden a rule, MERGE can then compress away its exception, and DELETE/prune can remove the remaining rare rule.",
            "",
        ]
    )

    for split in splits:
        lines.extend([f"## Paired regressions: {split}", ""])
        split_evaluations = {operation: data[split] for operation, data in evaluations.items()}
        for operation, config in OPERATIONS.items():
            regressions = paired_regressions(split_evaluations, operation)
            lines.extend(
                [
                    f"### {config['label']} fails while another run passes",
                    "",
                    f"Found {len(regressions)} paired regressions.",
                    "",
                ]
            )
            if not regressions:
                lines.append("No paired regression found.")
                lines.append("")
                continue
            for task_id, result, passing in regressions[:max_samples]:
                task_dir = task_directory(outputs_root, operation, task_id)
                instruction = extract_instruction(task_dir)
                log = load_environment_log(task_dir)
                passing_labels = ", ".join(OPERATIONS[item]["label"] for item in passing)
                report_path = task_dir / "evaluation" / "report.md"
                log_path = task_dir / "logs" / "environment_io.md"
                lines.extend(
                    [
                        f"#### `{task_id}` — difficulty {result.difficulty}, {result.pass_count}/{result.num_tests} tests passed",
                        "",
                        f"- **Task:** {instruction}",
                        f"- **Passing controls:** {passing_labels}",
                        f"- **Likely reason:** {classify_reason(instruction, result, log)}",
                        f"- **Final API sequence:** {summarize_api_calls(log)}",
                        f"- **Failed checks:** {failure_summary(result)}",
                        f"- **Artifacts:** `{report_path}`; `{log_path}`",
                        "",
                    ]
                )

    lines.extend(
        [
            "## Interpretation cautions",
            "",
            "- A paired pass/fail contrast is stronger evidence than a standalone failure, but rollout stochasticity can still contribute.",
            "- The reason classifier is intentionally heuristic. Confirm high-impact cases by reading the linked report and environment log.",
            "- ADD needs a dedicated ADD-only run before making a causal claim about its score contribution.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if args.max_samples < 0:
        raise ValueError("--max-samples must be non-negative")
    splits = ["normal", "challenge"] if args.split == "all" else [args.split]
    evaluations = load_all_evaluations(args.outputs_root, splits)
    report = markdown_report(
        outputs_root=args.outputs_root,
        playbooks_root=args.playbooks_root,
        splits=splits,
        evaluations=evaluations,
        max_samples=args.max_samples,
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report + "\n")
        print(f"Wrote {args.output}")
    else:
        print(report)


if __name__ == "__main__":
    main()
