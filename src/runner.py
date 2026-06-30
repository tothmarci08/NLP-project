"""
Experiment runner: architecture-agnostic loop over (architecture, domain, difficulty) cells.

Records per-row: id, question, gold_answer, prediction, exact_match, f1 (HotpotQA),
total_input_tokens, total_output_tokens, total_steps, iteration_cap, domain, difficulty, architecture.

Results are written incrementally (one row at a time) so a crash mid-run loses nothing.
"""

from __future__ import annotations

import csv
import json
import time
import traceback
from pathlib import Path
from typing import Any, Callable

from src.datasets import load_dataset, Domain, Difficulty
from src.evaluators import score
from src.state import AgentState


RESULTS_DIR = Path("results/raw")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

FIELDNAMES = [
    "run_id", "architecture", "domain", "difficulty", "iteration_cap",
    "id", "question", "gold_answer",
    "raw_prediction",   # model's exact output, before any post-processing
    "prediction",       # scored value (= raw_prediction for GSM8K and short HotpotQA answers)
    "exact_match", "f1",
    "total_input_tokens", "total_output_tokens", "total_steps",
    "elapsed_seconds",
    "top_k",
    "error",
]

# Prefixes that indicate the model embedded its answer inside a sentence rather than
# outputting a bare span. Stripped (case-insensitively) from the last line of a
# verbose HotpotQA prediction before scoring.
_ANSWER_PREFIXES = (
    "the final answer is ",
    "final answer: ",
    "the answer is ",
    "answer: ",
    "so the answer is ",
    "therefore, the answer is ",
    "thus, the answer is ",
    "therefore ",
    "thus ",
)


def _clean_hotpotqa_prediction(text: str) -> str:
    """
    Extract a clean answer span from a potentially verbose HotpotQA prediction.

    Two passes:
    1. Always strip a known reasoning prefix from the start of the text
       ("The answer is Paris" -> "Paris", "Answer: Yes" -> "Yes").
    2. For long responses (>150 chars), also take the last non-empty line and
       strip any prefix from that line too — handles models that write a
       reasoning paragraph and then state the answer on the final line.

    Short, clean predictions (e.g. "Paris", "Yes") pass through both passes
    unchanged because no prefix matches and len <= 150.
    """
    text = text.strip()

    def _strip_prefix(s: str) -> str:
        lower = s.lower()
        for prefix in _ANSWER_PREFIXES:
            if lower.startswith(prefix):
                return s[len(prefix):].strip()
        return s

    # Pass 1: prefix stripping on the full text (handles short verbose outputs)
    text = _strip_prefix(text)

    # Pass 2: last-line extraction for long multi-line outputs
    if len(text) > 150:
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        if lines:
            text = _strip_prefix(lines[-1])

    return text


def _make_initial_state(row: dict, domain: str, difficulty: str) -> AgentState:
    return AgentState(
        question=row["question"],
        domain=domain,
        difficulty=difficulty,
        gold_answer=row["gold_answer"],
        document_ids=row.get("document_ids", []),
        documents=row.get("context", {}),
        plan_steps=[],
        current_solution="",
        iteration=0,
        critic_reviews=[],
        final_answer="",
        total_input_tokens=0,
        total_output_tokens=0,
        total_steps=0,
        retrieved_lessons=[],
    )


def run_cell(
    architecture: str,
    domain: Domain,
    difficulty: Difficulty,
    graph_builder: Callable[[], Any],
    n: int = 30,
    seed: int = 42,
    iteration_cap: int = 2,
    top_k: int = 3,
    use_fallback: bool = False,
    run_id: str | None = None,
    delay_between_calls: float = 0.5,
) -> list[dict]:
    """
    Run one (architecture, domain, difficulty) cell.

    Args:
        architecture: Label string (e.g. "level1", "level2a", "level2b").
        domain: "math" or "hotpotqa".
        difficulty: "easy" or "hard".
        graph_builder: Callable that returns a compiled LangGraph graph.
        n: Number of samples.
        seed: Random seed for dataset sampling.
        iteration_cap: Max solver-critic iterations (passed to graph_builder for L2B).
        use_fallback: Use offline fallback data.
        run_id: Optional identifier for this experimental run.
        delay_between_calls: Seconds to wait between API calls to avoid rate limits.

    Returns:
        List of result dicts (one per row).
    """
    if run_id is None:
        k_suffix = f"_k{top_k}" if top_k != 3 else ""
        run_id = f"{architecture}_{domain}_{difficulty}{k_suffix}_cap{iteration_cap}"

    output_path = RESULTS_DIR / f"{run_id}.csv"
    rows = load_dataset(domain, difficulty, n=n, seed=seed, use_fallback=use_fallback)
    graph = graph_builder()

    results = []

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()

        for row in rows:
            record: dict = {
                "run_id": run_id,
                "architecture": architecture,
                "domain": domain,
                "difficulty": difficulty,
                "iteration_cap": iteration_cap,
                "id": row["id"],
                "question": row["question"],
                "gold_answer": row["gold_answer"],
                "raw_prediction": "",
                "prediction": "",
                "exact_match": 0,
                "f1": "",
                "total_input_tokens": 0,
                "total_output_tokens": 0,
                "total_steps": 0,
                "elapsed_seconds": 0.0,
                "top_k": top_k,
                "error": "",
            }

            t0 = time.time()
            try:
                initial_state = _make_initial_state(row, domain, difficulty)
                final_state = graph.invoke(initial_state)
                elapsed = round(time.time() - t0, 2)

                raw_prediction = final_state.get("final_answer", "")
                prediction = (
                    _clean_hotpotqa_prediction(raw_prediction)
                    if domain == "hotpotqa"
                    else raw_prediction
                )
                scores = score(domain, prediction, row["gold_answer"])

                record["raw_prediction"] = raw_prediction
                record["prediction"] = prediction
                record["exact_match"] = scores.get("exact_match", 0)
                record["f1"] = scores.get("f1", "")
                record["total_input_tokens"] = final_state.get("total_input_tokens", 0)
                record["total_output_tokens"] = final_state.get("total_output_tokens", 0)
                record["total_steps"] = final_state.get("total_steps", 0)
                record["elapsed_seconds"] = elapsed

            except Exception as e:
                record["elapsed_seconds"] = round(time.time() - t0, 2)
                record["error"] = f"{type(e).__name__}: {str(e)[:200]}"
                traceback.print_exc()

            writer.writerow(record)
            f.flush()
            results.append(record)

            if delay_between_calls > 0:
                time.sleep(delay_between_calls)

    print(f"[runner] Finished {run_id}: {len(results)} rows -> {output_path}")
    return results
