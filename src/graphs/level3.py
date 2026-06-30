"""
Level 3: Experience-Replaying Solver-Critic with Local Trajectory Memory.

Graph:
  memory_retriever --> solver --> critic --> (conditional)
                           ^                    |-- "retry" --> solver
                           |                    |-- "end"  --> finalize --> (conditional)
                           └────────────────────                                |-- "write" --> memory_builder --> END
                                                                                |-- "skip"  --> END

Four nodes:
1. memory_retriever  Loads trajectory_cache.json, filters by domain, asks the LLM to
                     select relevant past-failure lessons, injects them into graph state.
2. solver            Same as L2B. Already reads retrieved_lessons from state and injects
                     them into the prompt (Adaptive Strategy).
3. critic            Same as L2B.
4. memory_builder    Activated ONLY when a critic-triggered correction succeeded (at
                     least one retry AND final critic verdict == "correct"). Summarises
                     the failure mode into a compact lesson and appends it to the cache
                     (Memory Summarisation + Experience Replay).

The cache persists across questions within a run, so early questions in a run receive no
lessons (cold start) while later ones accumulate them. This is the intended behaviour.
"""

import json
from pathlib import Path

from langgraph.graph import StateGraph, END

from src.state import AgentState
from src.llm_client import call_llm
from src.prompts import memory_retriever_prompt, memory_builder_prompt

# Reuse the solver/critic/router/finalize from Level 2B — they are identical.
# Level 2B's solver already reads retrieved_lessons from state, so lesson injection
# is already wired; the memory_retriever just has to populate that field.
from src.graphs.level2b import (
    _make_solver_node,
    _make_critic_node,
    _make_router,
    _finalize_node,
)

CACHE_PATH = Path(__file__).resolve().parent.parent.parent / "trajectory_cache.json"


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _load_cache() -> list[dict]:
    """Load lessons from disk. Returns [] on missing file or parse error."""
    if not CACHE_PATH.exists():
        return []
    try:
        with open(CACHE_PATH, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return []


def _save_lesson(lesson: dict) -> None:
    """Append one lesson to the cache file atomically."""
    cache = _load_cache()
    cache.append(lesson)
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Node 1: Memory Retriever
# ---------------------------------------------------------------------------

def _memory_retriever_node(state: AgentState) -> dict:
    """
    Load past failure lessons for the current domain and select relevant ones.

    If the cache is empty for this domain (cold start), returns immediately with
    retrieved_lessons=[] and consumes zero tokens.
    Otherwise calls the LLM to pick which cached lessons apply to this question.
    """
    domain = state["domain"]
    cache = _load_cache()
    domain_lessons = [l for l in cache if l.get("domain") == domain]

    if not domain_lessons:
        # Cold start — no lessons accumulated yet, skip the LLM call
        return {"retrieved_lessons": []}

    prompt = memory_retriever_prompt(question=state["question"], lessons=domain_lessons)
    if not prompt:
        return {"retrieved_lessons": []}

    text, usage = call_llm(prompt=prompt, role="memory_retriever")

    # Parse the JSON list of indices the LLM returned
    try:
        indices = json.loads(text.strip())
        if not isinstance(indices, list):
            indices = []
    except (json.JSONDecodeError, ValueError):
        indices = []

    selected = [
        domain_lessons[i]["summary"]
        for i in indices
        if isinstance(i, int) and 0 <= i < len(domain_lessons)
    ]

    return {
        "retrieved_lessons": selected,
        "total_input_tokens": state.get("total_input_tokens", 0) + usage["input_tokens"],
        "total_output_tokens": state.get("total_output_tokens", 0) + usage["output_tokens"],
        "total_steps": state.get("total_steps", 0) + 1,
    }


# ---------------------------------------------------------------------------
# Node 4: Memory Builder
# ---------------------------------------------------------------------------

def _memory_builder_node(state: AgentState) -> dict:
    """
    Summarise the just-completed failure-and-fix trajectory into a compact lesson.

    Only activated when _memory_route returns "write", meaning:
    - At least one critic review said needs_fix (a retry happened), AND
    - The final critic review said correct (the correction succeeded).
    """
    domain = state["domain"]

    # Find the first review that flagged an error — that is the failure to learn from
    correction_review = next(
        (r for r in state.get("critic_reviews", []) if r.needs_retry),
        None,
    )
    if correction_review is None:
        return {}

    # Format the failure context for the summariser
    critic_feedback_parts = []
    if correction_review.errors_found:
        critic_feedback_parts.append("Errors: " + "; ".join(correction_review.errors_found))
    if correction_review.fix_instructions:
        critic_feedback_parts.append("Fix applied: " + correction_review.fix_instructions)
    critic_feedback = "\n".join(critic_feedback_parts)

    prompt = memory_builder_prompt(
        question=state["question"],
        solution="[initial draft — see critic errors below]",
        critic_feedback=critic_feedback,
        corrected_solution=state.get("current_solution", ""),
    )

    lesson_text, usage = call_llm(prompt=prompt, role="memory_builder")

    lesson = {
        "domain": domain,
        "difficulty": state.get("difficulty", ""),
        "question_snippet": state["question"][:120],
        "summary": lesson_text.strip(),
    }
    _save_lesson(lesson)

    return {
        "total_input_tokens": state.get("total_input_tokens", 0) + usage["input_tokens"],
        "total_output_tokens": state.get("total_output_tokens", 0) + usage["output_tokens"],
        "total_steps": state.get("total_steps", 0) + 1,
    }


# ---------------------------------------------------------------------------
# Conditional edge after finalize: should we write a memory?
# ---------------------------------------------------------------------------

def _memory_route(state: AgentState) -> str:
    """
    Return "write" only when:
    - At least one earlier critic review flagged an error (a retry happened), AND
    - The final critic review confirmed the correction succeeded.

    This avoids writing lessons for cap-exhausted runs where the error was never fixed.
    """
    reviews = state.get("critic_reviews", [])
    if len(reviews) < 2:
        # Only one critic call means no retry happened
        return "skip"

    had_error = any(r.needs_retry for r in reviews[:-1])
    final_correct = reviews[-1].verdict == "correct"

    if had_error and final_correct:
        return "write"
    return "skip"


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_graph(iteration_cap: int = 2, top_k: int = 3):
    """
    Build and compile the Level 3 graph.

    Args:
        iteration_cap: Maximum solver-critic loops (same sweep parameter as L2B).
        top_k: TF-IDF retrieval top-k for HotpotQA (passed to solver and critic).
    """
    graph = StateGraph(AgentState)

    graph.add_node("memory_retriever", _memory_retriever_node)
    graph.add_node("solver", _make_solver_node(top_k))
    graph.add_node("critic", _make_critic_node(top_k))
    graph.add_node("finalize", _finalize_node)
    graph.add_node("memory_builder", _memory_builder_node)

    graph.set_entry_point("memory_retriever")
    graph.add_edge("memory_retriever", "solver")
    graph.add_edge("solver", "critic")

    router = _make_router(iteration_cap)
    graph.add_conditional_edges(
        "critic",
        router,
        {"retry": "solver", "end": "finalize"},
    )

    graph.add_conditional_edges(
        "finalize",
        _memory_route,
        {"write": "memory_builder", "skip": END},
    )

    graph.add_edge("memory_builder", END)

    return graph.compile()
