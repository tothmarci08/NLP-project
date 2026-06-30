"""
Shared LangGraph state schema used by all architecture levels.

Design principle: keep fields minimal to prevent token explosion.
- Carry document_ids (not raw text) to reference HotpotQA context.
- Use an annotated reducer list for critic_reviews to accumulate without overwriting.
- Track token usage and step count here so the runner can read them at END.
"""

from typing import Annotated, Any, Optional
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages

from src.schemas import CriticReview


def _append_reviews(existing: list, new: list) -> list:
    """Reducer: append new critic reviews to the accumulated list."""
    return existing + new


class AgentState(TypedDict):
    # --- Input fields (set once at graph entry) ---
    question: str
    domain: str                          # "math" or "hotpotqa"
    difficulty: str                      # "easy" or "hard"
    gold_answer: str
    document_ids: list[str]             # HotpotQA: IDs of candidate paragraphs
    documents: dict[str, str]           # HotpotQA: id -> paragraph text (loaded once)

    # --- Planning (L2A) ---
    plan_steps: list[str]               # Planner output

    # --- Solving ---
    current_solution: str               # Latest solver/executor draft
    iteration: int                      # How many solver-critic loops completed

    # --- Critic (L2B) ---
    critic_reviews: Annotated[list[CriticReview], _append_reviews]

    # --- Final output ---
    final_answer: str

    # --- Efficiency tracking ---
    total_input_tokens: int
    total_output_tokens: int
    total_steps: int

    # --- L3: trajectory memory (optional) ---
    retrieved_lessons: list[str]        # Lessons retrieved from trajectory cache
